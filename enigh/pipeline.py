"""Orquestador del pipeline de datos: descarga, extracción, limpieza y agregados.

Uso típico::

    python -m enigh.pipeline                    # años habilitados, proceso completo
    python -m enigh.pipeline --anios 2024       # un solo año
    python -m enigh.pipeline --solo-descarga    # descarga sin procesar
    python -m enigh.pipeline --sin-bootstrap    # agregados sin intervalos (rápido)

El resultado vive en ``data/processed``::

    hogares/ano=AAAA/hogares.parquet     microdatos curados por hogar
    agregados/ano=AAAA/*.parquet         tablas listas para el tablero
    geografia/*.geojson                  cartografía para los mapas
    manifiesto.json                      trazabilidad de la ejecución
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import CLAVE_SONORA, NOMBRE_SONORA, __version__
from . import config, descarga, geografia, limpieza, metricas
from .estimacion import estadistico_gini, estadistico_media, estadistico_mediana
from .metricas import ResumenDeciles, resumen_deciles

log = logging.getLogger("enigh.pipeline")


# --------------------------------------------------------------------------
# Estructuras de resultado
# --------------------------------------------------------------------------


@dataclass
class ResultadoAnio:
    """Todo lo que el pipeline produce para un año."""

    anio: int
    hogares: pd.DataFrame
    deciles_nacionales: pd.DataFrame
    deciles_sonora: pd.DataFrame
    entidades: pd.DataFrame
    municipios: pd.DataFrame
    resumen_nacional: ResumenDeciles
    resumen_sonora: ResumenDeciles
    reporte: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Pasos del pipeline
# --------------------------------------------------------------------------


def descargar(anios: list[int], *, forzar: bool = False) -> dict[int, Any]:
    """Paso 1: obtiene los ZIP de INEGI."""
    log.info("=== Descarga de microdatos ===")
    return {anio: descarga.descargar_anio(anio, forzar=forzar) for anio in anios}


def procesar_anio(
    anio: int,
    *,
    cfg: config.Configuracion | None = None,
    con_bootstrap: bool = True,
    forzar: bool = False,
) -> ResultadoAnio:
    """Pasos 2 y 3: limpia los microdatos y calcula los agregados del año."""
    cfg = cfg or config.cargar_configuracion()
    inicio = time.monotonic()

    log.info("=== Procesando %s ===", anio)
    hogares, reporte = limpieza.construir_hogares(anio, cfg=cfg)

    # --- Cortes decílicos nacionales: son el patrón de comparación ---
    resumen_nac = resumen_deciles(
        hogares, columna_ingreso="ing_pc", columna_poblacion="tot_integ"
    )
    hogares = hogares.copy()
    hogares["decil_nacional"] = metricas.asignar_deciles(
        hogares["ing_pc"].to_numpy(dtype=float),
        hogares["factor"].to_numpy(dtype=float),
        resumen_nac.cortes,
    )

    # --- Sonora ---
    sonora = hogares[hogares["entidad"] == CLAVE_SONORA].copy()
    if sonora.empty:
        raise RuntimeError(
            f"[{anio}] No hay hogares de {NOMBRE_SONORA} en la muestra; "
            f"revise el catálogo ubica_geo."
        )
    # El nombre oficial del municipio se pega por clave cvegeo (= ``ubica_geo``),
    # nunca por nombre: 22 de los 72 municipios de Sonora llevan acento y las
    # fuentes geográficas no son consistentes entre sí.
    sonora = geografia.agregar_municipio(sonora)

    # Vista principal: hogares de Sonora clasificados con los cortes nacionales.
    resumen_son_nac = resumen_deciles(
        sonora, columna_ingreso="ing_pc", cortes=resumen_nac.cortes,
        columna_poblacion="tot_integ",
    )
    # Vista alterna: deciles propios del estado.
    resumen_son_propios = resumen_deciles(
        sonora, columna_ingreso="ing_pc", columna_poblacion="tot_integ"
    )
    sonora = sonora.copy()
    sonora["decil_sonora"] = metricas.asignar_deciles(
        sonora["ing_pc"].to_numpy(dtype=float),
        sonora["factor"].to_numpy(dtype=float),
        resumen_son_propios.cortes,
    )

    # Deciles alternos por ingreso de adulto equivalente, con cortes nacionales
    # calculados sobre esa misma métrica. El tablero los ofrece cuando el usuario
    # cambia la métrica; reagrupar con la escala de equivalencia cambia quién cae
    # en cada decil (un hogar con muchos menores sube de posición).
    resumen_nac_ae = resumen_deciles(
        hogares, columna_ingreso="ing_ae", columna_poblacion="tot_integ"
    )
    sonora["decil_sonora_ae"] = metricas.asignar_deciles(
        sonora["ing_ae"].to_numpy(dtype=float),
        sonora["factor"].to_numpy(dtype=float),
        resumen_nac_ae.cortes,
    )

    hogares = hogares.merge(
        sonora[["folioviv", "foliohog", "decil_sonora", "decil_sonora_ae"]],
        on=["folioviv", "foliohog"],
        how="left",
    )
    # --- Tablas por decil con intervalos de confianza ---
    perfiles: dict[str, pd.DataFrame] = {
        "deciles_nacionales": _tabla_deciles(
            hogares, columna_grupo="decil_nacional", cfg=cfg, con_bootstrap=con_bootstrap
        ),
        "deciles_sonora": _tabla_deciles(
            sonora, columna_grupo="decil_nacional", cfg=cfg, con_bootstrap=con_bootstrap
        ),
        "deciles_sonora_propios": _tabla_deciles(
            sonora, columna_grupo="decil_sonora", cfg=cfg, con_bootstrap=con_bootstrap
        ),
    }

    # --- Agregados por entidad federativa (mapa nacional) ---
    entidades = _agregados_entidad(hogares, cfg=cfg, con_bootstrap=con_bootstrap)

    # --- Agregados municipales de Sonora (mapa municipal) ---
    perfiles["municipios_sonora"] = _agregados_municipio(
        sonora, cfg=cfg, con_bootstrap=con_bootstrap
    )

    # --- Indicadores de bienestar por decil en Sonora ---
    from . import bienestar as mod_bienestar

    perfil_bienestar, omitidos_bienestar, avisos = mod_bienestar.perfil_por_decil(
        sonora,
        grupo="decil_nacional",
        cfg=cfg,
        replicas=cfg.umbrales.replicas_bootstrap if con_bootstrap else 0,
    )
    perfiles["bienestar_sonora"] = perfil_bienestar

    reporte["omitidos_bienestar"] = omitidos_bienestar
    reporte["avisos_bienestar"] = avisos[:80]
    reporte["segundos_proceso"] = round(time.monotonic() - inicio, 1)

    return ResultadoAnio(
        anio=anio,
        hogares=hogares,
        deciles_nacionales=perfiles["deciles_nacionales"],
        deciles_sonora=perfiles["deciles_sonora"],
        entidades=entidades,
        municipios=perfiles["municipios_sonora"],
        resumen_nacional=resumen_nac,
        resumen_sonora=resumen_son_nac,
        reporte=reporte,
    )


# Estadísticos globales del conjunto, estimados una sola vez sobre todos los
# hogares (no por grupo). Viven aparte de ESTADISTICOS_DECIL a propósito: el
# Gini de un decil es una medida *intra*-decil, típicamente cercana a 0.17, y
# confundirla con el Gini de la distribución completa (cercano a 0.45) daría una
# lectura totalmente equivocada de la desigualdad.
ESTADISTICOS_GLOBALES = {
    "gini_global": estadistico_gini("ing_pc"),
    "ing_pc_media_global": estadistico_media("ing_pc"),
    "ing_pc_mediana_global": estadistico_mediana("ing_pc"),
}

# Estadísticos que se calculan para cada decil.
ESTADISTICOS_DECIL = {
    "ing_pc": estadistico_media("ing_pc"),
    "ing_pc_mediana": estadistico_mediana("ing_pc"),
    "ing_cor": estadistico_media("ing_cor"),
    # Gini DENTRO del decil: mide la dispersión interna de cada grupo, no la
    # desigualdad de todo Sonora. El Gini de toda la distribución está en
    # ``gini_global``, en las tablas de ``serie_desigualdad``.
    "gini_dentro_del_decil": estadistico_gini("ing_pc"),
}

ESTADISTICOS_ENTIDAD = {
    "ing_pc": estadistico_media("ing_pc"),
    "ing_cor": estadistico_media("ing_cor"),
    # Dentro de la entidad sí es el Gini de la entidad completa.
    "gini": estadistico_gini("ing_pc"),
}


def _columnas_globales(datos: pd.DataFrame) -> dict[str, float]:
    """Estadísticos calculados una vez sobre todo el conjunto."""
    return {
        nombre: float(estadistico(datos))
        for nombre, estadistico in ESTADISTICOS_GLOBALES.items()
    }


def _tabla_deciles(
    datos: pd.DataFrame,
    *,
    columna_grupo: str,
    cfg: config.Configuracion,
    con_bootstrap: bool,
) -> pd.DataFrame:
    """Resumen por decil: media, mediana y participación del ingreso.

    Cada fila incluye además las columnas globales (``gini_global``,
    ``ing_pc_media_global``, ``ing_pc_mediana_global``) repetidas a propósito: son
    las que el tablero y la validación deben leer para hablar de "la desigualdad
    de Sonora". ``gini_dentro_del_decil`` es otra cosa, la dispersión interna de
    cada grupo.

    Con ``con_bootstrap`` las columnas ``*_ic_inf``/``*_ic_sup``/``*_cv``/
    ``*_fiable`` acompañan a cada estadístico.
    """
    if datos.empty:
        return pd.DataFrame()

    globales = _columnas_globales(datos)

    filas: list[dict[str, Any]] = []
    for decil, sub in datos.groupby(columna_grupo, dropna=False, sort=True):
        fila: dict[str, Any] = {
            "decil": int(decil) if pd.notna(decil) else -1,
            "n_hogares": int(len(sub)),
            "hogares_expandidos": float(sub["factor"].sum()),
            **globales,
        }
        for nombre, estadistico in ESTADISTICOS_DECIL.items():
            fila[nombre] = float(estadistico(sub))
        filas.append(fila)

    tabla = pd.DataFrame(filas)

    if con_bootstrap:
        from .estimacion import estimar_por_grupo

        resultado = estimar_por_grupo(
            datos, columna_grupo, ESTADISTICOS_DECIL, cfg=cfg
        )
        # ``estimar_por_grupo`` devuelve la clave del grupo en la columna
        # ``grupo`` sin importar cómo se llame la columna original; se renombra a
        # una clave interna fija para que el merge no dependa del nombre.
        con_ic = resultado.tabla.rename(columns={"grupo": "clave_grupo"})
        columnas_ic = [
            c for c in con_ic.columns if c not in ("clave_grupo", "n_hogares")
        ]
        con_ic["clave_grupo"] = con_ic["clave_grupo"].astype(int)
        tabla = tabla.merge(
            con_ic[["clave_grupo"] + columnas_ic],
            left_on="decil",
            right_on="clave_grupo",
            how="left",
        ).drop(columns="clave_grupo", errors="ignore")

    # Participación de cada decil en el ingreso total, recalculada por suma
    # directa para que las participaciones sumen 100 % sin sesgo de redondeo.
    total = float((datos["ing_pc"] * datos["factor"]).sum())
    sumas = (
        datos.assign(_aporte=datos["ing_pc"] * datos["factor"])
        .groupby(columna_grupo, sort=True)["_aporte"]
        .sum()
    )
    mapa_sumas = {int(k) if pd.notna(k) else -1: float(v) for k, v in sumas.items()}
    tabla["ingreso_ponderado"] = tabla["decil"].map(mapa_sumas)
    tabla["participacion_ingreso"] = (
        tabla["ingreso_ponderado"] / total * 100 if total > 0 else np.nan
    )
    tabla["decil"] = tabla["decil"].astype(int)
    return tabla


def _agregados_entidad(
    hogares: pd.DataFrame, *, cfg: config.Configuracion, con_bootstrap: bool
) -> pd.DataFrame:
    """Agregados por entidad federativa, para el mapa nacional."""
    filas: list[dict[str, Any]] = []
    for entidad, sub in hogares.groupby("entidad", sort=True):
        nombre = (
            sub["desc_ent"].dropna().iloc[0]
            if "desc_ent" in sub.columns and sub["desc_ent"].notna().any()
            else entidad
        )
        fila: dict[str, Any] = {
            "entidad": entidad,
            "nom_entidad": nombre,
            "n_hogares": int(len(sub)),
            "hogares_expandidos": float(sub["factor"].sum()),
            "es_sonora": bool(entidad == CLAVE_SONORA),
        }
        for clave, estadistico in ESTADISTICOS_ENTIDAD.items():
            fila[clave] = float(estadistico(sub))
        filas.append(fila)

    tabla = pd.DataFrame(filas)

    if con_bootstrap:
        from .estimacion import estimar_por_grupo

        resultado = estimar_por_grupo(hogares, "entidad", ESTADISTICOS_ENTIDAD, cfg=cfg)
        con_ic = resultado.tabla.rename(columns={"grupo": "entidad"})
        columnas_ic = [
            c for c in con_ic.columns if c not in ("entidad", "n_hogares")
        ]
        tabla = tabla.merge(
            con_ic[["entidad"] + columnas_ic],
            on="entidad",
            how="left",
        )

    return tabla


def _agregados_municipio(
    sonora: pd.DataFrame, *, cfg: config.Configuracion, con_bootstrap: bool
) -> pd.DataFrame:
    """Agregados por municipio de Sonora, con supresión de celdas frágiles.

    La ENIGH está diseñada para estimaciones estatales y nacionales, no
    municipales. Por eso cada fila lleva su ``n_hogares``, su ``n_upm`` y una
    bandera ``fiable``; el tablero solo publica el promedio de las filas fiables.
    """
    if sonora.empty:
        return pd.DataFrame()

    columna_nombre = "nom_municipio" if "nom_municipio" in sonora.columns else "municipio"
    minimo = cfg.umbrales.hogares_min_municipio

    filas: list[dict[str, Any]] = []
    for municipio, sub in sonora.groupby(columna_nombre, dropna=False, sort=True):
        n_hogares = int(len(sub))
        n_upm = int(sub["upm"].nunique()) if "upm" in sub.columns else 0
        fila: dict[str, Any] = {
            "nom_municipio": str(municipio),
            "clave_municipio": str(sub["municipio"].iloc[0]) if "municipio" in sub.columns else "",
            # Clave INEGI de 5 dígitos. Se guarda aquí explícitamente porque es la
            # única llave con la que el coroplético municipal debe cruzar el
            # GeoJSON: los nombres no son consistentes entre fuentes.
            "clave_cvegeo": str(sub["ubica_geo"].iloc[0])
            if "ubica_geo" in sub.columns
            else f"{CLAVE_SONORA}{sub['municipio'].iloc[0]}"
            if "municipio" in sub.columns
            else "",
            "n_hogares": n_hogares,
            "n_upm": n_upm,
            "hogares_expandidos": float(sub["factor"].sum()),
            "ing_pc": float(estadistico_media("ing_pc")(sub)),
            "ing_cor": float(estadistico_media("ing_cor")(sub)),
            "gini": float(estadistico_gini("ing_pc")(sub)),
            "fiable": bool(n_hogares >= minimo),
        }
        filas.append(fila)

    tabla = pd.DataFrame(filas)

    # Intervalos de confianza solo donde el bootstrap tiene sentido: celdas con
    # muestra suficiente y al menos dos UPM.
    if con_bootstrap and not tabla.empty:
        aptos = tabla[tabla["fiable"]]["nom_municipio"].tolist()
        if aptos:
            sub = sonora[sonora[columna_nombre].isin(aptos)]
            from .estimacion import estimar_por_grupo

            resultado = estimar_por_grupo(
                sub, columna_nombre, ESTADISTICOS_ENTIDAD, cfg=cfg
            )
            con_ic = resultado.tabla.rename(columns={"grupo": "nom_municipio"})
            con_ic["nom_municipio"] = con_ic["nom_municipio"].astype(str)
            columnas_ic = [
                c for c in con_ic.columns if c not in ("nom_municipio", "n_hogares")
            ]
            tabla = tabla.merge(con_ic[["nom_municipio"] + columnas_ic],
                                on="nom_municipio", how="left")

    return tabla.sort_values("ing_pc", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Persistencia
# --------------------------------------------------------------------------


def guardar_anio(resultado: ResultadoAnio) -> list[Path]:
    """Escribe en disco todos los productos de un año."""
    anio = resultado.anio
    escritos: list[Path] = []

    ruta_hogares = config.ruta_hogares(anio)
    ruta_hogares.parent.mkdir(parents=True, exist_ok=True)
    resultado.hogares.to_parquet(ruta_hogares, index=False, compression="zstd")
    escritos.append(ruta_hogares)

    for nombre, tabla in (
        ("deciles_nacionales", resultado.deciles_nacionales),
        ("deciles_sonora", resultado.deciles_sonora),
        ("entidades", resultado.entidades),
        ("municipios_sonora", resultado.municipios),
    ):
        ruta = config.ruta_agregados(anio, nombre)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        tabla.to_parquet(ruta, index=False, compression="zstd")
        escritos.append(ruta)

    for escritura in escritos:
        log.info(
            "  escrito %s (%.1f KB)",
            escritura.relative_to(config.RAIZ),
            escritura.stat().st_size / 1024,
        )
    return escritos


def guardar_manifiesto(
    resultados: list[ResultadoAnio], *, segundos: float, args: dict[str, Any]
) -> Path:
    """Registra URL, fecha, conteos y versiones para trazabilidad."""
    manifiesto = {
        "version_proyecto": __version__,
        "generado_en": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "segundos_totales": round(segundos, 1),
        "python": sys.version,
        "plataforma": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "argumentos": args,
        "anios": [
            {
                "anio": r.anio,
                "hogares_en_muestra": int(len(r.hogares)),
                "hogares_expandidos": float(r.hogares["factor"].sum()),
                "hogares_sonora_muestra": int(
                    (r.hogares["entidad"] == CLAVE_SONORA).sum()
                ),
                "hogares_sonora_expandidos": float(
                    r.hogares.loc[r.hogares["entidad"] == CLAVE_SONORA, "factor"].sum()
                ),
                "cortes_deciles_nacionales": [
                    round(float(c), 2) for c in r.resumen_nacional.cortes
                ],
                "gini_nacional": round(float(r.resumen_nacional.gini), 4),
                "gini_sonora": round(float(r.resumen_sonora.gini), 4),
                "reporte_limpieza": r.reporte,
            }
            for r in resultados
        ],
    }
    config.ARCHIVO_MANIFIESTO.parent.mkdir(parents=True, exist_ok=True)
    with config.ARCHIVO_MANIFIESTO.open("w", encoding="utf-8") as fh:
        json.dump(manifiesto, fh, ensure_ascii=False, indent=2, default=str)
    log.info("Manifiesto: %s", config.ARCHIVO_MANIFIESTO.relative_to(config.RAIZ))
    return config.ARCHIVO_MANIFIESTO


def _guardar_serie(resultados: list[ResultadoAnio]) -> None:
    """Consolida la serie de tiempo entre años en Parquet."""
    if len(resultados) < 2:
        log.info("Solo un año procesado: no hay serie de tiempo que consolidar.")
        return

    filas_deciles = []
    for r in resultados:
        tabla = r.deciles_sonora.copy()
        tabla["anio"] = r.anio
        filas_deciles.append(tabla)

    serie = pd.concat(filas_deciles, ignore_index=True, sort=False)
    ruta = config.DIR_PROCESSED / "agregados" / "serie_deciles_sonora.parquet"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    serie.to_parquet(ruta, index=False, compression="zstd")
    log.info("  escrito %s (%s filas)", ruta.relative_to(config.RAIZ), len(serie))

    resumen = pd.DataFrame(
        [
            {
                "anio": r.anio,
                "ambito": ambito,
                "gini": resumen.gini,
                "theil": resumen.theil,
                "palma": resumen.palma,
                "participacion_bajo40": resumen.participacion_bajo40,
                "participacion_alto10": resumen.participacion_alto10,
                "n_observaciones": resumen.n_observaciones,
                "poblacion_expandida": resumen.poblacion_expandida,
            }
            for r in resultados
            for ambito, resumen in (
                ("Nacional", r.resumen_nacional),
                ("Sonora", r.resumen_sonora),
            )
        ]
    )
    ruta_resumen = config.DIR_PROCESSED / "agregados" / "serie_desigualdad.parquet"
    ruta_resumen.parent.mkdir(parents=True, exist_ok=True)
    resumen.to_parquet(ruta_resumen, index=False, compression="zstd")
    log.info("  escrito %s (%s filas)", ruta_resumen.relative_to(config.RAIZ), len(resumen))

    # Serie alterna: deciles formados con el ingreso por adulto equivalente.
    # Es una agrupación distinta (la escala de equivalencia pondera por edad), y
    # el tablero la ofrece como vista alterna del selector de métrica. Se guarda
    # para que ese selector funcione también en la serie de tiempo.
    filas_ae = []
    for r in resultados:
        if "decil_sonora_ae" not in r.hogares.columns:
            continue
        sonora = r.hogares[r.hogares["entidad"] == CLAVE_SONORA]
        if sonora.empty:
            continue
        tabla = _tabla_deciles(
            sonora, columna_grupo="decil_sonora_ae",
            cfg=config.cargar_configuracion(), con_bootstrap=False,
        ).copy()
        tabla["anio"] = r.anio
        filas_ae.append(tabla)

    if filas_ae:
        serie_ae = pd.concat(filas_ae, ignore_index=True, sort=False)
        ruta_ae = config.DIR_PROCESSED / "agregados" / "serie_deciles_sonora_ae.parquet"
        serie_ae.to_parquet(ruta_ae, index=False, compression="zstd")
        log.info("  escrito %s (%s filas)", ruta_ae.relative_to(config.RAIZ), len(serie_ae))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _configurar_logging(verboso: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verboso else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m enigh.pipeline",
        description="Descarga y procesa la ENIGH (INEGI) para el tablero de Sonora.",
    )
    parser.add_argument("--anios", nargs="+", type=int, default=None,
                        help="Años a procesar (por omisión, los habilitados).")
    parser.add_argument("--solo-descarga", action="store_true",
                        help="Descarga los ZIP y termina.")
    parser.add_argument("--solo-geografia", action="store_true",
                        help="Solo descarga la cartografía de los mapas.")
    parser.add_argument("--forzar", action="store_true",
                        help="Ignora cachés y vuelve a descargar/procesar.")
    parser.add_argument("--sin-bootstrap", action="store_true",
                        help="Omite los intervalos de confianza (mucho más rápido).")
    parser.add_argument("--replicas", type=int, default=None,
                        help="Réplicas de bootstrap (por omisión, las de config).")
    parser.add_argument("--muestra", type=int, default=None, metavar="N",
                        help="MODO DEMOSTRACIÓN: conserva todos los hogares de "
                             "Sonora y recorta el resto a ~4N por año. Acelera el "
                             "proceso, pero los intervalos de confianza dejan de "
                             "reflejar el diseño completo de la encuesta.")
    parser.add_argument("--verboso", action="store_true", help="Log detallado.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = construir_parser().parse_args(argv)
    _configurar_logging(args.verboso)

    desde = time.monotonic()
    cfg = config.cargar_configuracion()
    anios = args.anios or cfg.anios_habilitados
    desconocidos = [a for a in anios if a not in cfg.anios]
    if desconocidos:
        log.error("Años no declarados en config/fuentes.yaml: %s", desconocidos)
        return 2

    config.asegurar_directorios()
    log.info("Años a procesar: %s", sorted(anios))

    if args.solo_geografia:
        geografia.asegurar_cartografia()
        return 0

    descargar(sorted(anios), forzar=args.forzar)

    if args.solo_descarga:
        log.info("Solo descarga: terminado en %.1f s", time.monotonic() - desde)
        return 0

    geografia.asegurar_cartografia()

    resultados: list[ResultadoAnio] = []
    for anio in sorted(anios):
        resultado = procesar_anio(
            anio, cfg=cfg, con_bootstrap=not args.sin_bootstrap, forzar=args.forzar
        )
        if args.muestra:
            resultado = _recortar_a_muestra(resultado, objetivo=args.muestra)
        guardar_anio(resultado)
        resultados.append(resultado)

    _guardar_serie(resultados)
    manifiesto = guardar_manifiesto(
        resultados, segundos=time.monotonic() - desde, args=vars(args)
    )
    if args.muestra:
        log.warning(
            "MUESTRA REDUCIDA: %s hogares por año. Es un modo de demostración: "
            "los intervalos de confianza NO reflejan el diseño completo de la "
            "encuesta. Vuelva a ejecutar sin --muestra para el análisis real.",
            args.muestra,
        )
        log.info("Manifiesto: %s", manifiesto.relative_to(config.RAIZ))

    log.info("=== Listo en %.1f s ===", time.monotonic() - desde)
    for r in resultados:
        en_sonora = int((r.hogares["entidad"] == CLAVE_SONORA).sum())
        log.info(
            "  %s: %s hogares (%s en %s) · Gini nacional %.3f · Gini Sonora %.3f",
            r.anio, f"{len(r.hogares):,}", f"{en_sonora:,}", NOMBRE_SONORA,
            r.resumen_nacional.gini, r.resumen_sonora.gini,
        )
    return 0


def _recortar_a_muestra(
    resultado: ResultadoAnio, *, objetivo: int, semilla: int = 20240101
) -> ResultadoAnio:
    """Recorta el resultado a una submuestra, para demostración rápida.

    Conserva **todos** los hogares de Sonora (el foco del tablero) y un número
    acotado del resto del país, y **no** recalcula los factores de expansión: eso
    mantendría los totales poblacionales, pero los intervalos de confianza
    quedarían demasiado estrechos, que es justo el error que se quiere evitar. En
    su lugar, el recorte se marca en el manifiesto y el tablero avisa.
    """
    import numpy as np

    hogares = resultado.hogares
    generador = np.random.default_rng(semilla)
    es_sonora = hogares["entidad"] == CLAVE_SONORA

    # Muestra sesgada a propósito hacia Sonora: es el sujeto del análisis.
    sonora = hogares[es_sonora]
    resto = hogares[~es_sonora]
    n_resto = max(objetivo * 4, 2_000)
    if len(resto) > n_resto:
        elegidos = generador.choice(len(resto), size=n_resto, replace=False)
        resto = resto.iloc[np.sort(elegidos)]

    recortado = pd.concat([sonora, resto], ignore_index=True)

    # Los agregados globales se recalculan sobre la submuestra para que sigan
    # siendo internamente consistentes con los microdatos recortados.
    resumen_nac = resumen_deciles(
        recortado, columna_ingreso="ing_pc", columna_poblacion="tot_integ"
    )
    sonora_recortada = recortado[recortado["entidad"] == CLAVE_SONORA]
    resumen_son = resumen_deciles(
        sonora_recortada, columna_ingreso="ing_pc",
        cortes=resumen_nac.cortes, columna_poblacion="tot_integ",
    )

    return ResultadoAnio(
        anio=resultado.anio,
        hogares=recortado,
        deciles_nacionales=_tabla_deciles(
            recortado, columna_grupo="decil_nacional", cfg=config.cargar_configuracion(),
            con_bootstrap=False,
        ),
        deciles_sonora=_tabla_deciles(
            sonora_recortada, columna_grupo="decil_nacional",
            cfg=config.cargar_configuracion(), con_bootstrap=False,
        ),
        entidades=resultado.entidades,
        municipios=resultado.municipios,
        resumen_nacional=resumen_nac,
        resumen_sonora=resumen_son,
        reporte={**resultado.reporte, "muestra_reducida": objetivo},
    )


if __name__ == "__main__":
    raise SystemExit(main())
