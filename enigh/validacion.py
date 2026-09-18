"""Validación de los datos procesados contra cifras de control.

Este módulo responde a la pregunta "¿podemos confiar en lo que pinta el tablero?".
Verifica tres familias de comprobaciones:

1. **Cifras de control** contra valores de referencia declarados en
   ``config/validacion.csv`` (conteos de hogares, número de entidades, suma de
   participaciones).
2. **Invariantes internas** que deben cumplirse siempre, como la monotonicidad
   del ingreso por decil o el rango del Gini.
3. **Coherencia entre archivos**: que los agregados sean consistentes con los
   microdatos de los que salieron.

Uso::

    python -m enigh.validacion              # valida los años procesados
    python -m enigh.validacion --anios 2024 --verboso
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import CLAVE_SONORA, NOMBRE_SONORA, config, limpieza

log = logging.getLogger("enigh.validacion")

ARCHIVO_CONTROLES = config.DIR_CONFIG / "validacion.csv"


@dataclass
class Comprobacion:
    """Resultado de una comprobación individual."""

    nombre: str
    paso: bool
    detalle: str
    valor: float | None = None
    esperado: float | None = None

    @property
    def simbolo(self) -> str:
        return "OK  " if self.paso else "FALLA"


@dataclass
class ReporteValidacion:
    """Conjunto de comprobaciones de un año."""

    anio: int
    comprobaciones: list[Comprobacion] = field(default_factory=list)

    @property
    def fallidas(self) -> list[Comprobacion]:
        return [c for c in self.comprobaciones if not c.paso]

    @property
    def paso(self) -> bool:
        return not self.fallidas

    def agregar(self, comprobacion: Comprobacion) -> None:
        self.comprobaciones.append(comprobacion)
        nivel = logging.INFO if comprobacion.paso else logging.ERROR
        log.log(
            nivel, "  [%s] %s: %s", comprobacion.simbolo, comprobacion.nombre,
            comprobacion.detalle,
        )


def leer_controles(ruta: Path | None = None) -> list[dict[str, str]]:
    """Lee la tabla de cifras de control."""
    ruta = ruta or ARCHIVO_CONTROLES
    if not ruta.exists():
        log.warning("No existe %s; se omiten las cifras de control.", ruta)
        return []
    with ruta.open(encoding="utf-8-sig", newline="") as fh:
        return [fila for fila in csv.DictReader(fh) if fila.get("anio")]


def _comparar_con_control(
    reporte: ReporteValidacion,
    nombre: str,
    valor: float,
    esperado: float,
    tolerancia: float,
) -> None:
    """Comprueba un valor contra su referencia con tolerancia relativa."""
    if esperado == 0:
        paso = valor == 0
        desviacion = 0.0 if paso else math.inf
    else:
        desviacion = abs(valor - esperado) / abs(esperado)
        paso = desviacion <= tolerancia

    reporte.agregar(
        Comprobacion(
            nombre=nombre,
            paso=paso,
            detalle=(
                f"obtenido {valor:,.2f} vs esperado {esperado:,.2f} "
                f"(desviación {desviacion:.2%}, tolerancia {tolerancia:.1%})"
            ),
            valor=valor,
            esperado=esperado,
        )
    )


def _metricas_de_hogares(hogares: pd.DataFrame) -> dict[str, float]:
    """Cifras derivadas de los microdatos que se pueden comparar."""
    sonora = hogares[hogares["entidad"] == CLAVE_SONORA]
    return {
        "nacional_hogares_expandidos": float(hogares["factor"].sum()),
        "sonora_hogares_expandidos": float(sonora["factor"].sum()),
        "entidades_distintas": float(hogares["entidad"].nunique()),
        "nacional_hogares_muestra": float(len(hogares)),
        "sonora_hogares_muestra": float(len(sonora)),
    }


def validar_anio(anio: int, *, controles: list[dict[str, str]] | None = None) -> ReporteValidacion:
    """Valida un año procesado: controles, invariantes y coherencia."""
    reporte = ReporteValidacion(anio=anio)
    controles = controles if controles is not None else leer_controles()

    log.info("=== Validando %s ===", anio)

    try:
        hogares = limpieza.cargar_hogares(anio)
    except FileNotFoundError as exc:
        reporte.agregar(
            Comprobacion(
                nombre="datos disponibles", paso=False,
                detalle=f"No hay datos procesados para {anio}: {exc}",
            )
        )
        return reporte

    metricas = _metricas_de_hogares(hogares)

    # ---- 1. Cifras de control declaradas ----
    for fila in controles:
        if int(fila["anio"]) != anio:
            continue
        nombre = fila["ambito"]
        if nombre not in metricas:
            log.debug("Control %s sin métrica asociada; se omite.", nombre)
            continue
        _comparar_con_control(
            reporte,
            nombre,
            metricas[nombre],
            float(fila["valor_esperado"]),
            float(fila["tolerancia_relativa"]),
        )

    # ---- 2. Invariantes internas ----
    deciles = pd.read_parquet(config.ruta_agregados(anio, "deciles_sonora"))
    deciles_nac = pd.read_parquet(config.ruta_agregados(anio, "deciles_nacionales"))

    # Todos los deciles poblados.
    deciles_con_hogares = int((deciles["n_hogares"] > 0).sum())
    _comparar_con_control(
        reporte, "deciles_con_hogares", deciles_con_hogares, 10, 0.0
    )

    # La participación de los deciles debe sumar 100 %.
    suma_participaciones = float(deciles["participacion_ingreso"].sum())
    _comparar_con_control(
        reporte, "participacion_total_deciles", suma_participaciones, 100.0, 1e-3
    )

    # Monotonicidad: el ingreso promedio debe crecer de decil en decil.
    promedios = deciles.sort_values("decil")["ing_pc"].to_numpy(dtype=float)
    pasos_crecientes = bool(np.all(np.diff(promedios) > 0))
    reporte.agregar(
        Comprobacion(
            nombre="monotonicidad_ingreso_por_decil",
            paso=pasos_crecientes,
            detalle=(
                "el ingreso promedio crece de D1 a D10"
                if pasos_crecientes
                else f"no es monótono: {np.round(promedios, 0).tolist()}"
            ),
        )
    )

    # Gini de toda la distribución. OJO: en las tablas de deciles el campo
    # ``gini_dentro_del_decil`` es la dispersión *interna* de cada decil (~0.17) y
    # NO es el Gini de la distribución completa (~0.45). Se lee la columna global
    # precisamente para no confundir una con otra; si no existiera, se recalcula
    # desde los microdatos.
    from .metricas import gini as calcular_gini

    if "gini_global" in deciles_nac.columns and not deciles_nac.empty:
        gini_nacional = float(deciles_nac["gini_global"].iloc[0])
    else:
        gini_nacional = calcular_gini(hogares["ing_pc"], hogares["factor"])

    reporte.agregar(
        Comprobacion(
            nombre="gini_nacional_en_rango",
            paso=bool(np.isfinite(gini_nacional) and 0 <= gini_nacional <= 1),
            detalle=(
                f"Gini nacional = {gini_nacional:.4f} (debe estar en [0, 1]); "
                "medido sobre la distribución completa"
            ),
            valor=gini_nacional,
        )
    )

    # El Gini intra-decil debe ser menor que el global. Si el global resultara
    # igual o menor, casi siempre significa que se leyó la columna equivocada.
    if "gini_dentro_del_decil" in deciles.columns and not deciles.empty:
        gini_intra_max = float(deciles["gini_dentro_del_decil"].max())
        reporte.agregar(
            Comprobacion(
                nombre="gini_decil_no_confundido_con_global",
                paso=bool(
                    np.isfinite(gini_nacional) and gini_intra_max < gini_nacional
                ),
                detalle=(
                    f"máximo Gini intra-decil = {gini_intra_max:.4f} < Gini global "
                    f"= {gini_nacional:.4f}"
                ),
            )
        )

    # Ningún decil con ingresos negativos.
    negativos = int((deciles["ing_pc"] < 0).sum())
    reporte.agregar(
        Comprobacion(
            nombre="sin_ingresos_negativos",
            paso=negativos == 0,
            detalle=f"{negativos} deciles con ingreso per cápita negativo",
        )
    )

    # Celdas con muestra insuficiente deben estar marcadas, no ocultas.
    if "ing_pc_fiable" in deciles.columns:
        no_fiables = int((~deciles["ing_pc_fiable"].fillna(False)).sum())
        reporte.agregar(
            Comprobacion(
                nombre="fiabilidad_reportada",
                paso=True,
                detalle=(
                    f"{no_fiables} de 10 deciles marcados como no fiables "
                    f"(CV > {config.cargar_configuracion().umbrales.cv_no_fiable:.0%})"
                    if no_fiables
                    else "los 10 deciles son fiables"
                ),
            )
        )

    # Ingreso per cápita siempre menor o igual que el ingreso del hogar
    # (el hogar tiene al menos un integrante).
    inconsistencia = int((hogares["ing_pc"] > hogares["ing_cor"] + 0.01).sum())
    reporte.agregar(
        Comprobacion(
            nombre="ing_pc_no_mayor_que_ing_cor",
            paso=inconsistencia == 0,
            detalle=f"{inconsistencia} hogares donde el ingreso per cápita excede el del hogar",
        )
    )

    # Adulto equivalente acotado por el per cápita (ver _adultos_equivalentes).
    con_denominador_valido = hogares[hogares["ing_ae"].notna() & hogares["ing_pc"].notna()]
    violaciones = int((con_denominador_valido["ing_ae"] < con_denominador_valido["ing_pc"] - 0.01).sum())
    reporte.agregar(
        Comprobacion(
            nombre="adulto_equivalente_acotado",
            paso=violaciones == 0,
            detalle=(
                f"{violaciones} hogares donde el ingreso por adulto equivalente "
                f"es menor que el per cápita (la escala debe acotarse a tot_integ)"
            ),
        )
    )

    # ---- 3. Coherencia entre archivos ----
    entidades = pd.read_parquet(config.ruta_agregados(anio, "entidades"))
    sonora = entidades[entidades["entidad"] == CLAVE_SONORA]
    if not sonora.empty:
        expandidos_agregado = float(sonora["hogares_expandidos"].iloc[0])
        _comparar_con_control(
            reporte,
            "coherencia_sonora_entre_archivos",
            expandidos_agregado,
            metricas["sonora_hogares_expandidos"],
            1e-9,
        )
    else:
        reporte.agregar(
            Comprobacion(
                nombre="coherencia_sonora_entre_archivos", paso=False,
                detalle=f"{NOMBRE_SONORA} no aparece en el agregado por entidad",
            )
        )

    return reporte


def validar_todo(anios: list[int] | None = None) -> list[ReporteValidacion]:
    """Valida varios años y devuelve sus reportes."""
    if anios is None:
        directorio = config.DIR_PROCESSED / "hogares"
        anios = sorted(
            int(ruta.name.split("=")[1])
            for ruta in directorio.glob("ano=*")
            if ruta.name.split("=")[1].isdigit()
        )
    controles = leer_controles()
    return [validar_anio(anio, controles=controles) for anio in anios]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m enigh.validacion",
        description="Valida los datos procesados contra cifras de control.",
    )
    parser.add_argument("--anios", nargs="+", type=int, default=None)
    parser.add_argument("--verboso", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verboso else logging.INFO,
        format="%(message)s",
    )

    reportes = validar_todo(args.anios)
    if not reportes:
        log.error("No hay años procesados. Ejecute primero: python -m enigh.pipeline")
        return 1

    total = sum(len(r.comprobaciones) for r in reportes)
    fallidas = sum(len(r.fallidas) for r in reportes)

    print()
    print("=" * 68)
    print(f"Validación: {total} comprobaciones, {fallidas} fallidas")
    for reporte in reportes:
        estado = "PASA" if reporte.paso else "FALLA"
        print(f"  {reporte.anio}: {estado} ({len(reporte.fallidas)} fallas)")
        for comprobacion in reporte.fallidas:
            print(f"      - {comprobacion.nombre}: {comprobacion.detalle}")
    print("=" * 68)

    return 0 if fallidas == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
