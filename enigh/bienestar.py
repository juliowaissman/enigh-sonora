"""Indicadores de bienestar por decil, con intervalos de confianza.

Cada indicador se declara aquí una sola vez, con su dominio, etiqueta, unidad y
la forma de calcularlo. El tablero consume este catálogo, de modo que agregar un
indicador no requiere tocar la interfaz.

Todos los indicadores se calculan **por hogar** y se ponderan con ``factor``. El
denominador excluye los hogares con dato faltante, por lo que el ``n`` de cada
celda se reporta junto con la estimación.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from . import config
from .estimacion import ESTADISTICO, estadistico_gini, estadistico_media, estimar_por_grupo
from .metricas import media_ponderada

log = logging.getLogger(__name__)

FormaEstimador = Literal[
    "media", "proporcion", "proporcion_mayor", "proporcion_menor", "proporcion_en",
    "razon", "gini",
]


@dataclass(frozen=True)
class IndicadorBienestar:
    """Definición declarativa de un indicador de bienestar."""

    clave: str
    etiqueta: str
    dominio: str
    unidad: str
    forma: FormaEstimador
    columna: str | None = None
    numerador: str | None = None
    denominador: str | None = None
    # Puede ser numérico (leña = 1) o una clave de texto ("06" para educa_jefe),
    # porque algunas variables categóricas conservan ceros a la izquierda.
    umbral: float | str | None = None
    descripcion: str = ""
    mayor_es_mejor: bool = True

    def columnas_requeridas(self) -> list[str]:
        return [c for c in (self.columna, self.numerador, self.denominador) if c]


def _comparar(serie: pd.Series, valor: float | str, operador: str) -> pd.Series:
    """Compara una serie contra un umbral numérico o de texto, sin coercionar mal.

    Si el umbral es texto se normaliza la serie a texto; si es numérico se
    coercionan ambos. Coercionar a número un umbral de texto (o al revés) produce
    comparaciones siempre falsas sin lanzar error, que es el peor resultado
    posible en un indicador: un cero silencioso que parece un dato.
    """
    if isinstance(valor, str):
        normalizada = serie.astype("string").str.strip()
        comparaciones = {
            "<": normalizada < valor,
            "<=": normalizada <= valor,
            ">": normalizada > valor,
            ">=": normalizada >= valor,
            "==": normalizada == valor,
        }
        resultado = comparaciones[operador]
    else:
        numerica = pd.to_numeric(serie, errors="coerce")
        comparaciones = {
            "<": numerica < valor,
            "<=": numerica <= valor,
            ">": numerica > valor,
            ">=": numerica >= valor,
            "==": numerica == valor,
        }
        resultado = comparaciones[operador]
    return pd.Series(
        np.where(serie.isna(), np.nan, resultado.astype(float)), index=serie.index
    )


def _estimador_indicador(indicador: IndicadorBienestar) -> ESTADISTICO:
    """Traduce la definición declarativa a una función sobre un DataFrame."""
    peso = "factor"

    if indicador.forma == "media":
        return estadistico_media(indicador.columna, peso)

    if indicador.forma == "proporcion":
        columna = indicador.columna
        valor = indicador.umbral if indicador.umbral is not None else 1

        def _fn(df: pd.DataFrame) -> float:
            return media_ponderada(_comparar(df[columna], valor, "=="), df[peso])

        _fn.__name__ = f"prop_{indicador.clave}"
        return _fn

    if indicador.forma == "proporcion_mayor":
        columna = indicador.columna
        umbral = indicador.umbral if indicador.umbral is not None else 0

        def _fn(df: pd.DataFrame) -> float:
            return media_ponderada(_comparar(df[columna], umbral, ">"), df[peso])

        _fn.__name__ = f"prop_mayor_{indicador.clave}"
        return _fn

    if indicador.forma == "proporcion_menor":
        columna = indicador.columna
        umbral = indicador.umbral if indicador.umbral is not None else 0

        def _fn(df: pd.DataFrame) -> float:
            return media_ponderada(_comparar(df[columna], umbral, "<"), df[peso])

        _fn.__name__ = f"prop_menor_{indicador.clave}"
        return _fn

    if indicador.forma == "proporcion_en":
        columna = indicador.columna
        umbral = indicador.umbral if indicador.umbral is not None else 0

        def _fn(df: pd.DataFrame) -> float:
            # Verdadero para las claves 1..umbral. Funciona igual con claves
            # numéricas (leña = 1, carbón = 2) y con claves de texto con ceros a
            # la izquierda (educa_jefe "01".."06"): en ambos casos el orden
            # lexicográfico coincide con el orden de la clave.
            return media_ponderada(_comparar(df[columna], umbral, "<="), df[peso])

        _fn.__name__ = f"prop_en_{indicador.clave}"
        return _fn

    if indicador.forma == "razon":
        num, den = indicador.numerador, indicador.denominador

        def _fn(df: pd.DataFrame) -> float:
            return media_ponderada(df[num], df[peso]) / media_ponderada(df[den], df[peso])

        _fn.__name__ = f"razon_{indicador.clave}"
        return _fn

    if indicador.forma == "gini":
        return estadistico_gini(indicador.columna, peso)

    raise ValueError(f"Forma de estimador desconocida: {indicador.forma}")


# --------------------------------------------------------------------------
# Catálogo de indicadores
# --------------------------------------------------------------------------

CATALOGO: list[IndicadorBienestar] = [
    # --- Ingreso y desigualdad ---
    IndicadorBienestar(
        clave="ing_cor_promedio", etiqueta="Ingreso corriente promedio por hogar",
        dominio="Ingreso", unidad="pesos por trimestre", forma="media",
        columna="ing_cor", descripcion="Ingreso corriente mensual del hogar, "
        "promedio dentro del decil.",
    ),
    IndicadorBienestar(
        clave="ing_pc_promedio", etiqueta="Ingreso corriente per cápita promedio",
        dominio="Ingreso", unidad="pesos por trimestre", forma="media",
        columna="ing_pc", descripcion="Ingreso corriente dividido entre los "
        "integrantes del hogar.",
    ),
    IndicadorBienestar(
        clave="gini_ing_pc", etiqueta="Gini del ingreso per cápita",
        dominio="Desigualdad", unidad="índice 0-1", forma="gini", columna="ing_pc",
        descripcion="Desigualdad dentro del decil.", mayor_es_mejor=False,
    ),
    IndicadorBienestar(
        clave="share_ingtrab", etiqueta="Participación del ingreso por trabajo",
        dominio="Ingreso", unidad="proporción", forma="media", columna="share_ingtrab",
        descripcion="Fracción del ingreso corriente que proviene del trabajo.",
    ),
    IndicadorBienestar(
        clave="share_transfer", etiqueta="Participación de transferencias",
        dominio="Ingreso", unidad="proporción", forma="media", columna="share_transfer",
        descripcion="Fracción del ingreso corriente que proviene de transferencias "
        "(jubilaciones, becas, remesas, beneficios gubernamentales).",
    ),
    IndicadorBienestar(
        clave="share_remesas", etiqueta="Participación de remesas",
        dominio="Ingreso", unidad="proporción", forma="media", columna="share_remesas",
        descripcion="Fracción del ingreso corriente que proviene de remesas.",
    ),

    # --- Alimentación ---
    IndicadorBienestar(
        clave="prop_gasto_alimentos", etiqueta="Gasto en alimentos / gasto monetario",
        dominio="Alimentación", unidad="proporción", forma="media",
        columna="prop_gasto_alimentos",
        descripcion="Un valor alto indica mayor vulnerabilidad: el hogar destina "
        "más de su gasto a alimentos.", mayor_es_mejor=False,
    ),
    IndicadorBienestar(
        clave="inseg_alimentaria_severa",
        etiqueta="Hogares con hambre o una comida al día",
        dominio="Alimentación", unidad="proporción", forma="proporcion",
        columna="inseg_alimentaria_severa", umbral=1.0,
        descripcion="El hogar reportó que alguien sintió hambre y no comió, o que "
        "comió una sola vez al día (reactivos 7, 8, 14 y 16 de acc_alim). Es el "
        "tramo más grave de la escala y la medida que conviene citar.",
        mayor_es_mejor=False,
    ),
    IndicadorBienestar(
        clave="dificultad_alimentaria",
        etiqueta="Hogares con alguna dificultad alimentaria",
        dominio="Alimentación", unidad="proporción", forma="proporcion",
        columna="dificultad_alimentaria", umbral=1.0,
        descripcion="El hogar reportó al menos una de las 16 situaciones del "
        "cuestionario, incluida la mera preocupación de que la comida se acabe. "
        "Es una medida AMPLIA (ronda el 95 % de los hogares) y NO equivale a la "
        "medición oficial de inseguridad alimentaria del INEGI.",
        mayor_es_mejor=False,
    ),

    # --- Educación ---
    IndicadorBienestar(
        clave="pct_asis_esc", etiqueta="Integrantes que asisten a la escuela",
        dominio="Educación", unidad="proporción", forma="media", columna="pct_asis_esc",
        descripcion="Proporción de integrantes del hogar inscritos en el sistema "
        "educativo.",
    ),
    IndicadorBienestar(
        clave="pct_alfabeta", etiqueta="Integrantes alfabetas",
        dominio="Educación", unidad="proporción", forma="media", columna="pct_alfabeta",
        descripcion="Proporción de integrantes que saben leer y escribir.",
    ),
    IndicadorBienestar(
        clave="pct_jefe_basica_o_menos", etiqueta="Jefatura sin educación media superior",
        dominio="Educación", unidad="proporción", forma="proporcion_en",
        columna="educa_jefe", umbral="06",
        descripcion="La jefatura del hogar no completó educación media superior "
        "(claves de educa_jefe 01 a 06).", mayor_es_mejor=False,
    ),

    # --- Salud ---
    IndicadorBienestar(
        clave="pct_segsoc", etiqueta="Integrantes con seguridad social",
        dominio="Salud", unidad="proporción", forma="media", columna="pct_segsoc",
        descripcion="Proporción de integrantes con cobertura de seguridad social.",
    ),
    IndicadorBienestar(
        clave="pct_atencion_salud", etiqueta="Atención recibida al enfermar",
        dominio="Salud", unidad="proporción", forma="media", columna="pct_atencion_salud",
        descripcion="Entre quienes reportaron enfermedad, proporción que recibió "
        "atención médica.",
    ),
    IndicadorBienestar(
        clave="pct_discapacidad", etiqueta="Integrantes con discapacidad visual",
        dominio="Salud", unidad="proporción", forma="media", columna="pct_discapacidad",
        descripcion="Proporción de integrantes con dificultad para ver.",
        mayor_es_mejor=False,
    ),

    # --- Vivienda y servicios ---
    # Claves verificadas en los catálogos de ENIGH 2024:
    #   agua_ent    1=dentro de la vivienda, 2=solo en el patio, 3=no tiene
    #   drenaje     1..4=conectado a alguna red de desagüe, 5=no tiene
    #   excusado    1=taza de baño, 2=letrina, 3=ninguno
    #   disp_elect  1..4=tiene energía eléctrica, 5=no tiene
    #   mat_pisos   1=tierra
    #   combus      1=leña, 2=carbón
    IndicadorBienestar(
        clave="tiene_agua_entubada", etiqueta="Vivienda con agua entubada",
        dominio="Vivienda", unidad="proporción", forma="proporcion_menor",
        columna="agua_ent", umbral=3.0,
        descripcion="Agua entubada dentro de la vivienda o en el patio.",
    ),
    IndicadorBienestar(
        clave="tiene_drenaje", etiqueta="Vivienda con drenaje",
        dominio="Vivienda", unidad="proporción", forma="proporcion_menor",
        columna="drenaje", umbral=5.0,
        descripcion="Conectada a red pública, fosa séptica o tubería de desagüe.",
    ),
    IndicadorBienestar(
        clave="tiene_excusado", etiqueta="Vivienda con taza de baño",
        dominio="Vivienda", unidad="proporción", forma="proporcion",
        columna="excusado", umbral=1.0,
    ),
    IndicadorBienestar(
        clave="tiene_electricidad", etiqueta="Vivienda con energía eléctrica",
        dominio="Vivienda", unidad="proporción", forma="proporcion_menor",
        columna="disp_elect", umbral=5.0,
        descripcion="Cualquier fuente: red pública, planta particular o panel solar.",
    ),
    IndicadorBienestar(
        clave="hacinamiento", etiqueta="Integrantes por cuarto para dormir",
        dominio="Vivienda", unidad="personas/cuarto", forma="media",
        columna="hacinamiento", descripcion="Más de 2.5 se considera hacinamiento.",
        mayor_es_mejor=False,
    ),
    IndicadorBienestar(
        clave="piso_tierra", etiqueta="Vivienda con piso de tierra",
        dominio="Vivienda", unidad="proporción", forma="proporcion",
        columna="mat_pisos", umbral=1.0, mayor_es_mejor=False,
    ),
    IndicadorBienestar(
        clave="cocina_lena", etiqueta="Cocina con leña o carbón",
        dominio="Vivienda", unidad="proporción", forma="proporcion_en",
        columna="combus", umbral=2.0,
        descripcion="El hogar cocina con leña (clave 1) o carbón (clave 2).",
        mayor_es_mejor=False,
    ),

    # --- Conectividad y bienes ---
    IndicadorBienestar(
        clave="tiene_internet", etiqueta="Hogar con conexión a internet",
        dominio="Conectividad", unidad="proporción", forma="proporcion",
        columna="conex_inte", umbral=1.0,
    ),
    IndicadorBienestar(
        clave="tiene_celular", etiqueta="Hogar con teléfono celular",
        dominio="Conectividad", unidad="proporción", forma="proporcion",
        columna="celular", umbral=1.0,
    ),
    IndicadorBienestar(
        clave="tiene_computadora", etiqueta="Hogar con computadora",
        dominio="Conectividad", unidad="proporción", forma="proporcion_mayor",
        columna="num_compu", umbral=0.0,
    ),
    IndicadorBienestar(
        clave="tiene_automovil", etiqueta="Hogar con automóvil",
        dominio="Bienes", unidad="proporción", forma="proporcion_mayor",
        columna="num_auto", umbral=0.0,
    ),
    IndicadorBienestar(
        clave="tiene_refrigerador", etiqueta="Hogar con refrigerador",
        dominio="Bienes", unidad="proporción", forma="proporcion_mayor",
        columna="num_refri", umbral=0.0,
    ),
    IndicadorBienestar(
        clave="tiene_tarjeta", etiqueta="Hogar con tarjeta de crédito",
        dominio="Bienes", unidad="proporción", forma="proporcion",
        columna="tarjeta", umbral=1.0,
    ),

    # --- Demografía y vulnerabilidad ---
    IndicadorBienestar(
        clave="mujer_jefa", etiqueta="Hogares con jefatura femenina",
        dominio="Demografía", unidad="proporción", forma="proporcion",
        columna="sexo_jefe", umbral="2",
        descripcion="El catálogo de INEGI codifica 1 = hombre y 2 = mujer.",
    ),
    IndicadorBienestar(
        clave="tasa_dependencia", etiqueta="Tasa de dependencia",
        dominio="Demografía", unidad="dependientes/activos", forma="razon",
        numerador="dependientes_calc", denominador="activos_calc",
        descripcion="Integrantes menores de 12 o de 65 y más, por cada integrante "
        "de 12 a 64 años.", mayor_es_mejor=False,
    ),
    IndicadorBienestar(
        clave="integrantes", etiqueta="Integrantes por hogar",
        dominio="Demografía", unidad="personas", forma="media", columna="tot_integ",
    ),
    IndicadorBienestar(
        clave="perceptores_por_hogar", etiqueta="Perceptores de ingreso por hogar",
        dominio="Demografía", unidad="personas", forma="media", columna="percep_ing",
    ),
    IndicadorBienestar(
        clave="pct_etnia", etiqueta="Integrantes que se autoidentifican indígenas",
        dominio="Identidad", unidad="proporción", forma="media", columna="pct_etnia",
    ),
    IndicadorBienestar(
        clave="dep_remesas", etiqueta="Hogares que reciben remesas",
        dominio="Vulnerabilidad", unidad="proporción", forma="proporcion_mayor",
        columna="remesas", umbral=0.0,
    ),
    IndicadorBienestar(
        clave="dep_bene_gob", etiqueta="Hogares con beneficios gubernamentales",
        dominio="Vulnerabilidad", unidad="proporción", forma="proporcion_mayor",
        columna="bene_gob", umbral=0.0,
    ),
    IndicadorBienestar(
        clave="dep_jubilacion", etiqueta="Hogares con ingresos por jubilación",
        dominio="Vulnerabilidad", unidad="proporción", forma="proporcion_mayor",
        columna="jubilacion", umbral=0.0,
    ),
    IndicadorBienestar(
        clave="dep_ocupados", etiqueta="Hogares sin ningún ocupado",
        dominio="Vulnerabilidad", unidad="proporción", forma="proporcion",
        columna="ocupados", umbral=0.0, mayor_es_mejor=False,
    ),
]

POR_CLAVE: dict[str, IndicadorBienestar] = {i.clave: i for i in CATALOGO}

DOMINIOS: list[str] = list(dict.fromkeys(i.dominio for i in CATALOGO))


# --------------------------------------------------------------------------
# Cálculo
# --------------------------------------------------------------------------


def indicadores_disponibles(datos: pd.DataFrame) -> tuple[list[IndicadorBienestar], dict[str, str]]:
    """Separa el catálogo en indicadores calculables y omitidos con su motivo."""
    disponibles: list[IndicadorBienestar] = []
    omitidos: dict[str, str] = {}
    for indicador in CATALOGO:
        faltantes = [c for c in indicador.columnas_requeridas() if c not in datos.columns]
        if faltantes:
            omitidos[indicador.clave] = f"faltan columnas: {', '.join(faltantes)}"
            continue
        disponibles.append(indicador)
    return disponibles, omitidos


def perfil_por_decil(
    datos: pd.DataFrame,
    *,
    grupo: str = "decil",
    claves: list[str] | None = None,
    cfg: config.Configuracion | None = None,
    replicas: int | None = None,
) -> tuple[pd.DataFrame, dict[str, str], list[str]]:
    """Calcula un indicador por grupo con su intervalo de confianza.

    Returns:
        ``(tabla, omitidos, avisos)``. La tabla trae una fila por grupo e
        indicador, con columnas ``grupo, indicador, etiqueta, dominio, unidad,
        valor, error_estandar, ic_inferior, ic_superior, cv, fiable, n_hogares,
        n_upm, replicas_validas, motivo``.
    """
    cfg = cfg or config.cargar_configuracion()
    disponibles, omitidos = indicadores_disponibles(datos)
    if claves:
        disponibles = [i for i in disponibles if i.clave in set(claves)]

    filas: list[dict[str, object]] = []
    avisos: list[str] = []

    for indicador in disponibles:
        estadisticos = {indicador.clave: _estimador_indicador(indicador)}
        resultado = estimar_por_grupo(
            datos, grupo, estadisticos, cfg=cfg, replicas=replicas
        )
        avisos.extend(
            f"{indicador.etiqueta}: {a}" for a in resultado.avisos
        )
        for _, fila in resultado.tabla.iterrows():
            base = f"{indicador.clave}_"
            filas.append(
                {
                    "grupo": fila["grupo"],
                    "indicador": indicador.clave,
                    "etiqueta": indicador.etiqueta,
                    "dominio": indicador.dominio,
                    "unidad": indicador.unidad,
                    "descripcion": indicador.descripcion,
                    "mayor_es_mejor": indicador.mayor_es_mejor,
                    "valor": fila.get(f"{base}valor"),
                    "error_estandar": fila.get(f"{base}error_estandar"),
                    "ic_inferior": fila.get(f"{base}ic_inferior"),
                    "ic_superior": fila.get(f"{base}ic_superior"),
                    "cv": fila.get(f"{base}cv"),
                    "fiable": fila.get(f"{base}fiable"),
                    "motivo": fila.get(f"{base}motivo", ""),
                    "n_hogares": fila.get("n_hogares"),
                    "n_upm": fila.get(f"{base}n_upm"),
                    "replicas_validas": fila.get(f"{base}replicas_validas"),
                }
            )

    return pd.DataFrame(filas), omitidos, avisos


def tabla_ancha(perfil: pd.DataFrame) -> pd.DataFrame:
    """Convierte el perfil largo en tabla ``grupo × indicador`` de valores."""
    if perfil.empty:
        return perfil
    return perfil.pivot_table(
        index="grupo", columns="indicador", values="valor", aggfunc="first"
    ).reset_index()


def comparar_grupos(
    perfil: pd.DataFrame, clave: str, grupo_a: int, grupo_b: int
) -> dict[str, float]:
    """Diferencia entre dos grupos para un indicador, con su intervalo.

    Los intervalos se propagan de forma conservadora (suma de semianchos), que
    sobreestima la incertidumbre; se usa solo como lectura orientativa.
    """
    sub = perfil[perfil["indicador"] == clave]
    a = sub[sub["grupo"] == grupo_a]
    b = sub[sub["grupo"] == grupo_b]
    if a.empty or b.empty:
        return {"diferencia": float("nan"), "ic_inferior": float("nan"),
                "ic_superior": float("nan")}
    va, vb = float(a["valor"].iloc[0]), float(b["valor"].iloc[0])
    sa = (float(a["ic_superior"].iloc[0]) - float(a["ic_inferior"].iloc[0])) / 2
    sb = (float(b["ic_superior"].iloc[0]) - float(b["ic_inferior"].iloc[0])) / 2
    diferencia = va - vb
    margen = float(np.hypot(sa, sb))
    return {
        "diferencia": diferencia,
        "ic_inferior": diferencia - margen,
        "ic_superior": diferencia + margen,
    }
