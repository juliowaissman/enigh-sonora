"""Medidas de ingreso ponderadas por el factor de expansión de la ENIGH.

Regla del proyecto: **nunca** se calculan promedios simples sobre los microdatos.
Todas las funciones de este módulo reciben una serie de ponderadores y las
respetan.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

NUM_DECILES = 10


class ErrorDeMetrica(ValueError):
    """Los insumos no permiten calcular la medida solicitada."""


# --------------------------------------------------------------------------
# Primitivas ponderadas
# --------------------------------------------------------------------------


def _limpiar(
    valores: pd.Series | np.ndarray,
    pesos: pd.Series | np.ndarray,
    *,
    permitir_cero: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Valida y alinea valores y pesos, descartando nulos y pesos no positivos."""
    v = np.asarray(valores, dtype=float)
    w = np.asarray(pesos, dtype=float)
    if v.shape != w.shape:
        raise ErrorDeMetrica(
            f"valores y pesos deben tener la misma forma: {v.shape} vs {w.shape}"
        )
    valido = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not permitir_cero:
        valido &= v > 0
    return v[valido], w[valido]


def media_ponderada(valores, pesos) -> float:
    """Media ponderada, ``NaN`` si no hay observaciones válidas."""
    v, w = _limpiar(valores, pesos)
    if v.size == 0:
        return float("nan")
    return float(np.sum(v * w) / np.sum(w))


def _bloque_del_objetivo(
    v_ordenado: np.ndarray, w_acum: np.ndarray, objetivo: float
) -> tuple[int, bool]:
    """Localiza el bloque de peso en el que cae el objetivo ponderado.

    Returns:
        ``(indice, exacto)``.

        * ``exacto = False``: el objetivo cae **dentro** del bloque ``indice``,
          entre ``w_acum[indice-1]`` y ``w_acum[indice]``. Toca repartir el peso
          de forma proporcional.
        * ``exacto = True``: el objetivo cae **justo en la frontera** al final del
          bloque ``indice``. Ese bloque entra completo y no se toca el siguiente.

    La distinción es la que hace que con pesos uniformes el resultado coincida con
    ``numpy.percentile``: para ``valores=[1, 3]`` y ``pesos=[1, 1]``, el objetivo
    0.5 cae en la frontera, y la mediana es el promedio de ambos (1.0), no el
    primer valor (que era el resultado anterior).

    La comparación usa tolerancia relativa porque ``0.9 * 1000`` puede quedar en
    ``899.9999999999999`` según el redondeo, y una comparación exacta dejaba las
    participaciones desfasadas 0.1 puntos porcentuales.
    """
    n = v_ordenado.size
    tolerancia = max(abs(w_acum[-1]), 1.0) * 1e-9

    # ``fin`` es el primer índice cuyo peso acumulado alcanza el objetivo.
    fin = int(np.searchsorted(w_acum, objetivo, side="left"))
    if fin >= n:
        return n - 1, False

    if fin > 0 and abs(objetivo - w_acum[fin - 1]) <= tolerancia:
        return fin - 1, True
    if abs(objetivo - w_acum[fin]) <= tolerancia:
        return fin, True

    return fin, False


def percentil_ponderado(valores, pesos, q: float) -> float:
    """Percentil ponderado, inverso de la distribución acumulada ponderada.

    Con pesos uniformes coincide prácticamente con ``numpy.percentile`` (mismo
    criterio de interpolación lineal), que es la referencia que se usa en las
    pruebas.

    Args:
        q: cuantil en ``(0, 1]``.
    """
    if not 0 < q <= 1:
        raise ErrorDeMetrica(f"El cuantil debe estar en (0, 1]; se recibió {q}.")
    v, w = _limpiar(valores, pesos)
    if v.size == 0:
        return float("nan")
    orden = np.argsort(v, kind="stable")
    v, w = v[orden], w[orden]
    w_acum = np.cumsum(w)
    objetivo = q * w_acum[-1]

    indice, exacto = _bloque_del_objetivo(v, w_acum, objetivo)

    if exacto:
        # Frontera entre el bloque anterior y éste: se promedian los dos valores.
        anterior = max(indice - 1, 0)
        if anterior == indice:
            return float(v[indice])
        return float((v[anterior] + v[indice]) / 2)

    if indice <= 0:
        return float(v[0])
    inicio = w_acum[indice - 1]
    ancho = w_acum[indice] - inicio
    if ancho <= 0:
        return float(v[indice])
    fraccion = (objetivo - inicio) / ancho
    return float(v[indice - 1] + fraccion * (v[indice] - v[indice - 1]))


def mediana_ponderada(valores, pesos) -> float:
    """Mediana ponderada."""
    return percentil_ponderado(valores, pesos, 0.5)


def cortes_deciles(valores, pesos) -> np.ndarray:
    """Nueve cortes que dividen la población ponderada en diez partes iguales.

    Returns:
        Arreglo de 9 umbrales crecientes (p10..p90). ``numpy.nan`` si no hay datos.
    """
    v, w = _limpiar(valores, pesos)
    if v.size == 0:
        return np.full(NUM_DECILES - 1, np.nan)
    return np.array([percentil_ponderado(v, w, d / NUM_DECILES) for d in range(1, NUM_DECILES)])


def asignar_deciles(valores, pesos, cortes: np.ndarray | None = None) -> np.ndarray:
    """Etiqueta cada observación con su decil (1 = más pobre, 10 = más rico).

    Si se pasan ``cortes``, se usan los de otra población (por ejemplo, los
    cortes nacionales para clasificar a los hogares de Sonora). Si no, se
    calculan sobre los propios datos.
    """
    v = np.asarray(valores, dtype=float)
    if cortes is None:
        cortes = cortes_deciles(valores, pesos)
    if np.all(np.isnan(cortes)):
        return np.full(v.shape, np.nan)

    # searchsorted con side='right': un valor igual al corte pertenece al decil
    # inferior, que es la convención conservadora.
    etiquetas = np.searchsorted(cortes, v, side="right") + 1
    etiquetas = np.where(np.isfinite(v), etiquetas, np.nan)
    return etiquetas.astype(float)


# --------------------------------------------------------------------------
# Desigualdad
# --------------------------------------------------------------------------


def gini(valores, pesos) -> float:
    """Coeficiente de Gini ponderado (0 = igualdad total, 1 = desigualdad total)."""
    v, w = _limpiar(valores, pesos)
    if v.size == 0:
        return float("nan")
    orden = np.argsort(v, kind="stable")
    v, w = v[orden], w[orden]
    w_acum = np.cumsum(w)
    w_total = w_acum[-1]
    if w_total <= 0:
        return float("nan")
    # Fórmula de la covarianza ponderada de Sen.
    media = np.sum(v * w) / w_total
    if media <= 0:
        return float("nan")
    # Posición acumulada centrada de cada observación dentro de su peso.
    p_anterior = (w_acum - w) / w_total
    p_centro = p_anterior + (w / w_total) / 2
    return float(1 - 2 * np.sum((1 - p_centro) * v * w) / (w_total * media))


def theil_t(valores, pesos) -> float:
    """Índice de Theil-T ponderado. Requiere valores estrictamente positivos."""
    v, w = _limpiar(valores, pesos, permitir_cero=False)
    if v.size == 0:
        return float("nan")
    w_total = np.sum(w)
    media = np.sum(v * w) / w_total
    if media <= 0:
        return float("nan")
    razones = v / media
    return float(np.sum(w * razones * np.log(razones)) / w_total)


def razon_palma(valores, pesos) -> float:
    """Razón de Palma: participación del 10 % más rico / la del 40 % más pobre.

    Se leen los campos **por nombre** (no por posición) porque la versión anterior
    desempaquetaba la tupla en el orden equivocado y devolvía 0.25 en lugar de 1
    cuando todos los ingresos eran iguales: un error que ningún promedio habría
    delatado.
    """
    participaciones = participacion_por_grupo(valores, pesos)
    if participaciones is None:
        return float("nan")
    if participaciones.bajo40 == 0:
        return float("nan")
    return float(participaciones.alto10 / participaciones.bajo40)


@dataclass(frozen=True)
class Participaciones:
    """Participaciones acumuladas en el ingreso total, en porcentaje.

    Es un objeto con nombre en lugar de una tupla posicional: con una tupla, dos
    campos intercambiables del mismo tipo producen un resultado numéricamente
    plausible pero equivocado, sin error y sin aviso.
    """

    bajo40: float
    bajo50: float
    bajo60: float
    bajo90: float
    alto10: float

    def como_tupla(self) -> tuple[float, float, float, float, float]:
        return (self.bajo40, self.bajo50, self.bajo60, self.bajo90, self.alto10)


def participacion_por_grupo(valores, pesos) -> Participaciones | None:
    """Participaciones en el ingreso total de los grupos acumulados."""
    v, w = _limpiar(valores, pesos)
    if v.size == 0:
        return None
    total = np.sum(v * w)
    if total <= 0:
        return None
    acumuladas = {
        q: _participacion_acumulada(v, w, q) / total * 100
        for q in (0.4, 0.5, 0.6, 0.9)
    }
    return Participaciones(
        bajo40=float(acumuladas[0.4]),
        bajo50=float(acumuladas[0.5]),
        bajo60=float(acumuladas[0.6]),
        bajo90=float(acumuladas[0.9]),
        # El 10 % más rico es el complemento del 90 % más pobre.
        alto10=float(100.0 - acumuladas[0.9]),
    )


def _participacion_acumulada(v: np.ndarray, w: np.ndarray, q: float) -> float:
    """Suma de ``v*w`` para la fracción más pobre ``q`` de la población ponderada.

    Es la integral de la curva de Lorenz hasta ``q``. La frontera se reparte de
    forma proporcional al peso, y cuando el objetivo cae justo en una frontera el
    bloque entra completo. Sin ese cuidado, con ingreso constante la participación
    del 40 % más pobre daba 39.9 % en lugar de 40 % y la razón de Palma salía
    0.25 en lugar de 1.
    """
    if q >= 1.0:
        return float(np.sum(v * w))

    orden = np.argsort(v, kind="stable")
    v_ord, w_ord = v[orden], w[orden]
    w_acum = np.cumsum(w_ord)
    objetivo = q * w_acum[-1]

    indice, exacto = _bloque_del_objetivo(v_ord, w_acum, objetivo)

    if exacto:
        # El bloque ``indice`` entra completo.
        return float(np.sum(v_ord[: indice + 1] * w_ord[: indice + 1]))

    suma = float(np.sum(v_ord[:indice] * w_ord[:indice]))
    inicio = w_acum[indice - 1] if indice > 0 else 0.0
    restante = objetivo - inicio
    if restante > 0 and indice < v_ord.size:
        suma += float(restante * v_ord[indice])
    return suma


# --------------------------------------------------------------------------
# Resumen por decil
# --------------------------------------------------------------------------


@dataclass
class ResumenDeciles:
    """Tabla de deciles y los escalares de desigualdad del mismo conjunto."""

    tabla: pd.DataFrame
    gini: float
    theil: float
    palma: float
    participacion_bajo40: float
    participacion_alto10: float
    cortes: np.ndarray
    n_observaciones: int
    poblacion_expandida: float


def resumen_deciles(
    df: pd.DataFrame,
    *,
    columna_ingreso: str,
    columna_peso: str = "factor",
    cortes: np.ndarray | None = None,
    columna_poblacion: str | None = None,
    nombre_metrica: str | None = None,
) -> ResumenDeciles:
    """Resumen de ingreso por decil más los escalares de desigualdad.

    Args:
        df: microdatos de hogares.
        columna_ingreso: métrica que ordena los deciles (p. ej. ``ing_pc``).
        columna_peso: factor de expansión.
        cortes: cortes decílicos a usar; si se omiten, se calculan de ``df``.
        columna_poblacion: si se indica, se reporta población además de hogares.
        nombre_metrica: nombre de la columna de ingreso en la tabla de salida.
            Por omisión se usa ``columna_ingreso``. Se expone para que el
            consumidor pueda pedir explícitamente ``ing_pc`` o ``ing_ae``: con
            nombres genéricos era fácil que el tablero leyera una columna
            inexistente y callera silenciosamente en el per cápita.

    Returns:
        :class:`ResumenDeciles` con una fila por decil (1..10).
    """
    if columna_ingreso not in df.columns:
        raise ErrorDeMetrica(f"La columna de ingreso {columna_ingreso!r} no existe.")
    if columna_peso not in df.columns:
        raise ErrorDeMetrica(f"La columna de peso {columna_peso!r} no existe.")

    trabajo = df[[columna_ingreso, columna_peso] + ([columna_poblacion] if columna_poblacion else [])].copy()
    trabajo = trabajo[np.isfinite(trabajo[columna_ingreso]) & (trabajo[columna_peso] > 0)]
    if trabajo.empty:
        raise ErrorDeMetrica("No quedaron hogares válidos para calcular los deciles.")

    pesos = trabajo[columna_peso].to_numpy(dtype=float)
    valores = trabajo[columna_ingreso].to_numpy(dtype=float)
    cortes_usados = cortes_deciles(valores, pesos) if cortes is None else np.asarray(cortes, dtype=float)

    trabajo["decil"] = asignar_deciles(valores, pesos, cortes_usados)
    trabajo = trabajo.dropna(subset=["decil"])
    trabajo["decil"] = trabajo["decil"].astype(int)

    filas = []
    for decil in range(1, NUM_DECILES + 1):
        grupo = trabajo[trabajo["decil"] == decil]
        if grupo.empty:
            filas.append(
                {
                    "decil": decil,
                    "n_hogares": 0,
                    "hogares_expandidos": 0.0,
                    "ingreso_promedio": float("nan"),
                    "ingreso_mediana": float("nan"),
                    "limite_inferior": float("nan"),
                    "limite_superior": float("nan"),
                    "participacion_ingreso": float("nan"),
                }
            )
            continue
        v = grupo[columna_ingreso].to_numpy(dtype=float)
        w = grupo[columna_peso].to_numpy(dtype=float)
        fila = {
            "decil": decil,
            "n_hogares": int(len(grupo)),
            "hogares_expandidos": float(w.sum()),
            "ingreso_promedio": media_ponderada(v, w),
            "ingreso_mediana": mediana_ponderada(v, w),
            "limite_inferior": float(np.min(v)),
            "limite_superior": float(np.max(v)),
        }
        if columna_poblacion:
            fila["poblacion_expandida"] = float(
                (grupo[columna_poblacion].to_numpy(dtype=float) * w).sum()
            )
        filas.append(fila)

    tabla = pd.DataFrame(filas)

    total_ingreso = float(np.sum(valores * pesos))
    tabla["participacion_ingreso"] = np.where(
        total_ingreso > 0,
        tabla["hogares_expandidos"] * tabla["ingreso_promedio"] / total_ingreso * 100,
        np.nan,
    )
    # La participación de cada decil se recalcula con suma directa para que los
    # errores de redondeo no acumulen una desviación sobre 100 %.
    sumas = np.array(
        [
            float(np.sum(valores[trabajo["decil"].to_numpy() == d] * pesos[trabajo["decil"].to_numpy() == d]))
            for d in range(1, NUM_DECILES + 1)
        ]
    )
    tabla["suma_ingreso_ponderado"] = sumas
    if total_ingreso > 0:
        tabla["participacion_ingreso"] = sumas / total_ingreso * 100

    p40 = _participacion_acumulada(valores, pesos, 0.4) / total_ingreso * 100
    p10 = _participacion_acumulada(valores, pesos, 0.9) / total_ingreso * 100

    # La tabla se publica con nombres **explícitos** de la métrica, nunca con los
    # genéricos ``ingreso_promedio``/``ingreso_mediana``. Con nombres genéricos,
    # un consumidor que pedía ``["ing_pc"]`` obtenía un KeyError; y si el nombre
    # genérico coincidía por casualidad, el tablero mostraba el per cápita aunque
    # el usuario hubiera elegido el adulto equivalente.
    etiqueta = nombre_metrica or columna_ingreso
    tabla = tabla.rename(
        columns={
            "ingreso_promedio": etiqueta,
            "ingreso_mediana": f"{etiqueta}_mediana",
        }
    )

    return ResumenDeciles(
        tabla=tabla,
        gini=gini(valores, pesos),
        theil=theil_t(valores, pesos),
        palma=float(p10 / p40) if p40 > 0 else float("nan"),
        participacion_bajo40=p40,
        participacion_alto10=p10,
        cortes=cortes_usados,
        n_observaciones=int(len(trabajo)),
        poblacion_expandida=float(pesos.sum()),
    )


def escala_equivalencia_por_defecto() -> dict[str, float]:
    """Escala de adulto equivalente usada por el proyecto (documentada)."""
    return {"menores": 0.7, "p12_64": 0.8, "p65mas": 1.0}
