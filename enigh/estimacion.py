"""Intervalos de confianza por bootstrap con el diseño muestral de la ENIGH.

La ENIGH es una encuesta con diseño complejo: estratos (``est_dis``), unidades
primarias de muestreo (``upm``) y factores de expansión (``factor``). Remuestrear
hogares sueltos subestima la varianza, porque los hogares dentro de una misma UPM
no son independientes.

Este módulo remuestrea **UPM completas dentro de cada estrato**, que es la
aproximación estándar cuando no se dispone de las variables de diseño completas
(pseudo-estratos y pseudo-UPM de INEGI no se publican).

Además reporta el coeficiente de variación de cada estimación, que es lo que
permite marcar en el tablero las celdas con muestra insuficiente. Con 2,642
hogares en Sonora, algunos deciles municipales son genuinamente frágiles y el
tablero debe decirlo en lugar de mostrar un número inventado.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config

ESTADISTICO = Callable[[pd.DataFrame], float]


@dataclass(frozen=True)
class EstimacionPuntual:
    """Una estimación con su incertidumbre y su veredicto de fiabilidad."""

    valor: float
    error_estandar: float
    ic_inferior: float
    ic_superior: float
    cv: float
    n_hogares: int
    n_upm: int
    replicas_validas: int
    fiable: bool
    motivo: str = ""

    def como_dict(self) -> dict[str, object]:
        return {
            "valor": self.valor,
            "error_estandar": self.error_estandar,
            "ic_inferior": self.ic_inferior,
            "ic_superior": self.ic_superior,
            "cv": self.cv,
            "n_hogares": self.n_hogares,
            "n_upm": self.n_upm,
            "replicas_validas": self.replicas_validas,
            "fiable": self.fiable,
            "motivo": self.motivo,
        }


def _estimar_una(
    datos: pd.DataFrame,
    estadistico: ESTADISTICO,
    *,
    replicas: int,
    semilla: int,
    cv_no_fiable: float,
    upm_min: int,
    columna_estrato: str = "est_dis",
    columna_upm: str = "upm",
    columna_peso: str = "factor",
) -> EstimacionPuntual:
    """Calcula la estimación puntual y su intervalo por bootstrap de UPM."""
    n_hogares = int(len(datos))
    if n_hogares == 0:
        return EstimacionPuntual(
            float("nan"), float("nan"), float("nan"), float("nan"), float("nan"),
            0, 0, 0, False, "sin observaciones",
        )

    valor = float(estadistico(datos))

    tiene_diseno = (
        columna_upm in datos.columns
        and columna_estrato in datos.columns
        and datos[columna_upm].notna().any()
    )
    if not tiene_diseno:
        return EstimacionPuntual(
            valor, float("nan"), float("nan"), float("nan"), float("nan"),
            n_hogares, 0, 0, False,
            "sin variables de diseño; no se puede estimar la varianza",
        )

    upm_por_estrato = datos.groupby(columna_estrato, dropna=False)[columna_upm].nunique()
    n_upm = int(upm_por_estrato.sum())
    if n_upm < upm_min:
        return EstimacionPuntual(
            valor, float("nan"), float("nan"), float("nan"), float("nan"),
            n_hogares, n_upm, 0, False,
            f"menos de {upm_min} UPM en la celda ({n_upm})",
        )

    # Precomputa la estructura del diseño una sola vez. Todo el remuestreo se
    # hace con NumPy sobre índices de fila; hacerlo con ``groupby``/``loc`` de
    # pandas dentro del bucle de réplicas era ~20 veces más lento (33 indicadores
    # × 10 deciles × 500 réplicas tardaba 40 minutos por año).
    raiz = datos.reset_index(drop=True)
    n_filas = len(raiz)
    codigos_estrato, _ = pd.factorize(raiz[columna_estrato], use_na_sentinel=False)

    # Cada celda es un par distinto (estrato, UPM): la misma UPM puede aparecer
    # en estratos distintos y no deben mezclarse.
    marco = pd.DataFrame({
        "estrato": np.asarray(codigos_estrato, dtype=np.int64),
        "upm": raiz[columna_upm].to_numpy(),
    })
    etiquetas_celda = pd.factorize(pd.MultiIndex.from_frame(marco))[0]
    n_celdas = int(etiquetas_celda.max()) + 1 if n_filas else 0

    # Posiciones de fila de cada celda, contiguas por construcción.
    orden = np.argsort(etiquetas_celda, kind="stable")
    inicio_celda = np.zeros(n_celdas, dtype=np.int64)
    cuenta_celda = np.bincount(etiquetas_celda, minlength=n_celdas)
    if n_celdas:
        inicio_celda[1:] = np.cumsum(cuenta_celda)[:-1]
    fin_celda = inicio_celda + cuenta_celda

    # Estrato de cada celda y agrupación de celdas por estrato (para remuestrear
    # UPM dentro de estrato, nunca entre estratos).
    estrato_por_celda = np.asarray(codigos_estrato, dtype=np.int64)[orden[inicio_celda]]
    celdas_por_estrato: dict[int, np.ndarray] = {
        int(estrato): np.flatnonzero(estrato_por_celda == estrato)
        for estrato in np.unique(estrato_por_celda)
    } if n_celdas else {}

    generador = np.random.default_rng(semilla)
    valores = np.empty(replicas, dtype=float)
    validas = 0
    indices_orden = orden

    for _ in range(replicas):
        bloques = []
        for celdas in celdas_por_estrato.values():
            elegidas = generador.integers(0, celdas.size, size=celdas.size)
            for indice_celda in celdas[elegidas]:
                bloques.append(
                    indices_orden[inicio_celda[indice_celda]:fin_celda[indice_celda]]
                )
        if not bloques:
            continue
        muestra = raiz.iloc[np.concatenate(bloques)]
        try:
            estimado = float(estadistico(muestra))
        except (ValueError, ZeroDivisionError, IndexError):
            continue
        if np.isfinite(estimado):
            valores[validas] = estimado
            validas += 1

    if validas < max(20, replicas // 10):
        return EstimacionPuntual(
            valor, float("nan"), float("nan"), float("nan"), float("nan"),
            n_hogares, n_upm, validas, False,
            f"solo {validas} réplicas válidas de {replicas}",
        )

    usadas = valores[:validas]
    error_estandar = float(np.std(usadas, ddof=1))
    inferior, superior = np.percentile(usadas, [2.5, 97.5])
    # El CV se mide contra la estimación puntual original.
    cv = float(error_estandar / valor) if valor not in (0.0,) and np.isfinite(valor) else float("nan")
    if cv < 0:
        cv = float("nan")

    fiable = bool(np.isfinite(cv) and cv <= cv_no_fiable)
    motivo = "" if fiable else (
        f"CV={cv:.2f} > {cv_no_fiable}" if np.isfinite(cv) else "CV no estimable"
    )

    return EstimacionPuntual(
        valor=valor,
        error_estandar=error_estandar,
        ic_inferior=float(inferior),
        ic_superior=float(superior),
        cv=cv,
        n_hogares=n_hogares,
        n_upm=n_upm,
        replicas_validas=validas,
        fiable=fiable,
        motivo=motivo,
    )


# --------------------------------------------------------------------------
# API de alto nivel
# --------------------------------------------------------------------------


@dataclass
class ResultadoEstimaciones:
    """Estimaciones por grupo (por ejemplo, por decil)."""

    tabla: pd.DataFrame
    replicas: int
    semilla: int
    avisos: list[str] = field(default_factory=list)


def estimar_por_grupo(
    datos: pd.DataFrame,
    grupo: str,
    estadisticos: dict[str, ESTADISTICO],
    *,
    cfg: config.Configuracion | None = None,
    replicas: int | None = None,
    semilla: int | None = None,
) -> ResultadoEstimaciones:
    """Estima uno o varios estadísticos para cada valor de ``grupo``.

    Args:
        datos: microdatos de hogares.
        grupo: columna que define los grupos (normalmente ``decil``).
        estadisticos: ``{nombre: función(df) -> float}``.
        cfg: configuración; se usa la global si se omite.

    Returns:
        :class:`ResultadoEstimaciones` con una fila por grupo y, por cada
        estadístico, las columnas ``{nombre}``, ``{nombre}_ee``, ``{nombre}_ic_inf``,
        ``{nombre}_ic_sup``, ``{nombre}_cv`` y ``{nombre}_fiable``.
    """
    cfg = cfg or config.cargar_configuracion()
    replicas = replicas or cfg.umbrales.replicas_bootstrap
    semilla = semilla if semilla is not None else cfg.umbrales.semilla_bootstrap

    filas = []
    avisos: list[str] = []

    for valor_grupo, sub in datos.groupby(grupo, dropna=False, sort=True):
        fila: dict[str, object] = {"grupo": valor_grupo, "n_hogares": int(len(sub))}
        for nombre, estadistico in estadisticos.items():
            resultado = _estimar_una(
                sub,
                estadistico,
                replicas=replicas,
                semilla=semilla,
                cv_no_fiable=cfg.umbrales.cv_no_fiable,
                upm_min=cfg.umbrales.upm_min_bootstrap,
            )
            fila.update({f"{nombre}_{k}": v for k, v in resultado.como_dict().items()
                         if k != "n_hogares"})
            fila["n_hogares"] = max(int(fila.get("n_hogares", 0)), resultado.n_hogares)
            if not resultado.fiable and resultado.motivo:
                avisos.append(f"{grupo}={valor_grupo} · {nombre}: {resultado.motivo}")

        filas.append(fila)

    return ResultadoEstimaciones(
        tabla=pd.DataFrame(filas), replicas=replicas, semilla=semilla, avisos=avisos
    )


# --------------------------------------------------------------------------
# Estadísticos reutilizables
# --------------------------------------------------------------------------


def estadistico_media(columna: str, columna_peso: str = "factor") -> ESTADISTICO:
    """Media ponderada de una columna."""

    def _fn(df: pd.DataFrame) -> float:
        from .metricas import media_ponderada

        return media_ponderada(df[columna], df[columna_peso])

    _fn.__name__ = f"media_{columna}"
    return _fn


def estadistico_mediana(columna: str, columna_peso: str = "factor") -> ESTADISTICO:
    """Mediana ponderada de una columna."""
    from .metricas import mediana_ponderada

    def _fn(df: pd.DataFrame) -> float:
        return mediana_ponderada(df[columna], df[columna_peso])

    _fn.__name__ = f"mediana_{columna}"
    return _fn


def estadistico_proporcion(columna: str, valor: float = 1.0, columna_peso: str = "factor") -> ESTADISTICO:
    """Proporción ponderada de hogares donde ``columna == valor``."""
    from .metricas import media_ponderada

    def _fn(df: pd.DataFrame) -> float:
        indicador = (pd.to_numeric(df[columna], errors="coerce") == valor).astype(float)
        indicador[df[columna].isna()] = np.nan
        return media_ponderada(indicador, df[columna_peso])

    _fn.__name__ = f"prop_{columna}"
    return _fn


def estadistico_gini(columna: str, columna_peso: str = "factor") -> ESTADISTICO:
    """Coeficiente de Gini ponderado de una columna."""
    from .metricas import gini

    def _fn(df: pd.DataFrame) -> float:
        return gini(df[columna], df[columna_peso])

    _fn.__name__ = f"gini_{columna}"
    return _fn


def estadistico_razon(columna_num: str, columna_den: str, columna_peso: str = "factor") -> ESTADISTICO:
    """Razón ponderada de dos columnas (p. ej. participación de un decil)."""
    from .metricas import media_ponderada

    def _fn(df: pd.DataFrame) -> float:
        return media_ponderada(df[columna_num], df[columna_peso]) / media_ponderada(
            df[columna_den], df[columna_peso]
        )

    _fn.__name__ = f"razon_{columna_num}_{columna_den}"
    return _fn
