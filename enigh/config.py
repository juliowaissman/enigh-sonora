"""Configuración central del proyecto: rutas, fuentes y umbrales metodológicos.

Toda la configuración variable vive en ``config/`` (YAML/CSV) para que el código
no tenga URLs ni umbrales incrustados. Este módulo es la única puerta de entrada
a esa configuración.
"""

from __future__ import annotations

import csv
import functools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# --------------------------------------------------------------------------
# Rutas
# --------------------------------------------------------------------------

RAIZ = Path(__file__).resolve().parent.parent
DIR_CONFIG = RAIZ / "config"
DIR_DATOS = RAIZ / "data"
DIR_RAW = DIR_DATOS / "raw"
DIR_INTERIM = DIR_DATOS / "interim"
DIR_PROCESSED = DIR_DATOS / "processed"
DIR_GEO = DIR_PROCESSED / "geografia"
DIR_MUESTRA = DIR_DATOS / "muestra"

ARCHIVO_FUENTES = DIR_CONFIG / "fuentes.yaml"
ARCHIVO_DEFLACTORES = DIR_CONFIG / "deflactores.csv"
ARCHIVO_INDICADORES = DIR_CONFIG / "indicadores.yaml"
ARCHIVO_MANIFIESTO = DIR_PROCESSED / "manifiesto.json"

# Clave de entidad del estado de interés.
CLAVE_SONORA = "26"
NOMBRE_SONORA = "Sonora"


class ErrorDeConfiguracion(RuntimeError):
    """La configuración del proyecto es inválida o está incompleta."""


# --------------------------------------------------------------------------
# Modelo de la configuración de fuentes
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FuenteAnual:
    """Un levantamiento de la ENIGH: su ZIP de microdatos y sus metadatos."""

    anio: int
    url: str
    bytes_esperados: int | None = None
    habilitado: bool = True
    etiqueta: str = ""
    nota: str = ""

    @property
    def nombre_archivo(self) -> str:
        return self.url.rsplit("/", 1)[-1]


@dataclass(frozen=True)
class Tabla:
    """Una tabla de microdatos dentro del ZIP."""

    nombre: str
    descripcion: str = ""
    obligatoria: bool = False


@dataclass(frozen=True)
class Umbrales:
    """Umbrales metodológicos que controlan la supresión y la fiabilidad."""

    cv_no_fiable: float = 0.25
    hogares_min_municipio: int = 30
    upm_min_bootstrap: int = 20
    replicas_bootstrap: int = 500
    semilla_bootstrap: int = 20240101


@dataclass(frozen=True)
class Configuracion:
    """Configuración completa y validada del proyecto."""

    base_url: str
    anio_base_deflactor: int
    anios: dict[int, FuenteAnual]
    tablas: dict[str, Tabla]
    tablas_excluidas: list[str]
    columnas_requeridas_concentradohogar: list[str]
    componentes_ing_cor: list[str]
    umbrales: Umbrales
    deflactores: dict[int, float] = field(default_factory=dict)

    @property
    def anios_habilitados(self) -> list[int]:
        return sorted(a for a, f in self.anios.items() if f.habilitado)

    def fuente(self, anio: int) -> FuenteAnual:
        try:
            return self.anios[int(anio)]
        except KeyError as exc:  # pragma: no cover - mensaje de error útil
            disponibles = ", ".join(str(a) for a in sorted(self.anios))
            raise ErrorDeConfiguracion(
                f"El año {anio} no está en config/fuentes.yaml. "
                f"Años declarados: {disponibles}."
            ) from exc

    def deflactor(self, anio: int) -> float | None:
        """Factor para convertir pesos del ``anio`` a pesos del año base.

        Devuelve ``None`` si el año no está en la tabla de deflactores: el
        llamador debe deshabilitar la opción "pesos reales" en lugar de
        interpolar en silencio.
        """
        indice_anio = self.deflactores.get(int(anio))
        indice_base = self.deflactores.get(self.anio_base_deflactor)
        if not indice_anio or not indice_base:
            return None
        return indice_base / indice_anio


# --------------------------------------------------------------------------
# Carga
# --------------------------------------------------------------------------


def _leer_yaml(ruta: Path) -> dict[str, Any]:
    if not ruta.exists():
        raise ErrorDeConfiguracion(f"No existe el archivo de configuración: {ruta}")
    with ruta.open(encoding="utf-8") as fh:
        datos = yaml.safe_load(fh) or {}
    if not isinstance(datos, dict):
        raise ErrorDeConfiguracion(f"{ruta} no contiene un mapeo YAML válido.")
    return datos


def cargar_deflactores(ruta: Path | None = None) -> dict[int, float]:
    """Lee la tabla de INPC anual (``config/deflactores.csv``).

    Columnas esperadas: ``anio,indice_inpc,fuente``. Se ignoran las filas sin
    índice numérico para que la tabla pueda documentar fuentes pendientes.
    """
    ruta = ruta or ARCHIVO_DEFLACTORES
    if not ruta.exists():
        return {}
    salida: dict[int, float] = {}
    with ruta.open(encoding="utf-8-sig", newline="") as fh:
        for fila in csv.DictReader(fh):
            anio_txt = (fila.get("anio") or "").strip()
            indice_txt = (fila.get("indice_inpc") or "").strip()
            if not anio_txt or not indice_txt:
                continue
            try:
                salida[int(anio_txt)] = float(indice_txt)
            except ValueError:
                continue
    return salida


@functools.lru_cache(maxsize=1)
def cargar_configuracion(ruta: Path | None = None) -> Configuracion:
    """Carga y valida la configuración del proyecto (cacheada)."""
    datos = _leer_yaml(ruta or ARCHIVO_FUENTES)

    anios: dict[int, FuenteAnual] = {}
    for anio, bruto in (datos.get("anios") or {}).items():
        if not isinstance(bruto, dict) or not bruto.get("url"):
            raise ErrorDeConfiguracion(f"El año {anio} no declara 'url' en fuentes.yaml.")
        anios[int(anio)] = FuenteAnual(
            anio=int(anio),
            url=str(bruto["url"]),
            bytes_esperados=bruto.get("bytes_esperados"),
            habilitado=bool(bruto.get("habilitado", True)),
            etiqueta=str(bruto.get("etiqueta", anio)),
            nota=str(bruto.get("nota", "")),
        )

    tablas = {
        nombre: Tabla(
            nombre=nombre,
            descripcion=str(bruto.get("descripcion", "")),
            obligatoria=bool(bruto.get("obligatoria", False)),
        )
        for nombre, bruto in (datos.get("tablas") or {}).items()
    }

    umbrales_brutos = datos.get("umbrales") or {}
    umbrales = Umbrales(**{k: v for k, v in umbrales_brutos.items() if k in Umbrales.__dataclass_fields__})

    config = Configuracion(
        base_url=str(datos.get("base_url", "")),
        anio_base_deflactor=int(datos.get("anio_base_deflactor", 2024)),
        anios=anios,
        tablas=tablas,
        tablas_excluidas=list(datos.get("tablas_excluidas") or []),
        columnas_requeridas_concentradohogar=list(
            datos.get("columnas_requeridas_concentradohogar") or []
        ),
        componentes_ing_cor=list(datos.get("componentes_ing_cor") or []),
        umbrales=umbrales,
        deflactores=cargar_deflactores(),
    )

    habilitados = config.anios_habilitados
    if not habilitados:
        raise ErrorDeConfiguracion("No hay ningún año habilitado en config/fuentes.yaml.")
    if config.anio_base_deflactor not in config.deflactores:
        # No es fatal: solo deshabilita la vista de pesos reales.
        pass
    return config


# --------------------------------------------------------------------------
# Utilidades de rutas
# --------------------------------------------------------------------------


def ruta_hogares(anio: int) -> Path:
    """Parquet curado de hogares para un año."""
    return DIR_PROCESSED / "hogares" / f"ano={anio}" / "hogares.parquet"


def ruta_agregados(anio: int, ambito: str) -> Path:
    """Parquet de agregados precomputados para un año y ámbito."""
    return DIR_PROCESSED / "agregados" / f"ano={anio}" / f"{ambito}.parquet"


def ruta_zip(anio: int) -> Path:
    """ZIP original descargado (git-ignored)."""
    return DIR_RAW / f"enigh_{anio}.zip"


def ruta_extraccion(anio: int) -> Path:
    """Directorio de CSV extraídos de un año (git-ignored)."""
    return DIR_INTERIM / str(anio)


def asegurar_directorios() -> None:
    """Crea el árbol de directorios de datos si no existe."""
    for directorio in (DIR_RAW, DIR_INTERIM, DIR_PROCESSED, DIR_GEO, DIR_MUESTRA):
        directorio.mkdir(parents=True, exist_ok=True)
