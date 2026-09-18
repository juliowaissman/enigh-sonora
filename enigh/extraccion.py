"""Descompresión selectiva de las tablas de la ENIGH.

El ZIP de INEGI contiene ~20 tablas, algunas enormes (``gastoshogar`` descomprime
579 MB). Este módulo extrae únicamente las tablas declaradas como útiles en
``config/fuentes.yaml`` y deja el resto dentro del ZIP.

La estructura interna del ZIP no es estable entre años::

    conjunto_de_datos_concentradohogar_enigh2024_ns/
    conjunto_de_datos_concentradohogar_enigh_2020_ns/
    conjunto_de_datos_concentradohogar_enigh2022_ns/

Por eso la localización se hace por búsqueda de prefijo y no por nombre fijo.
"""

from __future__ import annotations

import csv
import logging
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

from . import config

log = logging.getLogger(__name__)


def _codificacion(ruta: Path) -> str:
    """Codificación del archivo de INEGI (varía entre años: 2020 utf-8, 2022 latin-1).

    Se importa de :mod:`enigh.limpieza` para tener una sola implementación de la
    detección, sin duplicar el orden de intentos.
    """
    from .limpieza import detectar_codificacion

    return detectar_codificacion(ruta)

# carpeta/conjunto_de_datos/archivo.csv
PATRON_CSV_DATOS = "conjunto_de_datos/*.csv"
PATRON_CSV_CATALOGO = "catalogos/*.csv"


@dataclass(frozen=True)
class TablaExtraida:
    """Ubicación de una tabla ya extraída a disco."""

    nombre: str
    anio: int
    directorio: Path
    csv_datos: Path
    csv_diccionario: Path | None
    directorio_catalogos: Path | None

    def existe(self) -> bool:
        return self.csv_datos.exists()


def _sufijo_tabla(anio: int) -> str:
    """Sufijo del directorio de tablas para un año.

    Verificado: 2020 -> ``enigh_2020_ns``, 2022 -> ``enigh2022_ns``,
    2024 -> ``enigh2024_ns``. Se calcula pero la localización real usa búsqueda
    por prefijo, porque INEGI cambia el patrón sin aviso.
    """
    return f"enigh{anio}_ns"


def _directorios_tabla(zf: zipfile.ZipFile, tabla: str, anio: int) -> list[str]:
    """Directorios de primer nivel del ZIP que corresponden a ``tabla``.

    Se ancla el prefijo exacto ``conjunto_de_datos_{tabla}_`` para no confundir
    ``ingresos`` con ``ingresos_jcf`` ni ``agro`` con ``agroconsumo``.
    """
    prefijos = [f"conjunto_de_datos_{tabla}_", f"conjunto_de_datos_{tabla}_{anio}"]
    encontrados = []
    for nombre in zf.namelist():
        if "/" not in nombre:
            continue
        raiz = nombre.split("/", 1)[0]
        if any(raiz.startswith(p) for p in prefijos):
            if raiz not in encontrados:
                encontrados.append(raiz)
    return encontrados


def _elegir_directorio(
    zf: zipfile.ZipFile, tabla: str, anio: int, destino: Path
) -> str | None:
    """Elige el directorio de la tabla, con vuelta atrás por coincidencia difusa."""
    candidatos = _directorios_tabla(zf, tabla, anio)
    if not candidatos:
        # Vuelta atrás: cualquier raíz que contenga la tabla como palabra.
        patron = re.compile(rf"^conjunto_de_datos_{re.escape(tabla)}(_|$)")
        candidatos = sorted(
            {n.split("/", 1)[0] for n in zf.namelist() if "/" in n}
            & set()
        ) or [
            n.split("/", 1)[0]
            for n in zf.namelist()
            if "/" in n and patron.match(n.split("/", 1)[0])
        ]
        candidatos = sorted(set(candidatos))
    if not candidatos:
        return None
    if len(candidatos) == 1:
        return candidatos[0]
    # Preferir el que ya exista en disco (re-ejecuciones) y luego el más corto.
    return sorted(candidatos, key=lambda c: (not (destino / c).exists(), len(c)))[0]


def extraer_anio(
    anio: int,
    *,
    forzar: bool = False,
    cfg: config.Configuracion | None = None,
) -> dict[str, TablaExtraida]:
    """Extrae a disco las tablas útiles de un año.

    Returns:
        Diccionario ``nombre_tabla -> TablaExtraida`` solo con las tablas que se
        encontraron en el ZIP.
    """
    cfg = cfg or config.cargar_configuracion()
    ruta_zip = config.ruta_zip(anio)
    if not ruta_zip.exists():
        raise FileNotFoundError(
            f"No existe {ruta_zip}. Ejecute primero la descarga "
            f"(python -m enigh.pipeline --descargar)."
        )

    destino = config.ruta_extraccion(anio)
    destino.mkdir(parents=True, exist_ok=True)

    extraidas: dict[str, TablaExtraida] = {}
    with zipfile.ZipFile(ruta_zip) as zf:
        for nombre_tabla, tabla in cfg.tablas.items():
            raiz = _elegir_directorio(zf, nombre_tabla, anio, destino)
            if raiz is None:
                mensaje = (
                    f"[{anio}] No se encontró la tabla {nombre_tabla!r} en "
                    f"{ruta_zip.name}."
                )
                if tabla.obligatoria:
                    raise RuntimeError(mensaje)
                log.warning("%s Se omite; los indicadores que dependen de ella "
                            "quedarán como no disponibles.", mensaje)
                continue

            dir_tabla = destino / raiz
            csv_datos = _localizar_o_extraer(zf, raiz, PATRON_CSV_DATOS, dir_tabla, forzar)
            if csv_datos is None:
                mensaje = f"[{anio}] La tabla {nombre_tabla!r} no contiene datos en {raiz}."
                if tabla.obligatoria:
                    raise RuntimeError(mensaje)
                log.warning(mensaje)
                continue

            csv_dicc = _localizar_o_extraer(
                zf, raiz, "diccionario_de_datos/*.csv", dir_tabla, forzar
            )
            dir_cat = _extraer_catalogos(zf, raiz, dir_tabla, forzar)

            extraidas[nombre_tabla] = TablaExtraida(
                nombre=nombre_tabla,
                anio=anio,
                directorio=dir_tabla,
                csv_datos=csv_datos,
                csv_diccionario=csv_dicc,
                directorio_catalogos=dir_cat,
            )
            log.info("[%s] Tabla %s extraída: %s", anio, nombre_tabla, csv_datos.name)

    if "concentradohogar" not in extraidas:
        raise RuntimeError(
            f"[{anio}] La tabla obligatoria 'concentradohogar' no se pudo extraer."
        )
    return extraidas


def _localizar_o_extraer(
    zf: zipfile.ZipFile,
    raiz: str,
    patron: str,
    dir_tabla: Path,
    forzar: bool,
) -> Path | None:
    """Devuelve la ruta local del primer CSV que casa con ``patron``, extrayéndolo."""
    import fnmatch

    prefijo = f"{raiz}/"
    coincidencias = [
        n for n in zf.namelist()
        if n.startswith(prefijo) and fnmatch.fnmatch(n[len(prefijo):], patron)
    ]
    if not coincidencias:
        return None
    miembro = sorted(coincidencias)[0]
    relativo = miembro[len(prefijo):]
    salida = dir_tabla / relativo

    if salida.exists() and not forzar and salida.stat().st_size > 0:
        return salida

    salida.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(miembro) as origen, salida.open("wb") as destino_fh:
        while True:
            bloque = origen.read(1 << 22)
            if not bloque:
                break
            destino_fh.write(bloque)
    return salida


def _extraer_catalogos(
    zf: zipfile.ZipFile, raiz: str, dir_tabla: Path, forzar: bool
) -> Path | None:
    """Extrae los catálogos (incluido ``ubica_geo.csv``) de una tabla."""
    prefijo = f"{raiz}/catalogos/"
    miembros = [n for n in zf.namelist() if n.startswith(prefijo) and n.endswith(".csv")]
    if not miembros:
        return None
    dir_cat = dir_tabla / "catalogos"
    dir_cat.mkdir(parents=True, exist_ok=True)
    for miembro in miembros:
        salida = dir_tabla / miembro[len(f"{raiz}/"):]
        if salida.exists() and not forzar and salida.stat().st_size > 0:
            continue
        salida.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(miembro) as origen, salida.open("wb") as destino_fh:
            destino_fh.write(origen.read())
    return dir_cat


# --------------------------------------------------------------------------
# Lectura del diccionario de datos y de los catálogos
# --------------------------------------------------------------------------


def leer_diccionario(ruta: Path) -> dict[str, dict[str, str]]:
    """Lee el diccionario de datos de INEGI.

    El archivo no trae encabezado de nombres de variable en la primera fila de
    forma consistente: la columna ``nemónico`` contiene el nombre real de la
    variable. Devuelve ``{nombre_variable: {resto_de_campos}}``.

    La primera columna es la descripción en español y su encabezado aparece
    literalmente como ``nombre_campo``.
    """
    if not ruta.exists():
        return {}
    salida: dict[str, dict[str, str]] = {}
    with ruta.open(encoding=_codificacion(ruta), newline="") as fh:
        lector = csv.DictReader(fh)
        for fila in lector:
            clave = (fila.get("nemónico") or fila.get("nemonico") or "").strip()
            if not clave:
                continue
            salida[clave] = {
                "descripcion": (fila.get("nombre_campo") or "").strip(),
                "tipo": (fila.get("tipo") or "").strip(),
                "longitud": (fila.get("longitud") or "").strip(),
                "catalogo": (fila.get("catálogo") or fila.get("catalogo") or "").strip(),
                "rango": (fila.get("rango_claves") or "").strip(),
            }
    return salida


def leer_catalogo(ruta: Path) -> dict[str, str]:
    """Lee un catálogo de códigos y devuelve ``{clave: descripción}``.

    Los catálogos de INEGI tienen dos columnas sin encabezado estándar; se usan
    la primera como clave y la última como descripción.
    """
    if not ruta.exists():
        return {}
    salida: dict[str, str] = {}
    with ruta.open(encoding=_codificacion(ruta), newline="") as fh:
        reader = csv.reader(fh)
        for fila in reader:
            if len(fila) < 2:
                continue
            clave = fila[0].strip().strip('"')
            if not clave or clave.lower() in {"clave", "cve", "id"}:
                continue
            salida[clave] = fila[-1].strip().strip('"')
    return salida
