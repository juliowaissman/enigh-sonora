"""Etiquetado territorial y cartografía para los mapas.

El tablero necesita dos capas:

1. **Nacional por entidad federativa** — 32 features. La capa estatal se obtiene
   de un GeoJSON público con ``properties.name``; el empate con los datos se hace
   por **nombre normalizado** contra el catálogo ``ubica_geo.csv`` que viene
   dentro del ZIP de INEGI, no por el ``id`` ISO del GeoJSON.
2. **Municipal de Sonora** — 72 municipios con clave INEGI ``cvegeo`` de cinco
   dígitos. El empate con los microdatos es por **``ubica_geo``**, que ya trae
   ``EEMMM``, así que no depende de nombres ni de acentos.

Ambas capas se guardan versionadas en ``data/processed/geografia/`` para que el
tablero funcione sin conexión. Si una capa no se puede obtener, el módulo lo
informa y el tablero degrada de forma explícita en lugar de fallar.
"""

from __future__ import annotations

import hashlib
import json
import logging
import unicodedata
from pathlib import Path

import pandas as pd

from . import config

log = logging.getLogger(__name__)

# Fuente verificada: 32 features, 184,555 bytes, ``properties.name`` incluye
# "Sonora". Es la capa estatal que usa el tablero.
URL_ENTIDADES = (
    "https://raw.githubusercontent.com/angelnmara/geojson/master/mexicoHigh.json"
)

# --- Municipios de Sonora ---------------------------------------------------
# Fuente recomendada tras verificar varias alternativas: 72 features (el
# universo municipal completo de Sonora), claves ``cvegeo`` de 5 dígitos
# idénticas a la API oficial del INEGI (72/72, 0 discrepancias de nombre),
# WGS84 (EPSG:4326).
#
# Se fija el commit SHA en lugar de la rama porque es un repositorio personal
# sin LICENSE y con pocas estrellas: si desaparece o cambia, la descarga falla
# de forma ruidosa en vez de traer datos distintos en silencio.
#
# Atribución: el dato subyacente es el Marco Geoestadístico del INEGI (Términos
# de Libre Uso). Se atribuye al INEGI, no al repositorio que lo redistribuye.
# Para producción a largo plazo conviene bajar el MGN oficial desde
# https://www.inegi.org.mx/temas/mg/ .
URL_MUNICIPIOS_SONORA = (
    "https://raw.githubusercontent.com/MacWilliXD/INEGI-geojson/"
    "8d38f708dc4acc14afcb461f704bebb43e332a9a/geojson_descargas/AGEM_26.geojson"
)
SHA256_MUNICIPIOS_SONORA = (
    "7d9dd414a87001a54d9dc6afac62a1372cece323b5fe497f82f2e6376e50891b"
)
BYTES_MUNICIPIOS_SONORA = 2_999_749
ATRIBUCION_MUNICIPIOS = (
    "Marco Geoestadístico, INEGI. Uso libre con atribución al INEGI."
)

NOMBRE_ENTIDADES = config.DIR_GEO / "mexico_entidades.geojson"
NOMBRE_MUNICIPIOS_SONORA = config.DIR_GEO / "sonora_municipios.geojson"
NOMBRE_INDICE_MUNICIPIOS = config.DIR_GEO / "sonora_municipios_index.csv"

CABECERAS = {"User-Agent": "Mozilla/5.0 (compatible; ENIGH-Sonora/0.1)"}

# Primeras palabras que son sustantivos genéricos y no un nombre propio: si el
# nombre oficial empieza con una de ellas, no se genera alias corto.
PREFIJOS_GENERICOS = {"ESTADO", "MUNICIPIO", "PROVINCIA", "DISTRITO", "REGION"}

# Claves candidatas para el municipio en un GeoJSON, en orden de preferencia.
CLAVES_CVEGEO = ("cvegeo", "CVEGEO", "concat", "adm2_pcode")
CLAVES_CVE_ENT = ("cve_agee", "CVE_ENT", "cve_ent")
CLAVES_NOMBRE = ("nom_agem", "nomgeo", "NOM_MUN", "nom_mun", "shapeName", "name")


def normalizar_nombre(texto: str | None) -> str:
    """Normaliza un nombre de entidad o municipio para poder empatar fuentes.

    Quita acentos, mayúsculas y puntuación, y colapsa espacios. Es necesario
    porque el GeoJSON y el catálogo de INEGI escriben distinto (por ejemplo
    "México" vs "MEXICO", "Álamos" vs "Alamos"). En el caso municipal esto es
    solo un respaldo: la vía principal es el empate por clave ``cvegeo``.
    """
    if texto is None or (isinstance(texto, float) and pd.isna(texto)):
        return ""
    plano = unicodedata.normalize("NFKD", str(texto))
    sin_acentos = "".join(c for c in plano if not unicodedata.combining(c))
    limpio = "".join(c if c.isalnum() or c.isspace() else " " for c in sin_acentos)
    return " ".join(limpio.upper().split())


# --------------------------------------------------------------------------
# Descarga
# --------------------------------------------------------------------------


def _descargar_json(url: str, destino: Path) -> dict | None:
    """Descarga un JSON y lo valida mínimamente como GeoJSON."""
    import requests

    try:
        respuesta = requests.get(url, timeout=120, headers=CABECERAS)
        respuesta.raise_for_status()
        datos = respuesta.json()
    except Exception as exc:  # noqa: BLE001 - se reporta y se degrada
        log.warning("No se pudo descargar %s: %s", url, exc)
        return None

    if not isinstance(datos, dict) or "features" not in datos:
        log.warning("%s no parece un GeoJSON (sin 'features').", url)
        return None
    if not datos.get("features"):
        log.warning("%s trae 0 features.", url)
        return None

    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", encoding="utf-8") as fh:
        json.dump(datos, fh, ensure_ascii=False)
    return datos


def asegurar_cartografia(*, forzar: bool = False) -> dict[str, Path | None]:
    """Garantiza las capas geográficas en disco.

    Returns:
        ``{"entidades": Path|None, "municipios_sonora": Path|None}``. Un valor
        ``None`` significa que esa capa no está disponible y el tablero debe
        mostrar el mapa correspondiente como no disponible.
    """
    config.asegurar_directorios()
    resultado: dict[str, Path | None] = {"entidades": None, "municipios_sonora": None}

    # --- Entidades federativas ---
    if NOMBRE_ENTIDADES.exists() and not forzar:
        resultado["entidades"] = NOMBRE_ENTIDADES
        log.info("Cartografía estatal ya presente: %s", NOMBRE_ENTIDADES.name)
    else:
        datos = _descargar_json(URL_ENTIDADES, NOMBRE_ENTIDADES)
        if datos is not None:
            resultado["entidades"] = NOMBRE_ENTIDADES
            log.info("Cartografía estatal: %s features", len(datos.get("features", [])))

    # --- Municipios de Sonora ---
    if NOMBRE_MUNICIPIOS_SONORA.exists() and not forzar:
        resultado["municipios_sonora"] = NOMBRE_MUNICIPIOS_SONORA
        log.info("Cartografía municipal ya presente: %s", NOMBRE_MUNICIPIOS_SONORA.name)
    else:
        datos = _descargar_json(URL_MUNICIPIOS_SONORA, NOMBRE_MUNICIPIOS_SONORA)
        if datos is not None:
            huella = _sha256(NOMBRE_MUNICIPIOS_SONORA)
            n_features = len(datos.get("features", []))
            if huella != SHA256_MUNICIPIOS_SONORA:
                log.warning(
                    "El GeoJSON municipal descargado tiene sha256 %s, distinto del "
                    "esperado %s. Se conserva, pero revise la fuente: la capa pudo "
                    "haber cambiado.",
                    huella[:16], SHA256_MUNICIPIOS_SONORA[:16],
                )
            else:
                log.info(
                    "Cartografía municipal verificada: %s features, sha256 %s…",
                    n_features, huella[:12],
                )
            _escribir_indice_municipios(datos)
            resultado["municipios_sonora"] = NOMBRE_MUNICIPIOS_SONORA

    if resultado["municipios_sonora"] is None:
        log.warning(
            "Sin cartografía municipal de Sonora. El tablero mostrará el desglose "
            "municipal como tabla y no como mapa coroplético."
        )
    return resultado


def _sha256(ruta: Path) -> str:
    """Huella SHA-256 de un archivo, en streaming."""
    digest = hashlib.sha256()
    with ruta.open("rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            digest.update(bloque)
    return digest.hexdigest()


def _clave_cvegeo(props: dict) -> str:
    """Extrae la clave INEGI de 5 dígitos de las propiedades de una feature."""
    for clave in CLAVES_CVEGEO:
        valor = props.get(clave)
        if valor is None:
            continue
        texto = str(valor).strip()
        if texto.startswith("MX") and len(texto) == 7:
            texto = texto[2:]
        if texto.isdigit() and len(texto) <= 5:
            return texto.zfill(5)
    return ""


def _nombre_municipio(props: dict) -> str:
    for clave in CLAVES_NOMBRE:
        valor = props.get(clave)
        if valor:
            return str(valor).strip()
    return ""


def _escribir_indice_municipios(datos: dict) -> None:
    """Escribe un índice ``cvegeo → nombre`` para el empate con los microdatos."""
    filas = []
    for feature in datos.get("features", []):
        props = feature.get("properties") or {}
        cvegeo = _clave_cvegeo(props)
        if not cvegeo:
            continue
        entidad = ""
        for clave in CLAVES_CVE_ENT:
            if props.get(clave):
                entidad = str(props[clave]).strip().zfill(2)
                break
        nombre = _nombre_municipio(props)
        filas.append(
            {
                "cvegeo": cvegeo,
                "cve_ent": entidad or cvegeo[:2],
                "cve_mun": cvegeo[2:],
                "nom_mun": nombre,
                "nom_mun_norm": normalizar_nombre(nombre),
            }
        )
    if filas:
        indice = pd.DataFrame(filas).drop_duplicates("cvegeo")
        indice.to_csv(NOMBRE_INDICE_MUNICIPIOS, index=False)
        log.info("Índice municipal escrito: %s municipios", len(indice))


# --------------------------------------------------------------------------
# Lectura
# --------------------------------------------------------------------------


def cargar_geojson(ruta: Path | None) -> dict | None:
    """Lee un GeoJSON de disco, o ``None`` si no existe."""
    if ruta is None or not Path(ruta).exists():
        return None
    with Path(ruta).open(encoding="utf-8") as fh:
        return json.load(fh)


def indice_municipios() -> pd.DataFrame:
    """Índice ``cvegeo → nombre`` de los municipios de Sonora, si existe."""
    if not NOMBRE_INDICE_MUNICIPIOS.exists():
        return pd.DataFrame(columns=["cvegeo", "cve_ent", "cve_mun", "nom_mun"])
    return pd.read_csv(NOMBRE_INDICE_MUNICIPIOS, dtype={"cvegeo": str, "cve_mun": str})


def agregar_municipio(datos_hogares: pd.DataFrame) -> pd.DataFrame:
    """Añade el nombre oficial del municipio a los hogares de Sonora.

    El empate es por la clave ``ubica_geo`` (``EEMMM``), que es exactamente la
    ``cvegeo`` de 5 dígitos. No se empata por nombre: 22 de los 72 municipios de
    Sonora llevan acento y las fuentes geográficas no son consistentes.

    Si el índice no está disponible, se conserva la columna ``municipio`` con la
    clave numérica para que el tablero nunca quede sin desglose.
    """
    salida = datos_hogares.copy()
    indice = indice_municipios()

    if "ubica_geo" in salida.columns and not indice.empty:
        salida["ubica_geo"] = salida["ubica_geo"].astype(str).str.zfill(5)
        mapa = dict(zip(indice["cvegeo"], indice["nom_mun"]))
        salida["nom_municipio"] = salida["ubica_geo"].map(mapa)
        faltantes = salida["nom_municipio"].isna().sum()
        if faltantes:
            log.warning(
                "%s hogares de Sonora no empataron con el índice municipal; "
                "se usa la clave como nombre.",
                int(faltantes),
            )
        if "municipio" in salida.columns:
            salida["nom_municipio"] = salida["nom_municipio"].fillna(
                "Municipio " + salida["municipio"].astype(str)
            )
        else:
            salida["nom_municipio"] = salida["nom_municipio"].fillna(
                "Municipio " + salida["ubica_geo"].str[2:5]
            )
    elif "desc_mun" in salida.columns:
        # Sin índice: se cae al nombre del catálogo de INEGI que trae el ZIP.
        salida["nom_municipio"] = salida["desc_mun"].fillna(
            "Municipio " + salida.get("municipio", pd.Series("", index=salida.index)).astype(str)
        )
    else:
        salida["nom_municipio"] = "Sin identificar"

    return salida


def poner_clave_cvegeo(datos: pd.DataFrame) -> pd.DataFrame:
    """Garantiza una columna ``clave_cvegeo`` para empatar con el GeoJSON municipal.

    Se resuelve en orden de preferencia: la columna ``clave_cvegeo`` que ya dejó
    el pipeline, luego ``ubica_geo`` (que es ``EEMMM``) y por último la
    composición de ``entidad`` + ``municipio``.

    El empate por nombre queda descartado a propósito: 22 de los 72 municipios de
    Sonora llevan acento y las fuentes geográficas escriben indistintamente
    "Álamos" y "Alamos".
    """
    salida = datos.copy()
    if "clave_cvegeo" in salida.columns and salida["clave_cvegeo"].notna().any():
        salida["clave_cvegeo"] = salida["clave_cvegeo"].astype(str).str.zfill(5)
        return salida
    if "ubica_geo" in salida.columns:
        salida["clave_cvegeo"] = salida["ubica_geo"].astype(str).str.zfill(5)
        return salida
    if {"entidad", "municipio"}.issubset(salida.columns):
        salida["clave_cvegeo"] = (
            salida["entidad"].astype(str).str.zfill(2)
            + salida["municipio"].astype(str).str.zfill(3)
        )
        return salida
    salida["clave_cvegeo"] = pd.NA
    log.warning(
        "No se pudo construir la clave cvegeo; el mapa municipal quedará vacío "
        "y solo se mostrará la tabla."
    )
    return salida


def alias_entidad(nombre: str | None) -> set[str]:
    """Formas alternativas de un nombre de entidad, para empatar con el GeoJSON.

    El catálogo del INEGI usa los nombres oficiales largos y la capa geográfica
    los cortos, así que tres entidades no empataban y quedaban en blanco en el
    mapa (solo se dibujaban 29 de 32). Verificado con los datos reales:

    ================================  ==========
    INEGI (``ubica_geo.csv``)         GeoJSON
    ================================  ==========
    Coahuila de Zaragoza              Coahuila
    Michoacán de Ocampo               Michoacán
    Veracruz de Ignacio de la Llave   Veracruz
    ================================  ==========

    La regla general es quedarse con la parte **anterior a la primera
    preposición** del nombre oficial, que es lo que distingue la forma larga de la
    corta ("Coahuila de Zaragoza" -> "Coahuila"). Se devuelven todas las variantes
    para poder empatar en cualquiera de las dos direcciones.
    """
    if not nombre:
        return set()
    norm = normalizar_nombre(nombre)
    if not norm:
        return set()

    particulas = {"DE", "DEL", "LA", "LAS", "LOS", "Y"}
    palabras = norm.split()
    corte = next(
        (i for i, palabra in enumerate(palabras) if palabra in particulas), None
    )
    variantes = {norm}
    # El alias corto es el prefijo anterior a la partícula, pero solo si ese
    # prefijo es un nombre propio. En "Coahuila de Zaragoza" el prefijo es
    # "COAHUILA" (el nombre corto real); en "Estado de México" el prefijo es el
    # sustantivo genérico "ESTADO", que serviría para cualquier entidad y por eso
    # se descarta con esta lista corta.
    if corte:
        prefijo = " ".join(palabras[:corte])
        if prefijo not in PREFIJOS_GENERICOS:
            variantes.add(prefijo)
    return variantes


def mapa_nombres_entidades() -> dict[str, str]:
    """``{variante_normalizada: nombre_en_el_geojson}``, incluyendo los alias.

    Es el mapa que debe usar el coroplético estatal: con él se dibujan las 32
    entidades, no 29.
    """
    datos = cargar_geojson(NOMBRE_ENTIDADES)
    if datos is None:
        return {}
    salida: dict[str, str] = {}
    for feature in datos.get("features", []):
        props = feature.get("properties", {}) or {}
        nombre = props.get("name")
        if not nombre:
            continue
        # Se registran todas las variantes del nombre del GeoJSON.
        for variante in alias_entidad(str(nombre)):
            salida.setdefault(variante, str(nombre))
    return salida


def nombres_entidades() -> dict[str, str]:
    """``{nombre_normalizado: nombre_del_geojson}`` de la capa estatal."""
    datos = cargar_geojson(NOMBRE_ENTIDADES)
    if datos is None:
        return {}
    salida = {}
    for feature in datos.get("features", []):
        props = feature.get("properties", {}) or {}
        nombre = props.get("name")
        if nombre:
            salida[normalizar_nombre(nombre)] = str(nombre)
    return salida
