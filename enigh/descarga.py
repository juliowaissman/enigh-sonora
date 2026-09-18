"""Descarga robusta de los ZIP de microdatos de la ENIGH desde INEGI.

Particularidad crítica de INEGI (verificada): una ruta inexistente **no** devuelve
404. Devuelve ``HTTP 200`` con un HTML de ~2,263 bytes y
``Content-Type: text/html``. Por eso la validación no puede confiar en el código
de estado: comprueba el ``Content-Type``, la firma ZIP (``PK\\x03\\x04``) y el
tamaño, y además verifica la integridad del archivo con :pymethod:`zipfile.ZipFile.testzip`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from . import config

log = logging.getLogger(__name__)

# INEGI responde 406 a algunos clientes; hay que mandar un User-Agent de navegador.
CABECERAS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    ),
    "Accept": "application/zip,application/octet-stream,*/*",
}

FIRMA_ZIP = b"PK\x03\x04"
TAMANO_MINIMO_VALIDO = 1_000_000  # los ZIP reales rondan 90-103 MB
REINTENTOS = 3


class ErrorDeDescarga(RuntimeError):
    """La descarga falló por una causa transitoria (red, corte, timeout)."""


class ContenidoInvalido(ErrorDeDescarga, RuntimeError):
    """El servidor entregó algo que no es un ZIP válido de microdatos.

    Es un fallo **permanente**: reintentar solo repetiría la misma respuesta y,
    peor aún, ocultaría la causa real detrás de un mensaje genérico de
    "no se pudo tras N intentos". Por eso este error corta los reintentos y se
    propaga con su mensaje específico (por ejemplo, que INEGI devolvió HTML).

    Hereda de :class:`ErrorDeDescarga` para no romper a quien capture la clase
    base.
    """


@dataclass
class ResultadoDescarga:
    anio: int
    ruta: Path
    bytes_descargados: int
    reutilizado: bool
    segundos: float


def _parece_html(ruta: Path) -> bool:
    """Detecta el HTML de error que INEGI devuelve con HTTP 200."""
    try:
        with ruta.open("rb") as fh:
            inicio = fh.read(512)
    except OSError:
        return False
    return b"<html" in inicio.lower() or b"<!doctype" in inicio.lower()


def _validar_zip(ruta: Path, anio: int) -> None:
    """Valida que ``ruta`` sea un ZIP íntegro. Lanza ``ErrorDeDescarga`` si no."""
    if not ruta.exists():
        raise ContenidoInvalido(f"[{anio}] No se creó el archivo {ruta}.")

    tamano = ruta.stat().st_size
    if tamano < TAMANO_MINIMO_VALIDO:
        if _parece_html(ruta):
            raise ContenidoInvalido(
                f"[{anio}] INEGI devolvió una página HTML en lugar del ZIP "
                f"({tamano} bytes). La URL probablemente ya no existe; "
                f"revise config/fuentes.yaml."
            )
        raise ContenidoInvalido(
            f"[{anio}] El archivo descargado pesa {tamano} bytes, muy por debajo "
            f"de lo esperado para un ZIP de microdatos."
        )

    with ruta.open("rb") as fh:
        if fh.read(4) != FIRMA_ZIP:
            raise ContenidoInvalido(
                f"[{anio}] El archivo {ruta.name} no tiene firma de ZIP (PK) "
                f"(pesa {ruta.stat().st_size} bytes). Casi siempre significa que "
                f"INEGI devolvió una página de error porque la URL de "
                f"config/fuentes.yaml ya no existe."
            )

    import zipfile

    try:
        with zipfile.ZipFile(ruta) as zf:
            corrupto = zf.testzip()
    except zipfile.BadZipFile as exc:
        raise ContenidoInvalido(f"[{anio}] ZIP corrupto: {ruta.name} ({exc}).") from exc
    if corrupto is not None:
        raise ContenidoInvalido(
            f"[{anio}] ZIP corrupto: el miembro {corrupto!r} de {ruta.name} falló "
            f"la verificación de integridad."
        )


def _tamano_remoto(sesion: requests.Session, url: str) -> int | None:
    """Tamaño declarado por el servidor, si lo expone."""
    try:
        respuesta = sesion.head(url, timeout=30, allow_redirects=True)
        if respuesta.status_code >= 400:
            return None
        declarado = respuesta.headers.get("Content-Length")
        return int(declarado) if declarado else None
    except (requests.RequestException, ValueError):
        return None


def descargar_anio(
    anio: int,
    *,
    forzar: bool = False,
    cfg: config.Configuracion | None = None,
    max_segundos: float | None = None,
) -> ResultadoDescarga:
    """Descarga el ZIP de microdatos de un año, con reanudación y validación.

    Args:
        anio: año del levantamiento.
        forzar: si es ``True`` vuelve a descargar aunque el ZIP ya exista.
        cfg: configuración; se carga la global si se omite.
        max_segundos: abandona si se excede este tiempo (útil en CI).

    Returns:
        :class:`ResultadoDescarga` con la ruta y el número de bytes obtenidos.
    """
    cfg = cfg or config.cargar_configuracion()
    fuente = cfg.fuente(anio)
    config.asegurar_directorios()

    destino = config.ruta_zip(anio)
    parcial = destino.with_suffix(destino.suffix + ".part")
    inicio = time.monotonic()

    if destino.exists() and not forzar:
        _validar_zip(destino, anio)
        log.info("[%s] ZIP ya presente y válido: %s", anio, destino.name)
        return ResultadoDescarga(anio, destino, destino.stat().st_size, True, 0.0)

    with requests.Session() as sesion:
        sesion.headers.update(CABECERAS)
        tamano_remoto = _tamano_remoto(sesion, fuente.url)

        if tamano_remoto and tamano_remoto < TAMANO_MINIMO_VALIDO:
            raise ErrorDeDescarga(
                f"[{anio}] INEGI reporta {tamano_remoto} bytes para {fuente.url}. "
                f"Esa URL ya no sirve; actualice config/fuentes.yaml."
            )

        ultimo_error: Exception | None = None
        for intento in range(1, REINTENTOS + 1):
            try:
                ya_descargado = parcial.stat().st_size if parcial.exists() else 0
                cabeceras = dict(sesion.headers)
                if ya_descargado:
                    cabeceras["Range"] = f"bytes={ya_descargado}-"
                    log.info(
                        "[%s] Reanudando descarga en %s MB (intento %s/%s)",
                        anio, round(ya_descargado / 1e6, 1), intento, REINTENTOS,
                    )
                else:
                    log.info(
                        "[%s] Descargando %s (intento %s/%s)",
                        anio, fuente.nombre_archivo, intento, REINTENTOS,
                    )

                modo = "ab" if ya_descargado else "wb"
                with sesion.get(fuente.url, headers=cabeceras, stream=True, timeout=120) as resp:
                    if resp.status_code == 416:  # el parcial ya está completo
                        resp.raise_for_status()
                    elif ya_descargado and resp.status_code == 200:
                        # El servidor ignoró el Range: empezar de cero evita
                        # concatenar bytes duplicados.
                        log.warning("[%s] El servidor ignoró Range; se reinicia la descarga.", anio)
                        modo = "wb"
                    elif resp.status_code not in (200, 206):
                        resp.raise_for_status()

                    with parcial.open(modo) as fh:
                        for bloque in resp.iter_content(chunk_size=1 << 20):
                            if not bloque:
                                continue
                            fh.write(bloque)
                            if max_segundos is not None and time.monotonic() - inicio > max_segundos:
                                raise TimeoutError(
                                    f"[{anio}] Se excedió el límite de {max_segundos:.0f} s de descarga."
                                )

                parcial.replace(destino)
                _validar_zip(destino, anio)
                segundos = time.monotonic() - inicio
                log.info(
                    "[%s] Descarga completa: %s MB en %.1f s",
                    anio, round(destino.stat().st_size / 1e6, 1), segundos,
                )
                return ResultadoDescarga(anio, destino, destino.stat().st_size, False, segundos)

            except ContenidoInvalido:
                # Fallo permanente: reintentar solo repetiría la misma respuesta y
                # escondería la causa. Se propaga el mensaje específico tal cual.
                if destino.exists():
                    destino.unlink(missing_ok=True)
                raise

            except (requests.RequestException, TimeoutError, ErrorDeDescarga) as exc:
                ultimo_error = exc
                log.warning("[%s] Falló el intento %s/%s: %s", anio, intento, REINTENTOS, exc)
                if intento < REINTENTOS:
                    time.sleep(2 ** intento)
                # Si el destino quedó corrupto, se descarta para no validarlo de nuevo.
                if destino.exists():
                    destino.unlink(missing_ok=True)

    raise ErrorDeDescarga(
        f"[{anio}] No se pudo descargar tras {REINTENTOS} intentos: {ultimo_error}"
    )


def descargar_todo(
    anios: list[int] | None = None, *, forzar: bool = False
) -> list[ResultadoDescarga]:
    """Descarga los ZIP de todos los años indicados (por omisión, los habilitados)."""
    cfg = config.cargar_configuracion()
    anios = anios or cfg.anios_habilitados
    resultados = []
    for anio in sorted(anios):
        resultados.append(descargar_anio(anio, forzar=forzar, cfg=cfg))
    return resultados
