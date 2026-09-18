"""Pruebas de la descarga robusta.

La trampa central que se prueba aquí es específica de INEGI y está verificada:
una ruta inexistente responde **HTTP 200 con un HTML de ~2,263 bytes** en lugar
de 404. Si la descarga confiara en el código de estado, guardaría un archivo de
error como si fueran microdatos y el fallo aparecería mucho después, al leer el
CSV con pandas.

Todas las pruebas usan dobles de ``requests``: no tocan la red.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest
import requests

from enigh import config, descarga

HTML_ERROR_INEGI = (
    b"<!DOCTYPE html><html><head><title>Descarga masiva</title></head><body>"
    b"<p>El archivo que estas buscando no existe o no cumple con el estandar "
    b"de datos abiertos.</p></body></html>"
)


class RespuestaFalsa:
    """Doble de ``requests.Response`` para streaming."""

    def __init__(self, contenido: bytes, status_code: int = 200, headers: dict | None = None):
        self._contenido = contenido
        self.status_code = status_code
        self.headers = headers or {"Content-Length": str(len(contenido))}

    def iter_content(self, chunk_size: int = 1 << 20):
        for inicio in range(0, len(self._contenido), chunk_size):
            yield self._contenido[inicio : inicio + chunk_size]

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class SesionFalsa:
    """Doble de ``requests.Session`` que sirve una secuencia de respuestas."""

    def __init__(self, respuestas: list, head_headers: dict | None = None):
        self.respuestas = list(respuestas)
        self.head_headers = head_headers or {}
        self.headers: dict = {}
        self.llamadas: list[dict] = []

    def head(self, url, **kwargs):
        return RespuestaFalsa(b"", 200, self.head_headers)

    def get(self, url, headers=None, **kwargs):
        self.llamadas.append({"url": url, "headers": dict(headers or {})})
        if not self.respuestas:
            raise requests.ConnectionError("sin respuestas programadas")
        siguiente = self.respuestas.pop(0)
        if isinstance(siguiente, Exception):
            raise siguiente
        return siguiente

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


@pytest.fixture(autouse=True)
def umbral_minimo_pequeno(monkeypatch):
    """Baja el umbral de tamaño para no tener que generar ZIP de prueba de 1 MB.

    El umbral real es de 1 MB (los ZIP de INEGI pesan 90-103 MB). Estos ZIP de
    prueba pesan unos pocos KB y comprimen bien, así que sin este ajuste todas
    las pruebas fallarían por el guardia de tamaño en lugar de por la lógica que
    se quiere verificar. La prueba del guardia real lo restaura explícitamente.
    """
    monkeypatch.setattr(descarga, "TAMANO_MINIMO_VALIDO", 64)


def _zip_valido() -> bytes:
    """ZIP mínimo con la estructura de tablas de INEGI."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("conjunto_de_datos/hogares.csv", "a,b\n1,2\n")
    return buffer.getvalue()


@pytest.fixture
def cfg_prueba(tmp_path: Path, monkeypatch) -> config.Configuracion:
    """Configuración apuntando a directorios temporales."""
    monkeypatch.setattr(config, "DIR_RAW", tmp_path / "raw")
    monkeypatch.setattr(config, "DIR_INTERIM", tmp_path / "interim")
    monkeypatch.setattr(config, "DIR_PROCESSED", tmp_path / "processed")
    return config.cargar_configuracion()


def test_html_de_error_de_inegi_se_detecta_como_fallo(tmp_path: Path, monkeypatch):
    """HTTP 200 + HTML no es un ZIP válido y debe fallar de forma explícita."""
    monkeypatch.setattr(config, "DIR_RAW", tmp_path)
    sesion = SesionFalsa([RespuestaFalsa(HTML_ERROR_INEGI, 200, {"Content-Type": "text/html"})])
    monkeypatch.setattr(descarga.requests, "Session", lambda: sesion)

    with pytest.raises(descarga.ErrorDeDescarga) as excinfo:
        descarga.descargar_anio(2024)
    assert "HTML" in str(excinfo.value) or "ZIP" in str(excinfo.value)


def test_descarga_exitosa_valida_y_renombra(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(config, "DIR_RAW", tmp_path)
    contenido = _zip_valido()
    sesion = SesionFalsa([RespuestaFalsa(contenido)])
    monkeypatch.setattr(descarga.requests, "Session", lambda: sesion)

    resultado = descarga.descargar_anio(2024)

    assert resultado.ruta.exists()
    assert resultado.bytes_descargados == len(contenido)
    assert resultado.reutilizado is False
    # El archivo parcial no debe quedar tirado.
    assert not resultado.ruta.with_suffix(".zip.part").exists()


def test_zip_corrupto_se_rechaza(tmp_path: Path, monkeypatch):
    """Un archivo con firma PK pero contenido corrupto no debe pasar."""
    monkeypatch.setattr(config, "DIR_RAW", tmp_path)
    corrupto = b"PK\x03\x04" + b"\x00" * (descarga.TAMANO_MINIMO_VALIDO + 10)
    sesion = SesionFalsa([RespuestaFalsa(corrupto)])
    monkeypatch.setattr(descarga.requests, "Session", lambda: sesion)

    with pytest.raises(descarga.ErrorDeDescarga):
        descarga.descargar_anio(2024)


def test_archivo_ya_presente_no_se_vuelve_a_descargar(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(config, "DIR_RAW", tmp_path)
    destino = config.ruta_zip(2024)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_bytes(_zip_valido())

    sesion = SesionFalsa([])
    monkeypatch.setattr(descarga.requests, "Session", lambda: sesion)

    resultado = descarga.descargar_anio(2024)
    assert resultado.reutilizado is True
    assert sesion.llamadas == [], "no debería haber pedido nada por HTTP"


def test_forzar_vuelve_a_descargar(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(config, "DIR_RAW", tmp_path)
    destino = config.ruta_zip(2024)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_bytes(b"contenido viejo")

    contenido = _zip_valido()
    sesion = SesionFalsa([RespuestaFalsa(contenido)])
    monkeypatch.setattr(descarga.requests, "Session", lambda: sesion)

    resultado = descarga.descargar_anio(2024, forzar=True)
    assert resultado.reutilizado is False
    assert resultado.ruta.stat().st_size == len(contenido)


def test_reintenta_con_backoff_y_termina_bien(tmp_path: Path, monkeypatch):
    """Si el primer intento falla y el segundo funciona, la descarga debe lograrlo."""
    monkeypatch.setattr(config, "DIR_RAW", tmp_path)
    monkeypatch.setattr(descarga.time, "sleep", lambda _s: None)

    contenido = _zip_valido()
    sesion = SesionFalsa([
        requests.ConnectionError("corte de red"),
        RespuestaFalsa(contenido),
    ])
    monkeypatch.setattr(descarga.requests, "Session", lambda: sesion)

    resultado = descarga.descargar_anio(2024)
    assert resultado.ruta.exists()
    assert resultado.reutilizado is False


def test_agota_reintentos_y_falla_con_mensaje_util(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(config, "DIR_RAW", tmp_path)
    monkeypatch.setattr(descarga.time, "sleep", lambda _s: None)

    sesion = SesionFalsa([requests.ConnectionError("caído")] * descarga.REINTENTOS)
    monkeypatch.setattr(descarga.requests, "Session", lambda: sesion)

    with pytest.raises(descarga.ErrorDeDescarga) as excinfo:
        descarga.descargar_anio(2024)
    assert "intentos" in str(excinfo.value)


def test_reanuda_desde_archivo_parcial(tmp_path: Path, monkeypatch):
    """Con un .part existente, debe pedir el resto con cabecera Range."""
    monkeypatch.setattr(config, "DIR_RAW", tmp_path)
    parcial = config.ruta_zip(2024).with_suffix(".zip.part")
    parcial.parent.mkdir(parents=True, exist_ok=True)

    contenido = _zip_valido()
    mitad = len(contenido) // 2
    # El .part contiene la primera mitad; el servidor manda la segunda.
    parcial.write_bytes(contenido[:mitad])

    sesion = SesionFalsa([RespuestaFalsa(contenido[mitad:], status_code=206)])
    monkeypatch.setattr(descarga.requests, "Session", lambda: sesion)

    resultado = descarga.descargar_anio(2024)

    assert resultado.ruta.read_bytes() == contenido
    assert sesion.llamadas, "debería haber hecho una petición GET"
    assert sesion.llamadas[0]["headers"].get("Range") == f"bytes={mitad}-"


def test_servidor_que_ignora_range_no_duplica_bytes(tmp_path: Path, monkeypatch):
    """Si el servidor responde 200 en vez de 206, se reinicia en lugar de concatenar."""
    monkeypatch.setattr(config, "DIR_RAW", tmp_path)
    parcial = config.ruta_zip(2024).with_suffix(".zip.part")
    parcial.parent.mkdir(parents=True, exist_ok=True)
    parcial.write_bytes(b"basura de un intento anterior")

    contenido = _zip_valido()
    sesion = SesionFalsa([RespuestaFalsa(contenido, status_code=200)])
    monkeypatch.setattr(descarga.requests, "Session", lambda: sesion)

    resultado = descarga.descargar_anio(2024)
    assert resultado.ruta.read_bytes() == contenido


def test_tamano_remoto_diminuto_aborta_antes_de_descargar(tmp_path: Path, monkeypatch):
    """El HTML de error de INEGI mide 2,263 bytes: con el umbral real debe abortar."""
    # Se restaura el umbral real porque esta prueba verifica precisamente ese guardia.
    monkeypatch.setattr(descarga, "TAMANO_MINIMO_VALIDO", 1_000_000)
    monkeypatch.setattr(config, "DIR_RAW", tmp_path)
    sesion = SesionFalsa([], head_headers={"Content-Length": "2263"})
    monkeypatch.setattr(descarga.requests, "Session", lambda: sesion)

    with pytest.raises(descarga.ErrorDeDescarga) as excinfo:
        descarga.descargar_anio(2024)
    assert "bytes" in str(excinfo.value)
    assert sesion.llamadas == [], "no debería haber descargado nada"


def test_umbral_real_rechaza_el_html_de_error_de_inegi(tmp_path: Path, monkeypatch):
    """Con el umbral de producción, el HTML de error se rechaza por tamaño."""
    monkeypatch.setattr(descarga, "TAMANO_MINIMO_VALIDO", 1_000_000)
    monkeypatch.setattr(config, "DIR_RAW", tmp_path)
    sesion = SesionFalsa([RespuestaFalsa(HTML_ERROR_INEGI)])
    monkeypatch.setattr(descarga.requests, "Session", lambda: sesion)

    with pytest.raises(descarga.ErrorDeDescarga) as excinfo:
        descarga.descargar_anio(2024)
    assert "HTML" in str(excinfo.value) or "ZIP" in str(excinfo.value)


def test_mensaje_de_error_nombra_el_archivo_de_configuracion(tmp_path: Path, monkeypatch):
    """El mensaje debe decir dónde arreglar la URL, no solo que falló."""
    monkeypatch.setattr(config, "DIR_RAW", tmp_path)
    sesion = SesionFalsa([RespuestaFalsa(HTML_ERROR_INEGI)])
    monkeypatch.setattr(descarga.requests, "Session", lambda: sesion)

    with pytest.raises(descarga.ErrorDeDescarga) as excinfo:
        descarga.descargar_anio(2024)
    assert "fuentes.yaml" in str(excinfo.value)
