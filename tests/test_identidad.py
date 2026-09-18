"""Pruebas de la identidad del proyecto: emblema, programa y autoría.

Estas pruebas cubren el modo de fallo más probable de este tipo de cambio: que el
recurso gráfico se quede **fuera del repositorio** y el logotipo no aparezca al
clonar. Por eso no basta con comprobar que el archivo existe en disco: también se
verifica que git no lo esté ignorando.

El resto de las aserciones comprueban que la leyenda de autoría y el nombre del
programa estén donde el usuario los pidió (README y tablero), y que las cadenas
coincidan con las que usa la interfaz, para que no se desincronicen.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from enigh import config

RAIZ = Path(__file__).resolve().parent.parent
LOGO = RAIZ / "assets" / "logo_mcd.png"
README = RAIZ / "README.md"

# Textos exigidos. Se comparan contra el módulo del tablero en la prueba de
# coherencia, en lugar de duplicar la cadena completa aquí.
PROGRAMA = "Maestría en Ciencia de Datos"
CREDITO_FRAGMENTOS = (
    "Julio Waissman",
    "julio.waissman@unison.mx",
    "DeepSeek Harness",
    "Streamlit",
)


# --------------------------------------------------------------------------
# El emblema
# --------------------------------------------------------------------------


def test_el_emblema_existe_y_es_un_png_razonable():
    assert LOGO.is_file(), f"falta el emblema en {LOGO}"
    assert LOGO.stat().st_size > 0
    # Es un PNG de 100x100 con transparencia; el límite evita que alguien
    # reemplace el emblema por un archivo enorme sin darse cuenta.
    assert LOGO.stat().st_size < 200_000, "el emblema pesa demasiado para versionarlo"
    assert LOGO.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", "no es un PNG válido"


def test_el_emblema_es_una_imagen_legible():
    """Comprueba que el archivo se pueda abrir como imagen y sus dimensiones."""
    imagen = pytest.importorskip("PIL.Image")
    with imagen.open(LOGO) as im:
        ancho, alto = im.size
    assert ancho > 0 and alto > 0
    # El diseño del tablero asume un emblema cuadrado y pequeño. Si alguien lo
    # reemplaza por un banner panorámico, esto lo avisa.
    assert ancho == alto, f"se esperaba un emblema cuadrado, es {ancho}x{alto}"
    assert ancho >= 64, "el emblema es demasiado pequeño para mostrarlo con nitidez"


def test_el_emblema_no_esta_ignorado_por_git():
    """Regresión: si ``assets/`` cae en ``.gitignore``, el logo no llega al clon."""
    resultado = subprocess.run(
        ["git", "check-ignore", "--quiet", str(LOGO)],
        capture_output=True,
        cwd=RAIZ,
    )
    # ``git check-ignore`` devuelve 0 cuando el archivo SÍ está ignorado.
    assert resultado.returncode != 0, (
        f"{LOGO.relative_to(RAIZ)} está ignorado por git: el emblema no se subiría "
        "al repositorio y no aparecería al clonar."
    )


def test_el_emblema_esta_versionado_o_listo_para_versionarse():
    """Debe estar en el índice de git (o aparecer como archivo nuevo a agregar)."""
    rastreado = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(LOGO.relative_to(RAIZ))],
        capture_output=True,
        cwd=RAIZ,
    )
    if rastreado.returncode == 0:
        return  # ya está en el índice
    estado = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", "assets/"],
        capture_output=True,
        text=True,
        cwd=RAIZ,
    )
    assert "logo_mcd.png" in estado.stdout, (
        "el emblema no está versionado ni aparece como archivo nuevo por agregar"
    )


def test_la_app_resuelve_la_ruta_del_emblema():
    """La ruta que usa el tablero debe apuntar al archivo real."""
    import app as tablero

    assert tablero.RUTA_LOGO_MCD == LOGO
    assert tablero.LOGO_MCD_DISPONIBLE is True


# --------------------------------------------------------------------------
# Los textos exigidos
# --------------------------------------------------------------------------


def test_el_readme_menciona_el_programa_y_la_autoria():
    texto = README.read_text(encoding="utf-8")

    assert PROGRAMA in texto, "el README no menciona la Maestría en Ciencia de Datos"
    for fragmento in CREDITO_FRAGMENTOS:
        assert fragmento in texto, f"el README no incluye {fragmento!r} en la autoría"


def test_el_readme_muestra_el_emblema():
    """El README debe referenciar el emblema con una ruta que exista."""
    texto = README.read_text(encoding="utf-8")
    assert "assets/logo_mcd.png" in texto, "el README no incluye el emblema"

    # La ruta declarada en el Markdown debe resolver a un archivo existente, o
    # GitHub mostraría una imagen rota.
    assert LOGO.is_file()


def test_las_cadenas_del_tablero_coinciden_con_las_esperadas():
    """Evita que el crédito del código y el de las pruebas se desincronicen."""
    import app as tablero

    assert tablero.PROGRAMA_MCD == PROGRAMA
    for fragmento in CREDITO_FRAGMENTOS:
        assert fragmento in tablero.CREDITO, f"falta {fragmento!r} en app.CREDITO"
    assert tablero.CREDITO.endswith("."), "la leyenda debe cerrar con punto"
    assert tablero.SITIO_MCD.startswith("https://"), "el sitio de la MCD debe ser https"


def test_la_configuracion_del_proyecto_sigue_intacta():
    """Comprobación de humo: el branding no debe alterar la carga de configuración."""
    assert config.cargar_configuracion().anios_habilitados == [2020, 2022, 2024]
