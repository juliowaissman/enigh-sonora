"""Pruebas de humo del tablero con el simulador de Streamlit.

Estas pruebas ejecutan ``app.py`` de verdad con ``AppTest``, la herramienta
oficial de Streamlit para probar aplicaciones sin navegador. Detectarían, por
ejemplo, un nombre de columna mal escrito o una propiedad inválida de Plotly, que
son fallos que solo aparecen al abrir la página correspondiente.

Requieren los datos procesados, así que se saltan si ``data/processed`` está
vacío.
"""

from __future__ import annotations

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest

from enigh import config, limpieza  # noqa: E402

PAGINAS = [
    "Panorama",
    "Deciles",
    "Mapas",
    "Serie de tiempo",
    "Bienestar",
    "Metodología y calidad",
]


def _hay_datos() -> bool:
    return (config.DIR_PROCESSED / "hogares").exists() and any(
        (config.DIR_PROCESSED / "hogares").glob("ano=*")
    )


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _hay_datos(),
        reason="No hay datos procesados; ejecute `make data` antes.",
    ),
]


@pytest.fixture(autouse=True)
def _cerrar_runtime_de_streamlit():
    """Libera el runtime de Streamlit después de cada prueba.

    Sin este cierre, los runtimes se acumulan en el mismo proceso y las pruebas
    parametrizadas se cuelgan a partir de la segunda: cada una pasa si se ejecuta
    sola, pero la suite completa se queda esperando indefinidamente. Es un
    problema de limpieza de Streamlit, no del código del tablero.
    """
    yield
    try:
        from streamlit.runtime import Runtime

        for runtime in Runtime._instances.values():
            runtime.stop()
        Runtime._instances.clear()
    except Exception:  # noqa: BLE001 - la limpieza nunca debe hacer fallar una prueba
        pass


def _aplicacion_nueva(timeout: int = 180):
    """Crea una instancia de AppTest y la ejecuta una vez."""
    at = AppTest.from_file("app.py", default_timeout=timeout)
    at.run()
    return at


@pytest.fixture(scope="module")
def aplicacion():
    """Instancia de AppTest ya ejecutada una vez."""
    return _aplicacion_nueva()


def test_el_tablero_arranca_sin_excepciones(aplicacion):
    assert not aplicacion.exception, f"el tablero falló al arrancar: {aplicacion.exception}"


def test_el_tablero_tiene_los_controles_esperados(aplicacion):
    # Vista, año, métrica, expresión y definición de deciles.
    assert len(aplicacion.sidebar.radio) == 4
    assert len(aplicacion.sidebar.selectbox) == 1
    # Añadir el emblema al sidebar no debe alterar los controles.
    assert len(aplicacion.sidebar.get("imgs")) >= 1


def test_el_tablero_muestra_la_identidad_de_la_mcd(aplicacion):
    """El emblema, el programa y la autoría deben verse en la barra lateral.

    Se comprueba sobre la instancia que ya está creada para no añadir otra
    ejecución completa del tablero (cada una tarda alrededor de un segundo).
    """
    import app as tablero

    captions = " ".join(c.value for c in aplicacion.sidebar.caption)
    markdown = " ".join(m.value for m in aplicacion.sidebar.markdown)
    texto = f"{captions} {markdown}"

    assert tablero.PROGRAMA_MCD in texto, "falta el nombre del programa en el sidebar"
    assert "Julio Waissman" in texto, "falta la autoría en el sidebar"
    assert "julio.waissman@unison.mx" in texto, "falta el correo en el sidebar"
    assert "DeepSeek Harness" in texto, "falta la mención de DeepSeek Harness"


@pytest.mark.parametrize("pagina", PAGINAS)
def test_cada_vista_se_ejecuta_sin_excepciones(pagina):
    at = AppTest.from_file("app.py", default_timeout=180)
    at.run()
    if at.exception:
        pytest.skip(f"el tablero no arrancó: {at.exception}")
    at.sidebar.radio[0].set_value(pagina).run()

    assert not at.exception, f"la vista {pagina!r} lanzó {at.exception}"
    assert not [e.value for e in at.error], (
        f"la vista {pagina!r} mostró errores: {[e.value for e in at.error]}"
    )


def test_la_vista_de_metodologia_incluye_la_seccion_de_creditos():
    """La atribución debe estar también en el cuerpo, con el emblema más grande."""
    import app as tablero

    at = _aplicacion_nueva()
    if at.exception:
        pytest.skip(f"el tablero no arrancó: {at.exception}")

    # La vista se llama "Metodología y calidad" en la barra lateral.
    at.sidebar.radio[0].set_value("Metodología y calidad").run()
    assert not at.exception

    subsecciones = [s.value for s in at.subheader]
    assert "Sobre este tablero" in subsecciones, (
        f"falta la sección de créditos; subsecciones: {subsecciones}"
    )

    cuerpo = " ".join(m.value for m in at.markdown)
    assert tablero.PROGRAMA_MCD in cuerpo, "falta el programa en la sección de créditos"
    assert tablero.CREDITO in cuerpo, "falta la leyenda de autoría completa"

    # El emblema se muestra en el sidebar y, además, en la sección de créditos.
    assert len(at.get("imgs")) >= 2, "falta el emblema en la sección de créditos"


@pytest.mark.parametrize(
    ("pagina", "anio", "metrica", "expresion", "cortes"),
    [
        ("Deciles", 2020, "ing_pc", "nominal", "nacionales"),
        ("Deciles", 2024, "ing_ae", "real", "propios"),
        ("Panorama", 2024, "ing_ae", "nominal", "nacionales"),
        ("Mapas", 2022, "ing_pc", "real", "propios"),
        ("Bienestar", 2024, "ing_ae", "real", "nacionales"),
    ],
)
def test_combinaciones_de_controles_no_rompen_el_tablero(
    pagina, anio, metrica, expresion, cortes
):
    """La métrica y la expresión monetaria deben poder cambiarse sin fallar.

    Es la prueba que habría detectado que el selector de "adulto equivalente"
    estaba inerte: la vista seguía leyendo la columna del per cápita.
    """
    at = AppTest.from_file("app.py", default_timeout=180)
    at.run()
    if at.exception:
        pytest.skip(f"el tablero no arrancó: {at.exception}")

    at.sidebar.radio[0].set_value(pagina)
    at.sidebar.selectbox[0].set_value(anio)
    at.sidebar.radio[1].set_value(metrica)
    at.sidebar.radio[2].set_value(expresion)
    at.sidebar.radio[3].set_value(cortes)
    at.run()

    assert not at.exception, f"combinación falló: {at.exception}"
    assert not [e.value for e in at.error]


def test_la_metrica_elegida_cambia_las_cifras_del_panorama():
    """El ingreso por adulto equivalente debe ser mayor que el per cápita.

    Con menores en el hogar el denominador equivalente es menor que el número de
    integrantes, así que el ingreso por adulto equivalente tiene que superar al
    per cápita. Si los dos mostraran el mismo número, el selector estaría inerte.
    """
    def metrica_del_kpi(metrica: str) -> float:
        at = AppTest.from_file("app.py", default_timeout=180)
        at.run()
        if at.exception:
            pytest.skip(f"el tablero no arrancó: {at.exception}")
        at.sidebar.radio[0].set_value("Panorama")
        at.sidebar.radio[1].set_value(metrica)
        at.run()
        assert not at.exception
        # El primer metric del panorama es el ingreso promedio.
        return float(at.metric[0].value.replace("$", "").replace(",", ""))

    per_capita = metrica_del_kpi("ing_pc")
    adulto_equivalente = metrica_del_kpi("ing_ae")

    assert adulto_equivalente > per_capita, (
        f"el KPI no cambió con la métrica: per cápita={per_capita}, "
        f"adulto equivalente={adulto_equivalente}. El selector está inerte."
    )


def test_la_cobertura_de_seguridad_social_se_muestra_por_decil():
    """Prueba de integración del indicador que llegó a estar constante en 100 %.

    El indicador por hogar es una proporción de integrantes, y como los hogares
    tienen entre 1 y ~8 personas, toma valores discretos (0, 0.25, 0.5, ...). Por
    eso el criterio correcto no es "cuántos valores distintos hay", sino que la
    distribución tenga dispersión real y no esté pegada a un extremo.
    """
    anio = sorted(
        int(ruta.name.split("=")[1])
        for ruta in (config.DIR_PROCESSED / "hogares").glob("ano=*")
        if ruta.name.split("=")[1].isdigit()
    )[-1]
    hogares = limpieza.cargar_hogares(anio)
    valores = hogares["pct_segsoc"].dropna().astype(float)

    assert valores.between(0, 1).all()
    # Dispersión real: con el bug de conversión el indicador era constantemente 1.
    assert valores.std() > 0.05, f"sin dispersión (std={valores.std():.4f})"
    assert valores.nunique() > 5, f"solo {valores.nunique()} valores distintos"
    assert valores.mean() < 0.95, f"casi todo en 100 % (media={valores.mean():.3f})"
    # Debe haber hogares en ambos extremos de la distribución.
    assert valores.min() == 0.0
    assert valores.max() == 1.0
