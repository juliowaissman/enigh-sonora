"""Pruebas de integración con los datos ya procesados de la ENIGH.

Estas pruebas se saltan solas cuando ``data/processed`` está vacío (por ejemplo en
CI, que no descarga 287 MB). Cuando hay datos, verifican que el camino que
recorre el tablero produce resultados coherentes con lo que publica INEGI:
conteos de hogares, rangos de ingreso, orden de los deciles y presencia de todas
las entidades.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from enigh import CLAVE_SONORA, NOMBRE_SONORA, config, limpieza

pytestmark = pytest.mark.integration


def _anios_procesados() -> list[int]:
    directorio = config.DIR_PROCESSED / "hogares"
    if not directorio.exists():
        return []
    anios = []
    for ruta in directorio.glob("ano=*"):
        try:
            anios.append(int(ruta.name.split("=")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(anios)


ANIOS = _anios_procesados()

requiere_datos = pytest.mark.skipif(
    not ANIOS,
    reason="No hay datos procesados en data/processed; ejecute `make data`.",
)


@requiere_datos
@pytest.mark.parametrize("anio", ANIOS or [1999])
def test_hogares_procesados_tienen_la_forma_esperada(anio):
    hogares = limpieza.cargar_hogares(anio)

    for columna in (
        "folioviv", "foliohog", "ubica_geo", "entidad", "factor", "tot_integ",
        "ing_cor", "ing_pc", "ing_ae", "decil_nacional",
    ):
        assert columna in hogares.columns, f"falta {columna} en {anio}"

    assert len(hogares) > 10_000
    assert hogares["factor"].gt(0).all()
    assert hogares["tot_integ"].ge(1).all()
    assert hogares["decil_nacional"].between(1, 10).all()
    assert hogares["entidad"].nunique() == 32


@requiere_datos
@pytest.mark.parametrize("anio", ANIOS or [1999])
def test_sonora_esta_presente_con_muestra_suficiente(anio):
    hogares = limpieza.cargar_hogares(anio)
    sonora = hogares[hogares["entidad"] == CLAVE_SONORA]

    assert not sonora.empty, f"sin hogares de {NOMBRE_SONORA} en {anio}"
    assert len(sonora) > 1_500
    assert sonora["factor"].sum() > 500_000
    # Los nombres vienen del catálogo de INEGI dentro del propio ZIP.
    assert (sonora["desc_ent"] == NOMBRE_SONORA).all()


@requiere_datos
@pytest.mark.parametrize("anio", ANIOS or [1999])
def test_ingreso_per_capita_dentro_de_rangos_plausibles(anio):
    """Órdenes de magnitud de la ENIGH: cientos o miles de pesos trimestrales."""
    hogares = limpieza.cargar_hogares(anio)
    ingreso = hogares["ing_pc"].dropna()

    assert (ingreso >= 0).all(), "no debe haber ingreso per cápita negativo"
    # La mediana per cápita trimestral en México ronda los 20-30 mil pesos.
    assert 8_000 < ingreso.median() < 60_000
    # El máximo no debe ser absurdo (menos de 100 millones por trimestre).
    assert ingreso.max() < 1e8


@requiere_datos
@pytest.mark.parametrize("anio", ANIOS or [1999])
def test_tabla_de_deciles_es_coherente(anio):
    deciles = pd.read_parquet(config.ruta_agregados(anio, "deciles_sonora"))

    assert len(deciles) == 10
    assert deciles["decil"].tolist() == list(range(1, 11))
    assert deciles["participacion_ingreso"].sum() == pytest.approx(100.0, abs=0.01)
    # Monotonía del ingreso promedio.
    assert deciles.sort_values("decil")["ing_pc"].is_monotonic_increasing
    # El Gini intra-decil debe ser menor que el global: son cosas distintas.
    if "gini_global" in deciles.columns and "gini_dentro_del_decil" in deciles.columns:
        assert deciles["gini_dentro_del_decil"].max() < deciles["gini_global"].iloc[0]


@requiere_datos
@pytest.mark.parametrize("anio", ANIOS or [1999])
def test_agregado_por_entidad_cubre_las_32_entidades(anio):
    entidades = pd.read_parquet(config.ruta_agregados(anio, "entidades"))

    assert len(entidades) == 32
    assert entidades["entidad"].nunique() == 32
    assert entidades["es_sonora"].sum() == 1
    assert entidades["nom_entidad"].notna().all()
    assert entidades["hogares_expandidos"].gt(0).all()


@requiere_datos
@pytest.mark.parametrize("anio", ANIOS or [1999])
def test_municipios_de_sonora_traen_clave_cvegeo_y_marca_de_fiabilidad(anio):
    municipios = pd.read_parquet(config.ruta_agregados(anio, "municipios_sonora"))

    assert not municipios.empty
    assert "clave_cvegeo" in municipios.columns
    assert "fiable" in municipios.columns
    # Todas las claves deben empezar con 26 (Sonora) y tener 5 dígitos.
    claves = municipios["clave_cvegeo"].dropna().astype(str)
    assert claves.str.startswith(CLAVE_SONORA).all()
    assert claves.str.len().eq(5).all()
    # Nunca se publica un promedio municipal por debajo del umbral configurado.
    minimo = config.cargar_configuracion().umbrales.hogares_min_municipio
    assert (municipios.loc[~municipios["fiable"], "n_hogares"] < minimo).all()


@requiere_datos
def test_el_ingreso_per_capita_de_sonora_es_menor_que_el_del_hogar():
    """Invariante aritmética: el hogar tiene al menos un integrante."""
    anio = ANIOS[-1]
    hogares = limpieza.cargar_hogares(anio)
    sonora = hogares[hogares["entidad"] == CLAVE_SONORA]

    assert (sonora["ing_pc"] <= sonora["ing_cor"] + 0.01).all()
    assert (sonora["ing_ae"] >= sonora["ing_pc"] - 0.01).all()


@requiere_datos
@pytest.mark.skipif(len(ANIOS) < 2, reason="Se necesitan al menos dos años procesados.")
def test_la_serie_de_deciles_cubre_todos_los_anios_y_deciles():
    serie = pd.read_parquet(
        config.DIR_PROCESSED / "agregados" / "serie_deciles_sonora.parquet"
    )

    assert set(serie["anio"].unique()) == set(ANIOS)
    assert set(serie["decil"].unique()) == set(range(1, 11))
    assert len(serie) == 10 * len(ANIOS)


@requiere_datos
@pytest.mark.skipif(len(ANIOS) < 2, reason="Se necesitan al menos dos años procesados.")
def test_serie_de_desigualdad_tiene_ambitos_y_anios():
    serie = pd.read_parquet(
        config.DIR_PROCESSED / "agregados" / "serie_desigualdad.parquet"
    )

    assert {"Nacional", NOMBRE_SONORA} <= set(serie["ambito"].unique())
    assert set(serie["anio"].unique()) == set(ANIOS)
    assert serie["gini"].between(0, 1).all()
    # El Gini de un país nunca es 0.17: si saliera así, se leyó la columna
    # intra-decil por error.
    assert serie["gini"].min() > 0.30


@requiere_datos
def test_el_gini_de_sonora_esta_en_un_rango_creible():
    """El Gini estatal mexicano ronda 0.35-0.55; valores fuera indican un error."""
    anio = ANIOS[-1]
    hogares = limpieza.cargar_hogares(anio)
    sonora = hogares[hogares["entidad"] == CLAVE_SONORA]

    from enigh.metricas import gini

    valor = gini(sonora["ing_pc"], sonora["factor"])
    assert 0.30 < valor < 0.60, f"Gini de {NOMBRE_SONORA} fuera de rango: {valor}"


@requiere_datos
def test_manifiesto_documenta_la_ejecucion():
    import json

    with config.ARCHIVO_MANIFIESTO.open(encoding="utf-8") as fh:
        manifiesto = json.load(fh)

    assert manifiesto["anios"], "el manifiesto debe listar los años procesados"
    assert set(manifiesto["anios"][0]["reporte_limpieza"]), "sin reporte de limpieza"
    for entrada in manifiesto["anios"]:
        assert entrada["hogares_en_muestra"] > 10_000
        assert entrada["hogares_sonora_muestra"] > 1_500
        # La identidad de ing_cor debe haberse verificado sin discrepancias.
        identidad = entrada["reporte_limpieza"].get("identidad_ing_cor")
        if isinstance(identidad, dict):
            assert identidad["discrepancias"] == 0


@requiere_datos
def test_indicadores_de_bienestar_se_pudieron_construir():
    """Los indicadores de las tres tablas auxiliares deben estar presentes.

    OJO: los indicadores declarados en ``bienestar.py`` (``tiene_internet``,
    ``tiene_drenaje``...) se calculan en el catálogo a partir de las columnas
    crudas (``conex_inte``, ``drenaje``...), no se materializan en el Parquet de
    hogares. Aquí se verifica lo que sí queda en el archivo.
    """
    anio = ANIOS[-1]
    hogares = limpieza.cargar_hogares(anio)

    # Derivados de la tabla de personas, agregados a hogar.
    de_personas = ["pct_segsoc", "pct_asis_esc", "pct_alfabeta", "pct_etnia"]
    # Crudos de hogares y viviendas que consume el catálogo de bienestar.
    crudos = [
        "dificultad_alimentaria", "inseg_alimentaria_severa",
        "conex_inte", "celular", "tarjeta", "num_compu",
        "drenaje", "excusado", "disp_elect", "agua_ent", "mat_pisos", "combus",
        "cuart_dorm", "hacinamiento", "sexo_jefe", "educa_jefe",
    ]
    faltantes = [c for c in de_personas + crudos if c not in hogares.columns]
    assert not faltantes, f"faltan columnas de bienestar: {faltantes}"

    # Las proporciones deben estar en [0, 1] cuando hay dato.
    for columna in ("pct_segsoc", "pct_asis_esc", "pct_alfabeta"):
        valores = pd.to_numeric(hogares[columna], errors="coerce").dropna()
        assert valores.between(0, 1).all(), f"{columna} fuera de [0, 1]"

    # Regresión: la conversión silenciosamente fallida de ``pd.to_numeric`` sobre
    # el dtype ``string`` dejaba estos indicadores en un 100 % constante. Cualquier
    # valor degenerado debe hacer fallar esta prueba.
    for columna in ("pct_segsoc", "pct_asis_esc", "pct_alfabeta"):
        valores = pd.to_numeric(hogares[columna], errors="coerce").dropna()
        assert valores.nunique() > 10, (
            f"{columna} tiene solo {valores.nunique()} valores distintos: "
            "señal de conversión numérica fallida"
        )
        assert not (valores == 1.0).all(), f"{columna} es constantemente 100 %"


@requiere_datos
def test_cobertura_de_seguridad_social_es_creible():
    """Control de orden de magnitud contra las cifras publicadas de la ENIGH.

    En México, entre 37 % y 45 % de la población reporta cobertura de seguridad
    social en la ENIGH. Un valor cercano a 0 % o a 100 % indica un error de
    codificación (INEGI usa 1 = sí, 2 = no).
    """
    anio = ANIOS[-1]
    hogares = limpieza.cargar_hogares(anio)

    num = pd.to_numeric(hogares["segsoc_num"], errors="coerce")
    den = pd.to_numeric(hogares["segsoc_den"], errors="coerce")
    tasa = float((num * hogares["factor"]).sum() / (den * hogares["factor"]).sum())

    assert 0.30 < tasa < 0.50, f"cobertura de seguridad social implausible: {tasa:.2%}"


@requiere_datos
def test_indicadores_de_alimentacion_tienen_prevalencia_creible():
    """Las dos medidas escalonadas deben ser binarias y estar bien ordenadas.

    La medida amplia (alguna dificultad) ronda el 95 % porque el reactivo 1 es
    una simple preocupación; la severa (hambre o una comida al día) debe ser
    sustancialmente menor. Si la severa saliera mayor que la amplia, la
    definición estaría mal.
    """
    anio = ANIOS[-1]
    hogares = limpieza.cargar_hogares(anio)
    factor = hogares["factor"].to_numpy(dtype=float)
    peso_total = factor.sum()

    def prevalencia(columna: str) -> float:
        valor = pd.to_numeric(hogares[columna], errors="coerce").to_numpy(dtype=float)
        return float(np.nansum(valor * factor) / peso_total)

    amplia = prevalencia("dificultad_alimentaria")
    severa = prevalencia("inseg_alimentaria_severa")

    # La medida amplia debe estar en un rango alto pero no degenerado.
    assert 0.50 < amplia < 1.0, f"dificultad alimentaria implausible: {amplia:.1%}"
    # La severa debe ser claramente menor y de magnitud creíble.
    assert 0.02 < severa < 0.45, f"inseguridad severa implausible: {severa:.1%}"
    assert severa < amplia, "la medida severa no puede superar a la amplia"

    for columna in ("dificultad_alimentaria", "inseg_alimentaria_severa"):
        valores = pd.to_numeric(hogares[columna], errors="coerce").dropna()
        assert valores.nunique() == 2, f"{columna} debe ser binario"
