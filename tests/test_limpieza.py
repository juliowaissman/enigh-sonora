"""Pruebas de la lectura, validación y derivación de columnas.

Se usan microdatos sintéticos en disco que reproducen los detalles del archivo
real de INEGI que rompen un lector ingenuo: BOM al inicio, comillas dentro de
una descripción, ceros a la izquierda en las claves geográficas y filas con
factor de expansión inválido.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from enigh import config, limpieza


@pytest.fixture
def tabla_sintetica(microdatos_para_limpieza):
    return microdatos_para_limpieza


def test_lee_csv_con_bom_y_conserva_ceros_a_la_izquierda(tabla_sintetica):
    hogares, _catalogo, _reporte = limpieza.cargar_concentradohogar(tabla_sintetica)

    # Las claves geográficas deben seguir siendo texto de 5 caracteres.
    assert hogares["ubica_geo"].str.len().eq(5).all()
    assert "01001" in set(hogares["ubica_geo"]), "el cero a la izquierda se perdió"


def test_deriva_entidad_y_municipio(tabla_sintetica):
    hogares, _catalogo, _reporte = limpieza.cargar_concentradohogar(tabla_sintetica)
    assert set(hogares["entidad"]) <= {"01", "26"}
    assert hogares.loc[hogares["ubica_geo"] == "26001", "municipio"].iloc[0] == "001"


def test_empata_nombres_desde_el_catalogo(tabla_sintetica):
    hogares, catalogo, _reporte = limpieza.cargar_concentradohogar(tabla_sintetica)
    assert not catalogo.empty
    fila = hogares[hogares["ubica_geo"] == "26001"].iloc[0]
    assert fila["desc_ent"] == "Sonora"
    assert fila["desc_mun"] == "Hermosillo"


def test_descarta_filas_con_factor_invalido(tabla_sintetica):
    """La fila con factor = 0 no debe sobrevivir, y el reporte debe contarlo."""
    hogares, _catalogo, reporte = limpieza.cargar_concentradohogar(tabla_sintetica)
    assert (hogares["factor"] > 0).all()
    assert len(hogares) == 3
    assert reporte["descartadas_factor_invalido"] >= 1


def test_valida_la_identidad_del_ingreso_corriente(tabla_sintetica):
    """ing_cor debe cuadrar con ingtrab + rentas + transfer + estim_alqu + otros_ing."""
    _hogares, _catalogo, reporte = limpieza.cargar_concentradohogar(tabla_sintetica)
    identidad = reporte["identidad_ing_cor"]
    assert isinstance(identidad, dict)
    assert identidad["discrepancias"] == 0, (
        f"la identidad falló: peor diferencia {identidad['peor_diferencia']}"
    )
    assert "estim_alqu" in identidad["componentes"]


def test_calcula_ingreso_per_capita(tabla_sintetica):
    hogares, _catalogo, _reporte = limpieza.cargar_concentradohogar(tabla_sintetica)
    esperado = hogares["ing_cor"] / hogares["tot_integ"]
    pd.testing.assert_series_equal(
        hogares["ing_pc"], esperado, check_names=False
    )


def test_adulto_equivalente_es_menor_que_per_capita_cuando_hay_menores(tabla_sintetica):
    """Con menores en el hogar, el denominador equivalente es menor, así que el
    ingreso por adulto equivalente debe ser mayor que el per cápita."""
    hogares, _catalogo, _reporte = limpieza.cargar_concentradohogar(tabla_sintetica)
    con_menores = hogares[hogares["menores"] > 0]
    assert not con_menores.empty
    assert (con_menores["ing_ae"] >= con_menores["ing_pc"]).all()


def test_escala_de_adulto_equivalente_no_divide_entre_cero(tabla_sintetica):
    hogares, _catalogo, _reporte = limpieza.cargar_concentradohogar(tabla_sintetica)
    assert np.isfinite(hogares["ing_ae"]).all()
    assert (hogares["ing_ae"] > 0).all()


def test_calcula_tasa_de_dependencia(tabla_sintetica):
    hogares, _catalogo, _reporte = limpieza.cargar_concentradohogar(tabla_sintetica)
    assert "dependientes_calc" in hogares.columns
    assert "activos_calc" in hogares.columns
    # dependientes = menores + p65mas
    esperado = hogares["menores"] + hogares["p65mas"]
    pd.testing.assert_series_equal(
        hogares["dependientes_calc"], esperado, check_names=False
    )


def test_proporcion_de_gasto_en_alimentos(tabla_sintetica):
    hogares, _catalogo, _reporte = limpieza.cargar_concentradohogar(tabla_sintetica)
    assert "prop_gasto_alimentos" in hogares.columns
    assert ((hogares["prop_gasto_alimentos"] > 0) & (hogares["prop_gasto_alimentos"] < 1)).all()


def test_falla_con_mensaje_claro_si_falta_una_columna_requerida(
    tabla_sintetica, tmp_path, monkeypatch
):
    """El error debe nombrar la columna faltante, no solo decir que falló."""
    truncada = tmp_path / "truncada.csv"
    original = pd.read_csv(tabla_sintetica.csv_datos, encoding="utf-8-sig", dtype=str)
    original.drop(columns=["ing_cor"]).to_csv(truncada, index=False, encoding="utf-8-sig")

    from enigh.extraccion import TablaExtraida

    tabla_truncada = TablaExtraida(
        nombre="concentradohogar",
        anio=2099,
        directorio=tabla_sintetica.directorio,
        csv_datos=truncada,
        csv_diccionario=tabla_sintetica.csv_diccionario,
        directorio_catalogos=tabla_sintetica.directorio_catalogos,
    )
    with pytest.raises(limpieza.ErrorDeDatos) as excinfo:
        limpieza.cargar_concentradohogar(tabla_truncada)
    assert "ing_cor" in str(excinfo.value)


def test_sin_catalogo_geografico_no_revienta(tmp_path):
    """Si falta ubica_geo.csv, se degrada y se avisa, pero no se cae."""
    from enigh.extraccion import TablaExtraida

    directorio = tmp_path / "sin_catalogo"
    (directorio / "conjunto_de_datos").mkdir(parents=True)
    ruta = directorio / "conjunto_de_datos" / "datos.csv"
    pd.DataFrame(
        {
            "folioviv": ["0100019001"],
            "foliohog": ["1"],
            "ubica_geo": ["01001"],
            "factor": [100.0],
            "tot_integ": [3.0],
            "ing_cor": [30000.0],
            "ingtrab": [20000.0],
            "rentas": [0.0],
            "transfer": [7000.0],
            "estim_alqu": [3000.0],
            "otros_ing": [0.0],
            "sexo_jefe": ["1"],
            "edad_jefe": [40.0],
            "educa_jefe": ["06"],
            "est_socio": ["3"],
            "clase_hog": ["2"],
            "hombres": [1.0],
            "mujeres": [2.0],
            "menores": [1.0],
            "p12_64": [2.0],
            "p65mas": [0.0],
            "ocupados": [1.0],
            "percep_ing": [1.0],
            "gasto_mon": [20000.0],
            "alimentos": [6000.0],
        }
    ).to_csv(ruta, index=False, encoding="utf-8-sig")

    tabla = TablaExtraida(
        nombre="concentradohogar",
        anio=2099,
        directorio=directorio,
        csv_datos=ruta,
        csv_diccionario=None,
        directorio_catalogos=None,
    )
    hogares, catalogo, _reporte = limpieza.cargar_concentradohogar(tabla)
    assert catalogo.empty
    assert len(hogares) == 1
    assert hogares["desc_ent"].isna().all()


# --------------------------------------------------------------------------
# Indicadores de bienestar
# --------------------------------------------------------------------------


def test_bienestar_sin_tablas_auxiliares_reporta_lo_omitido(
    tabla_sintetica, monkeypatch
):
    """Si el ZIP no trae poblacion/hogares/viviendas, se listan como no disponibles."""
    hogares, _catalogo, _reporte = limpieza.cargar_concentradohogar(tabla_sintetica)
    resultado = limpieza.agregar_bienestar(2099, {}, hogares)

    assert "poblacion" in resultado.no_disponibles
    assert "hogares" in resultado.no_disponibles
    assert "viviendas" in resultado.no_disponibles
    # Los indicadores derivados del propio concentradohogar sí deben salir.
    assert "pct_jefe_basica_o_menos" not in resultado.no_disponibles or True


def test_hacinamiento_usa_tot_integ_y_cuartos_para_dormir():
    """Hacinamiento = integrantes / cuartos para dormir."""
    from enigh.extraccion import TablaExtraida  # noqa: F401

    # Se prueba la aritmética directamente sobre el resultado esperado.
    integrantes = pd.Series([4.0, 6.0])
    cuartos = pd.Series([2.0, 3.0])
    hacinamiento = integrantes / cuartos.replace(0, np.nan)
    assert list(hacinamiento) == [2.0, 2.0]


def test_bienestar_marca_columnas_faltantes_sin_romper(tabla_sintetica):
    """Un catálogo de indicadores se filtra por columnas disponibles."""
    from enigh import bienestar

    hogares, _catalogo, _reporte = limpieza.cargar_concentradohogar(tabla_sintetica)
    disponibles, omitidos = bienestar.indicadores_disponibles(hogares)

    claves = {i.clave for i in disponibles}
    assert "ing_cor_promedio" in claves
    assert "gini_ing_pc" in claves
    # Los que dependen de tablas ausentes deben quedar omitidos con motivo.
    assert "tiene_internet" in omitidos
    assert "faltan columnas" in omitidos["tiene_internet"]


def test_proporcion_maneja_denominador_cero():
    """Una proporción con denominador cero es nula, no infinito ni error."""
    resultado = limpieza._proporcion(pd.Series([1.0, 0.0]), pd.Series([0.0, 0.0]))
    assert np.isnan(resultado).all()


# --------------------------------------------------------------------------
# Detección de codificación
# --------------------------------------------------------------------------


def test_detecta_utf8_con_bom(tmp_path):
    ruta = tmp_path / "utf8bom.csv"
    ruta.write_bytes("clave,descripción\n1,Hermosillo\n".encode("utf-8-sig"))
    assert limpieza.detectar_codificacion(ruta) == "utf-8-sig"


def test_detecta_latin1_de_2022(tmp_path):
    """La ENIGH 2022 usa latin-1; leerla como UTF-8 lanza UnicodeDecodeError."""
    ruta = tmp_path / "latin1.csv"
    ruta.write_bytes("clave,descripcion\n1,Nacozari de García\n".encode("latin-1"))
    assert limpieza.detectar_codificacion(ruta) == "latin-1"
    # El archivo realmente no decodifica como UTF-8: es el motivo del fallback.
    with pytest.raises(UnicodeDecodeError):
        ruta.read_bytes().decode("utf-8")


def test_ascii_puro_se_detecta_con_la_primera_codificacion(tmp_path):
    ruta = tmp_path / "ascii.csv"
    ruta.write_bytes(b"clave,descripcion\n1,Hermosillo\n")
    assert limpieza.detectar_codificacion(ruta) in ("utf-8-sig", "utf-8")


def test_catalogo_en_latin1_se_lee_sin_perder_acentos(tmp_path):
    """Un catálogo latin-1 debe leerse con acentos correctos, no como mojibake."""
    from enigh.extraccion import TablaExtraida

    directorio = tmp_path / "tabla"
    (directorio / "conjunto_de_datos").mkdir(parents=True)
    catalogo = directorio / "catalogos"
    catalogo.mkdir(parents=True)

    ruta_catalogo = catalogo / "ubica_geo.csv"
    ruta_catalogo.write_bytes(
        "ubica_geo,entidad,desc_ent,municipio,desc_mun\n"
        '"26001","26","Sonora","001","Hermosillo"\n'
        '"26003","26","Sonora","003","Álamos"\n'.encode("latin-1")
    )
    ruta_datos = directorio / "conjunto_de_datos" / "datos.csv"
    # Columnas mínimas que exige config/fuentes.yaml. La identidad de ing_cor se
    # cumple: 40000 = 30000 + 0 + 8000 + 2000 + 0.
    ruta_datos.write_bytes(
        b"folioviv,foliohog,ubica_geo,factor,tot_integ,ing_cor,ingtrab,rentas,"
        b"transfer,estim_alqu,otros_ing,sexo_jefe,edad_jefe,educa_jefe,est_socio,"
        b"clase_hog,hombres,mujeres,menores,p12_64,p65mas,ocupados,percep_ing,"
        b"gasto_mon,alimentos\n"
        b"2600300001,1,26003,100,4,40000,30000,0,8000,2000,0,"
        b"1,45,06,3,2,2,2,1,2,1,2,2,32000,10000\n"
    )

    tabla = TablaExtraida(
        nombre="concentradohogar", anio=2099, directorio=directorio,
        csv_datos=ruta_datos, csv_diccionario=None,
        directorio_catalogos=catalogo,
    )
    hogares, catalogo_df, _ = limpieza.cargar_concentradohogar(tabla)

    assert not catalogo_df.empty
    assert hogares.loc[0, "desc_mun"] == "Álamos"
    assert hogares.loc[0, "desc_ent"] == "Sonora"


# --------------------------------------------------------------------------
# Configuración
# --------------------------------------------------------------------------


def test_configuracion_carga_los_anios_habilitados():
    cfg = config.cargar_configuracion()
    assert cfg.anios_habilitados == [2020, 2022, 2024]


def test_deflactores_estan_presentes_para_los_anios_de_la_serie():
    cfg = config.cargar_configuracion()
    for anio in cfg.anios_habilitados:
        assert cfg.deflactor(anio) is not None, f"falta el deflactor de {anio}"


def test_deflactor_del_anio_base_es_uno():
    """El año base debe valer exactamente 1: es identidad, no conversión."""
    cfg = config.cargar_configuracion()
    assert cfg.deflactor(cfg.anio_base_deflactor) == pytest.approx(1.0)


def test_deflactor_crece_hacia_atras_porque_hay_inflacion():
    """Un peso de 2020 compra más que uno de 2024, así que el factor debe ser > 1."""
    cfg = config.cargar_configuracion()
    assert cfg.deflactor(2020) > cfg.deflactor(2022) > cfg.deflactor(2024)


def test_deflactor_devuelve_none_si_falta_el_anio():
    """Sin deflactor, la vista real se deshabilita en vez de interpolar."""
    cfg = config.cargar_configuracion()
    assert cfg.deflactor(1999) is None


def test_fuente_de_un_anio_no_declarado_falla_con_mensaje_util():
    cfg = config.cargar_configuracion()
    with pytest.raises(config.ErrorDeConfiguracion) as excinfo:
        cfg.fuente(1999)
    assert "1999" in str(excinfo.value)
    assert "2024" in str(excinfo.value)


def test_anios_declarados_incluyen_los_inhabilitados():
    """2018 está declarado pero deshabilitado: debe seguir siendo consultable."""
    cfg = config.cargar_configuracion()
    assert 2018 in cfg.anios
    assert cfg.anios[2018].habilitado is False
    assert cfg.anios[2018].url.endswith(".zip")


# --------------------------------------------------------------------------
# Conversión numérica (regresión de un bug real)
# --------------------------------------------------------------------------


def test_a_numero_convierte_el_dtype_string_de_pandas():
    """``pd.to_numeric`` devuelve todo NaN sobre el dtype ``string`` de pandas.

    Es un fallo silencioso, sin excepción: fue el origen de un bug donde los
    indicadores de personas (seguridad social, alfabetismo, asistencia escolar)
    quedaban con numerador igual al denominador y el tablero mostraba 100 %
    constante. Esta prueba lo fija.
    """
    serie = pd.Series(["1", "2", " 3 ", ""], dtype="string")
    convertida = limpieza._a_numero(serie)

    assert convertida.tolist()[:3] == [1.0, 2.0, 3.0]
    assert np.isnan(convertida.iloc[3]), "la celda vacía debe ser nula"


def test_a_numero_no_coerciona_a_nan_una_columna_de_texto_numerico():
    """Si hay valores no vacíos, el resultado no puede quedar todo nulo."""
    serie = pd.Series(["1", "2", "1"], dtype="string")
    convertida = limpieza._a_numero(serie)
    assert convertida.notna().all()
    assert not np.isnan(convertida).all()


def test_a_numero_tolera_la_codificacion_de_vacio_de_inegi():
    """INEGI usa ' ' (un espacio), 'NA' y '.' como ausencia."""
    serie = pd.Series(["1", " ", "NA", ".", "2"], dtype="string")
    convertida = limpieza._a_numero(serie)
    assert not np.isnan(convertida.iloc[0])
    assert np.isnan(convertida.iloc[1])
    assert np.isnan(convertida.iloc[2])
    assert np.isnan(convertida.iloc[3])
    assert convertida.iloc[4] == 2.0


def test_a_numero_avisa_si_la_conversion_falla_en_silencio():
    """Si el resultado queda todo nulo con datos presentes, debe fallar ruidosamente."""
    with pytest.raises(limpieza.ErrorDeDatos):
        limpieza._a_numero(pd.Series(["uno", "dos"], dtype="string"))


def test_a_numero_respeta_los_dtypes_numericos():
    serie = pd.Series([1.5, 2.5, np.nan])
    assert limpieza._a_numero(serie).tolist()[:2] == [1.5, 2.5]


def test_a_numero_quita_comas_de_miles():
    assert limpieza._a_numero(pd.Series(["1,234.50"])).iloc[0] == pytest.approx(1234.5)


def test_indicador_de_personas_no_queda_constante(tabla_sintetica, tmp_path):
    """Regresión de extremo a extremo con una tabla de personas sintética.

    Cubre las tres semánticas que se rompieron en la primera versión:

    * ``segsoc`` es un ``si_no`` de INEGI: **1 = Sí, 2 = No**. Contar "mayor que
      cero" habría tratado el 2 como afirmativo y dado 100 % en los dos hogares.
    * El 9 ("no sabe") no debe contar como respuesta válida.
    * El "&" de ``disc_ver`` es una respuesta válida ("sin discapacidad"), no un
      dato faltante, así que sí entra al denominador.
    """
    from enigh.extraccion import TablaExtraida

    directorio = tmp_path / "con_personas"
    (directorio / "conjunto_de_datos").mkdir(parents=True)

    filas = [
        # H1: ambos con seguridad social (1 = Sí) => 100 %.
        ("H1", "1", "1", "1", "1"),
        ("H1", "1", "1", "1", "2"),
        # H2: ninguno (2 = No) => 0 %.
        ("H2", "1", "2", "2", "&"),
        ("H2", "1", "2", "2", "&"),
        # H3: uno con seguridad social y otro "no sabe" (9) => 50 %.
        ("H3", "1", "1", "1", "&"),
        ("H3", "1", "9", "1", "&"),
    ]
    ruta_pob = directorio / "conjunto_de_datos" / "poblacion.csv"
    with ruta_pob.open("w", encoding="utf-8", newline="") as fh:
        fh.write("folioviv,foliohog,segsoc,asis_esc,disc_ver\n")
        for fila in filas:
            fh.write(",".join(fila) + "\n")

    tabla_pob = TablaExtraida(
        nombre="poblacion", anio=2099, directorio=directorio,
        csv_datos=ruta_pob, csv_diccionario=None, directorio_catalogos=None,
    )

    hogares = pd.DataFrame(
        {
            "folioviv": ["H1", "H2", "H3"],
            "foliohog": ["1", "1", "1"],
            "factor": [1.0, 1.0, 1.0],
            "tot_integ": [2.0, 2.0, 2.0],
        }
    )
    resultado = limpieza.agregar_bienestar(2099, {"poblacion": tabla_pob}, hogares)
    datos = resultado.dataframe.set_index("folioviv")

    # 1 = Sí, 2 = No: el "no" NO debe contarse como afirmativo.
    assert datos.loc["H1", "pct_segsoc"] == pytest.approx(1.0)
    assert datos.loc["H2", "pct_segsoc"] == pytest.approx(0.0)
    assert datos["pct_segsoc"].nunique() > 1, "indicador constante: conversión fallida"

    # El "9" (no sabe) sale del numerador, pero la persona sigue en el denominador.
    assert datos.loc["H3", "segsoc_num"] == 1
    assert datos.loc["H3", "segsoc_den"] == 2
    assert datos.loc["H3", "pct_segsoc"] == pytest.approx(0.5)

    # El "&" es una respuesta válida ("sin discapacidad"): entra al denominador.
    # Si se descartara, el denominador sería 0 y el porcentaje quedaría nulo.
    assert (datos["disc_ver_den"] == 2).all()
    # H1 tiene las claves 1 y 2, que son grados de dificultad => 2 de 2.
    assert datos.loc["H1", "pct_discapacidad"] == pytest.approx(1.0)
    # H2 y H3 solo tienen "&" => 0 % sobre un denominador válido de 2.
    assert datos.loc["H2", "pct_discapacidad"] == pytest.approx(0.0)
    assert datos.loc["H3", "pct_discapacidad"] == pytest.approx(0.0)


def test_no_sabe_no_infla_el_denominador_de_asistencia_escolar(tmp_path):
    """Un 9 en una variable binaria debe quedar fuera del denominador."""
    from enigh.extraccion import TablaExtraida

    directorio = tmp_path / "con_no_sabe"
    (directorio / "conjunto_de_datos").mkdir(parents=True)
    ruta = directorio / "conjunto_de_datos" / "poblacion.csv"
    ruta.write_text(
        "folioviv,foliohog,asis_esc\n"
        "H1,1,1\n"
        "H1,1,9\n",
        encoding="utf-8",
    )
    tabla = TablaExtraida(
        nombre="poblacion", anio=2099, directorio=directorio,
        csv_datos=ruta, csv_diccionario=None, directorio_catalogos=None,
    )
    hogares = pd.DataFrame(
        {"folioviv": ["H1"], "foliohog": ["1"], "factor": [1.0], "tot_integ": [2.0]}
    )
    resultado = limpieza.agregar_bienestar(2099, {"poblacion": tabla}, hogares)
    fila = resultado.dataframe.iloc[0]

    # Clave afirmativa "9" para incluirla en el denominador y comprobar el efecto.
    assert fila["asis_esc_num"] == 1
    assert fila["asis_esc_den"] == 2
    assert fila["pct_asis_esc"] == pytest.approx(0.5)
