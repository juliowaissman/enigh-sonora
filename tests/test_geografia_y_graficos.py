"""Pruebas del etiquetado territorial y de la degradación de los mapas.

El punto crítico es que el empate de municipios se hace por **clave cvegeo**
(``ubica_geo``), no por nombre: 22 de los 72 municipios de Sonora llevan acento y
las fuentes geográficas no son consistentes entre sí ("Álamos" vs "Alamos").
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from enigh import geografia, graficos


# --------------------------------------------------------------------------
# Normalización de nombres
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("entrada", "esperado"),
    [
        ("Sonora", "SONORA"),
        ("México", "MEXICO"),
        ("San José de Gracia", "SAN JOSE DE GRACIA"),
        ("Álamos", "ALAMOS"),
        ("Nácori Chico", "NACORI CHICO"),
        ("  Nuevo   León  ", "NUEVO LEON"),
        ("Yécora", "YECORA"),
        ("Ónavas", "ONAVAS"),
    ],
)
def test_normalizar_nombre_quita_acentos_y_colapsa_espacios(entrada, esperado):
    assert geografia.normalizar_nombre(entrada) == esperado


def test_normalizar_nombre_tolera_nulos():
    assert geografia.normalizar_nombre(None) == ""
    assert geografia.normalizar_nombre(np.nan) == ""


def test_normalizar_nombre_elimina_puntuacion():
    assert geografia.normalizar_nombre("San Felipe de Jesús (Norte)") == (
        "SAN FELIPE DE JESUS NORTE"
    )


# --------------------------------------------------------------------------
# Extracción de claves
# --------------------------------------------------------------------------


def test_clave_cvegeo_desde_propiedades_de_inegi():
    props = {"cvegeo": "26023", "cve_agee": "26", "nom_agem": "Cumpas"}
    assert geografia._clave_cvegeo(props) == "26023"


def test_clave_cvegeo_rellena_con_ceros():
    """Una clave de 1 dígito debe quedar como 00001, no como 1."""
    assert geografia._clave_cvegeo({"cvegeo": "1"}) == "00001"


def test_clave_cvegeo_quita_el_prefijo_mx():
    """El estándar OCHA usa 'MX' + cvegeo."""
    assert geografia._clave_cvegeo({"adm2_pcode": "MX26001"}) == "26001"


def test_clave_cvegeo_acepta_varios_nombres_de_campo():
    assert geografia._clave_cvegeo({"CVEGEO": "01001"}) == "01001"
    assert geografia._clave_cvegeo({"concat": "26072"}) == "26072"


def test_clave_cvegeo_vacia_si_no_hay_clave():
    assert geografia._clave_cvegeo({"name": "Hermosillo"}) == ""


def test_nombre_municipio_acepta_varios_campos():
    assert geografia._nombre_municipio({"nom_agem": "Cumpas"}) == "Cumpas"
    assert geografia._nombre_municipio({"NOM_MUN": "Cumpas"}) == "Cumpas"
    assert geografia._nombre_municipio({"shapeName": "Cumpas"}) == "Cumpas"
    assert geografia._nombre_municipio({}) == ""


# --------------------------------------------------------------------------
# Índice municipal y empate con los microdatos
# --------------------------------------------------------------------------


def _geojson_falso() -> dict:
    """GeoJSON mínimo con dos municipios de Sonora, uno con acento."""
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "cvegeo": "26001", "cve_agee": "26", "nom_agem": "Hermosillo",
                },
                "geometry": {"type": "Point", "coordinates": [-110.9, 29.0]},
            },
            {
                "type": "Feature",
                "properties": {
                    "cvegeo": "26003", "cve_agee": "26", "nom_agem": "Álamos",
                },
                "geometry": {"type": "Point", "coordinates": [-108.9, 27.0]},
            },
        ],
    }


def test_escribe_indice_municipal_desde_el_geojson(tmp_path, monkeypatch):
    monkeypatch.setattr(geografia, "NOMBRE_INDICE_MUNICIPIOS", tmp_path / "indice.csv")
    geografia._escribir_indice_municipios(_geojson_falso())

    indice = pd.read_csv(tmp_path / "indice.csv", dtype=str)
    assert list(indice["cvegeo"]) == ["26001", "26003"]
    assert "Álamos" in set(indice["nom_mun"])
    # La versión normalizada debe servir para un empate por nombre de respaldo.
    assert "ALAMOS" in set(indice["nom_mun_norm"])


def test_agregar_municipio_empata_por_clave_no_por_nombre(tmp_path, monkeypatch):
    """Aunque el nombre del catálogo de INEGI venga sin acento, debe empatar."""
    monkeypatch.setattr(geografia, "NOMBRE_INDICE_MUNICIPIOS", tmp_path / "indice.csv")
    geografia._escribir_indice_municipios(_geojson_falso())

    hogares = pd.DataFrame(
        {
            "ubica_geo": ["26001", "26003", "26099"],
            "municipio": ["001", "003", "099"],
            # El catálogo de INEGI escribe sin acento; no debe importar.
            "desc_mun": ["Hermosillo", "Alamos", "Inexistente"],
        }
    )
    resultado = geografia.agregar_municipio(hogares)

    assert resultado.loc[0, "nom_municipio"] == "Hermosillo"
    assert resultado.loc[1, "nom_municipio"] == "Álamos", (
        "el empate por clave debe traer el nombre acentuado del GeoJSON"
    )
    # Un municipio que no está en el índice debe conservar una etiqueta legible.
    assert "099" in resultado.loc[2, "nom_municipio"]
    assert resultado["nom_municipio"].notna().all()


def test_agregar_municipio_sin_indice_usa_el_catalogo_de_inegi(tmp_path, monkeypatch):
    monkeypatch.setattr(geografia, "NOMBRE_INDICE_MUNICIPIOS", tmp_path / "no_existe.csv")
    hogares = pd.DataFrame(
        {"ubica_geo": ["26001"], "municipio": ["001"], "desc_mun": ["Hermosillo"]}
    )
    resultado = geografia.agregar_municipio(hogares)
    assert resultado.loc[0, "nom_municipio"] == "Hermosillo"


def test_agregar_municipio_sin_nada_no_revienta(tmp_path, monkeypatch):
    monkeypatch.setattr(geografia, "NOMBRE_INDICE_MUNICIPIOS", tmp_path / "no_existe.csv")
    hogares = pd.DataFrame({"ubica_geo": ["26001"]})
    resultado = geografia.agregar_municipio(hogares)
    assert "nom_municipio" in resultado.columns
    assert resultado["nom_municipio"].notna().all()


# --------------------------------------------------------------------------
# Lectura de GeoJSON
# --------------------------------------------------------------------------


def test_cargar_geojson_inexistente_devuelve_none(tmp_path):
    assert geografia.cargar_geojson(tmp_path / "no_existe.geojson") is None
    assert geografia.cargar_geojson(None) is None


def test_cargar_geojson_lee_el_archivo(tmp_path):
    ruta = tmp_path / "capa.geojson"
    ruta.write_text(json.dumps(_geojson_falso()), encoding="utf-8")
    datos = geografia.cargar_geojson(ruta)
    assert datos is not None
    assert len(datos["features"]) == 2


def test_sha256_de_un_archivo_conocido(tmp_path):
    ruta = tmp_path / "x.bin"
    ruta.write_bytes(b"abc")
    # SHA-256 de "abc" es un valor publicado conocido.
    assert geografia._sha256(ruta) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


# --------------------------------------------------------------------------
# Gráficas: degradación y supresión
# --------------------------------------------------------------------------


def test_figura_vacia_no_revienta_y_trae_un_aviso():
    figura = graficos._figura_vacia("sin datos")
    assert figura.layout.annotations
    assert figura.layout.annotations[0].text == "sin datos"


def test_mapa_entidades_sin_geojson_avisa_como_obtenerlo():
    entidades = pd.DataFrame({"entidad": ["26"], "nom_entidad": ["Sonora"], "ing_pc": [1.0]})
    figura = graficos.mapa_entidades(entidades, {})
    assert figura.layout.annotations
    assert "geografia" in figura.layout.annotations[0].text


def test_mapa_municipios_suprime_lo_poco_fiable():
    """Ningún municipio fiable => se avisa en lugar de pintar promedios frágiles."""
    municipal = pd.DataFrame(
        {
            "nom_municipio": ["Chico", "Mediano"],
            "n_hogares": [5, 12],
            "ing_pc": [5000.0, 7000.0],
        }
    )
    figura = graficos.mapa_municipios(
        municipal, _geojson_falso(), hogares_minimos=30
    )
    assert figura.layout.annotations
    assert "30 hogares" in figura.layout.annotations[0].text


def test_mapa_municipios_sin_geometria_ni_coordenadas_avisa():
    municipal = pd.DataFrame(
        {"nom_municipio": ["Hermosillo"], "n_hogares": [100], "ing_pc": [9000.0]}
    )
    figura = graficos.mapa_municipios(municipal, None, hogares_minimos=30)
    assert figura.layout.annotations
    assert "tabla municipal" in figura.layout.annotations[0].text


def test_barras_deciles_incluye_bigotes_de_intervalo():
    tabla = pd.DataFrame(
        {
            "decil": list(range(1, 11)),
            "ing_pc": np.linspace(1000, 10000, 10),
            "ing_pc_ic_inferior": np.linspace(900, 9900, 10),
            "ing_pc_ic_superior": np.linspace(1100, 10100, 10),
        }
    )
    figura = graficos.barras_deciles(tabla, columna="ing_pc")
    assert figura.data[0].error_y is not None
    assert len(figura.data[0].error_y.array) == 10


def test_barras_deciles_sin_columna_no_revienta():
    figura = graficos.barras_deciles(pd.DataFrame({"decil": [1]}), columna="ing_pc")
    assert figura.layout.annotations


def test_heatmap_ancho_de_filas_igual_al_numero_de_indicadores():
    perfil = pd.DataFrame(
        {
            "grupo": [1, 1, 2, 2],
            "etiqueta": ["Internet", "Agua", "Internet", "Agua"],
            "indicador": ["internet", "agua", "internet", "agua"],
            "valor": [0.9, 0.8, 0.4, 0.5],
        }
    )
    figura = graficos.heatmap_indicadores(perfil)
    assert len(figura.data[0].y) == 2
    assert len(figura.data[0].x) == 2


def test_linea_serie_usa_eje_categorico_para_no_interpolar():
    """Los años de la ENIGH no son continuos: el eje debe ser categórico."""
    serie = pd.DataFrame(
        {
            "anio": [2020, 2020, 2022, 2022, 2024, 2024],
            "decil": [1, 2, 1, 2, 1, 2],
            "ing_pc": [1000.0, 2000.0, 1200.0, 2400.0, 1300.0, 2600.0],
            "ing_pc_ic_inferior": [900.0] * 6,
            "ing_pc_ic_superior": [1100.0] * 6,
        }
    )
    figura = graficos.linea_serie_deciles(serie)
    assert figura.layout.xaxis.type == "category"
    assert len(figura.data) == 2  # dos deciles


def test_cambio_porcentual_requiere_dos_anios():
    serie = pd.DataFrame(
        {"anio": [2024, 2024], "decil": [1, 2], "ing_pc": [1000.0, 2000.0]}
    )
    figura = graficos.cambio_porcentual_por_decil(serie)
    assert figura.layout.annotations
    assert "dos levantamientos" in figura.layout.annotations[0].text


def test_cambio_porcentual_calcula_bien():
    serie = pd.DataFrame(
        {
            "anio": [2020, 2020, 2024, 2024],
            "decil": [1, 2, 1, 2],
            "ing_pc": [100.0, 200.0, 150.0, 300.0],
        }
    )
    figura = graficos.cambio_porcentual_por_decil(serie)
    valores = list(figura.data[0].y)
    assert valores == pytest.approx([50.0, 50.0])


def test_curva_lorenz_tiene_dos_trazas_y_termina_en_uno():
    valores = pd.Series([1.0, 2.0, 3.0, 4.0])
    pesos = pd.Series([1.0, 1.0, 1.0, 1.0])
    figura = graficos.curva_lorenz(valores, pesos)
    assert len(figura.data) == 2
    assert figura.data[0].x[-1] == pytest.approx(1.0)
    assert figura.data[0].y[-1] == pytest.approx(1.0)


def test_curva_lorenz_con_ingreso_cero_no_revienta():
    figura = graficos.curva_lorenz(pd.Series([0.0, 0.0]), pd.Series([1.0, 1.0]))
    assert figura.layout.annotations, "debe avisar en lugar de dividir entre cero"


def test_barras_composicion_sin_fuentes_avisa():
    figura = graficos.barras_composicion(pd.DataFrame({"decil": [1], "ing_pc": [1.0]}))
    assert figura.layout.annotations


# --------------------------------------------------------------------------
# Clave cvegeo para el coroplético municipal
# --------------------------------------------------------------------------


def test_poner_clave_cvegeo_usa_la_columna_existente():
    """Si el pipeline ya dejó ``clave_cvegeo``, se respeta y se normaliza."""
    datos = pd.DataFrame({"clave_cvegeo": ["26030"], "ing_pc": [9000.0]})
    assert geografia.poner_clave_cvegeo(datos).loc[0, "clave_cvegeo"] == "26030"


def test_poner_clave_cvegeo_rellena_con_ceros():
    datos = pd.DataFrame({"clave_cvegeo": ["26001"], "ing_pc": [1.0]})
    resultado = geografia.poner_clave_cvegeo(datos)
    assert resultado.loc[0, "clave_cvegeo"] == "26001"
    assert resultado.loc[0, "clave_cvegeo"].startswith("260")


def test_poner_clave_cvegeo_deriva_de_ubica_geo():
    datos = pd.DataFrame({"ubica_geo": ["26003"], "ing_pc": [1.0]})
    assert geografia.poner_clave_cvegeo(datos).loc[0, "clave_cvegeo"] == "26003"


def test_poner_clave_cvegeo_compone_entidad_y_municipio():
    """Un municipio clave '3' debe componerse como 26003, no como 263."""
    datos = pd.DataFrame({"entidad": ["26"], "municipio": ["3"], "ing_pc": [1.0]})
    assert geografia.poner_clave_cvegeo(datos).loc[0, "clave_cvegeo"] == "26003"


def test_poner_clave_cvegeo_sin_datos_suficientes_marca_na():
    datos = pd.DataFrame({"ing_pc": [1.0]})
    resultado = geografia.poner_clave_cvegeo(datos)
    assert "clave_cvegeo" in resultado.columns
    assert resultado["clave_cvegeo"].isna().all()


def test_mapa_municipios_empata_por_cvegeo_del_geojson_real():
    """El coroplético debe empatar por ``properties.cvegeo`` de la fuente usada.

    Se verifica contra las propiedades reales del GeoJSON del INEGI
    (``cvegeo``, no ``NOM_MUN`` ni ``name``), que es el error que tenía la
    primera versión del mapa.
    """
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "cvegeo": "26030", "cve_agee": "26", "nom_agem": "Hermosillo",
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[-111, 29], [-110, 29], [-110, 30], [-111, 30], [-111, 29]]],
                },
            },
            {
                "type": "Feature",
                "properties": {
                    "cvegeo": "26003", "cve_agee": "26", "nom_agem": "Álamos",
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[-109, 27], [-108, 27], [-108, 28], [-109, 28], [-109, 27]]],
                },
            },
        ],
    }
    municipal = pd.DataFrame(
        {
            "nom_municipio": ["Hermosillo", "Álamos"],
            "clave_cvegeo": ["26030", "26003"],
            "n_hogares": [120, 80],
            "ing_pc": [9500.0, 7200.0],
        }
    )
    figura = graficos.mapa_municipios(municipal, geojson, hogares_minimos=30)

    # Dos municipios dibujados y la llave usada es la columna cvegeo.
    assert figura.data[0].featureidkey == "properties.cvegeo"
    assert set(figura.data[0].locations) == {"26030", "26003"}
    assert len(figura.data[0].z) == 2


def test_mapa_municipios_no_dibuja_los_municipios_que_no_empatan():
    """Un municipio ausente del GeoJSON simplemente no se pinta (no inventa dato)."""
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"cvegeo": "26030", "nom_agem": "Hermosillo"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[-111, 29], [-110, 29], [-110, 30], [-111, 30], [-111, 29]]],
                },
            },
        ],
    }
    municipal = pd.DataFrame(
        {
            "nom_municipio": ["Hermosillo", "Fantasma"],
            "clave_cvegeo": ["26030", "26999"],
            "n_hogares": [120, 90],
            "ing_pc": [9500.0, 100.0],
        }
    )
    figura = graficos.mapa_municipios(municipal, geojson, hogares_minimos=30)
    assert set(figura.data[0].locations) == {"26030"}


# --------------------------------------------------------------------------
# Todos los constructores de gráficas deben poder construirse
# --------------------------------------------------------------------------


def test_todos_los_constructores_de_graficas_funcionan():
    """Ejercita cada figura con datos realistas.

    Plotly valida los nombres de las propiedades de forma perezosa: un
    ``rangomode`` en lugar de ``rangemode`` no falla hasta que se construye la
    figura, así que sin esta prueba el error aparecería solo al abrir esa vista
    del tablero. Aquí se recorren todas.
    """
    deciles = pd.DataFrame(
        {
            "decil": list(range(1, 11)),
            "n_hogares": [120] * 10,
            "ing_pc": np.linspace(5000, 90000, 10),
            "ing_pc_mediana": np.linspace(4800, 88000, 10),
            "ing_pc_ic_inferior": np.linspace(4700, 85000, 10),
            "ing_pc_ic_superior": np.linspace(5300, 95000, 10),
            "ing_pc_cv": [0.02] * 10,
            "ing_pc_fiable": [True] * 10,
            "ing_cor": np.linspace(20000, 300000, 10),
            "gini_global": [0.39] * 10,
            "gini_dentro_del_decil": np.linspace(0.12, 0.30, 10),
            "participacion_ingreso": [0.6, 1.4, 2.4, 3.8, 4.9, 6.6, 11.2, 14.3, 17.5, 37.2],
            "share_ingtrab": [0.5] * 10,
            "share_transfer": [0.3] * 10,
            "share_remesas": [0.1] * 10,
        }
    )
    entidades = pd.DataFrame(
        {
            "entidad": ["26"] + [f"{i:02d}" for i in range(1, 11)],
            "nom_entidad": ["Sonora"] + [f"Entidad {i}" for i in range(1, 11)],
            "ing_pc": np.linspace(5000, 25000, 11),
            "ing_cor": np.linspace(20000, 90000, 11),
            "gini": [0.39] * 11,
        }
    )
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "Sonora"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[-112, 27], [-109, 27], [-109, 32], [-112, 32], [-112, 27]]],
                },
            },
            {
                "type": "Feature",
                "properties": {"name": "Entidad 1"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[-100, 20], [-99, 20], [-99, 21], [-100, 21], [-100, 20]]],
                },
            },
        ],
    }
    municipal = pd.DataFrame(
        {
            "nom_municipio": ["Hermosillo", "Álamos"],
            "clave_cvegeo": ["26030", "26003"],
            "n_hogares": [120, 80],
            "ing_pc": [9500.0, 7200.0],
        }
    )
    serie = pd.DataFrame(
        {
            "anio": [2020, 2020, 2022, 2022, 2024, 2024],
            "decil": [1, 10, 1, 10, 1, 10],
            "ing_pc": [3332.0, 63096.0, 4703.0, 77937.0, 5908.0, 90130.0],
            "ing_pc_ic_inferior": [3200.0, 61000.0, 4600.0, 76000.0, 5700.0, 85000.0],
            "ing_pc_ic_superior": [3450.0, 65000.0, 4800.0, 80000.0, 6100.0, 95000.0],
        }
    )
    perfil = pd.DataFrame(
        {
            "grupo": [1, 1, 2, 2, 10, 10],
            "indicador": ["internet", "agua", "internet", "agua", "internet", "agua"],
            "etiqueta": ["Internet", "Agua", "Internet", "Agua", "Internet", "Agua"],
            "dominio": ["Conectividad", "Vivienda"] * 3,
            "unidad": ["proporción"] * 6,
            "valor": [0.35, 0.90, 0.50, 0.95, 0.95, 0.99],
        }
    )

    figuras = {
        "barras_deciles": lambda: graficos.barras_deciles(deciles, columna="ing_pc"),
        "barras_comparadas": lambda: graficos.barras_comparadas(
            {"Sonora": deciles, "Nacional": deciles}, columna="ing_pc", con_ic=True
        ),
        "linea_serie_deciles": lambda: graficos.linea_serie_deciles(serie),
        "cambio_porcentual": lambda: graficos.cambio_porcentual_por_decil(serie),
        "mapa_entidades": lambda: graficos.mapa_entidades(entidades, geojson),
        "puntos_entidades": lambda: graficos.puntos_entidades(entidades),
        "mapa_municipios": lambda: graficos.mapa_municipios(
            municipal, geojson, hogares_minimos=30
        ),
        "curva_lorenz": lambda: graficos.curva_lorenz(
            pd.Series(deciles["ing_pc"]), pd.Series([1.0] * 10)
        ),
        "heatmap_indicadores": lambda: graficos.heatmap_indicadores(perfil),
        "barras_composicion": lambda: graficos.barras_composicion(deciles),
    }

    for nombre, constructor in figuras.items():
        figura = constructor()
        assert figura is not None, f"{nombre} devolvió None"
        # Toda figura debe ser serializable a JSON, que es lo que hace Streamlit.
        figura.to_json()
        # Y tener contenido o un aviso, nunca quedar en blanco.
        assert figura.data or figura.layout.annotations, (
            f"{nombre} quedó vacía sin aviso"
        )


def test_ninguna_grafica_usa_propiedades_invalidas_de_plotly():
    """Las figuras deben poder reconstruirse desde su propio JSON."""
    tabla = pd.DataFrame(
        {
            "decil": [1, 2],
            "ing_pc": [1000.0, 2000.0],
            "ing_pc_ic_inferior": [900.0, 1900.0],
            "ing_pc_ic_superior": [1100.0, 2100.0],
        }
    )
    figura = graficos.barras_deciles(tabla, columna="ing_pc")
    reconstruida = go.Figure(figura.to_dict())
    assert len(reconstruida.data) == len(figura.data)


# --------------------------------------------------------------------------
# Alias de nombres de entidad (para que el mapa cubra las 32)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("oficial", "corto"),
    [
        ("Coahuila de Zaragoza", "COAHUILA"),
        ("Michoacán de Ocampo", "MICHOACAN"),
        ("Veracruz de Ignacio de la Llave", "VERACRUZ"),
        ("San Luis Potosí", "SAN LUIS POTOSI"),
    ],
)
def test_alias_entidad_genera_la_forma_corta_del_nombre_oficial(oficial, corto):
    """Los nombres oficiales largos del INEGI deben poder empatar con el GeoJSON corto."""
    assert corto in geografia.alias_entidad(oficial)
    assert geografia.normalizar_nombre(oficial) in geografia.alias_entidad(oficial)


def test_alias_entidad_no_genera_prefijos_genericos():
    """"Estado de México" no debe producir el alias "ESTADO".

    Ese alias podría empatar con cualquier entidad cuyo nombre empiece así y
    asignarle el valor de otra, que es un error de datos silencioso.
    """
    variantes = geografia.alias_entidad("Estado de México")
    assert "ESTADO" not in variantes
    assert "ESTADO DE MEXICO" in variantes


def test_alias_entidad_sin_particulas_devuelve_solo_el_nombre():
    assert geografia.alias_entidad("Sonora") == {"SONORA"}
    assert geografia.alias_entidad(None) == set()
    assert geografia.alias_entidad("") == set()


def test_alias_entidad_de_una_sola_palabra_con_particula_no_recorta():
    """Un nombre que empieza con partícula no debe producir un alias vacío."""
    variantes = geografia.alias_entidad("La Paz")
    assert variantes
    assert all(v for v in variantes)


def test_mapa_entidades_cubre_las_32_entidades_con_los_alias():
    """Regresión: sin alias, el mapa dibujaba 29 de 32 entidades."""
    oficiales = [
        "Aguascalientes", "Baja California", "Baja California Sur", "Campeche",
        "Coahuila de Zaragoza", "Colima", "Chiapas", "Chihuahua",
        "Ciudad de México", "Durango", "Guanajuato", "Guerrero", "Hidalgo",
        "Jalisco", "México", "Michoacán de Ocampo", "Morelos", "Nayarit",
        "Nuevo León", "Oaxaca", "Puebla", "Querétaro", "Quintana Roo",
        "San Luis Potosí", "Sinaloa", "Sonora", "Tabasco", "Tamaulipas",
        "Tlaxcala", "Veracruz de Ignacio de la Llave", "Yucatán", "Zacatecas",
    ]
    # La capa geográfica usa los nombres cortos, como la fuente real.
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": geografia.alias_entidad(nombre)
                               and sorted(geografia.alias_entidad(nombre))[0]},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[-100, 20], [-99, 20], [-99, 21], [-100, 21], [-100, 20]]],
                },
            }
            for nombre in oficiales
        ],
    }
    entidades = pd.DataFrame(
        {
            "entidad": [f"{i:02d}" for i in range(1, 33)],
            "nom_entidad": oficiales,
            "ing_pc": np.linspace(5000, 30000, 32),
        }
    )
    figura = graficos.mapa_entidades(entidades, geojson, columna="ing_pc")
    assert len(figura.data[0].z) == 32, (
        f"el mapa dibuja {len(figura.data[0].z)} de 32 entidades; "
        "los alias de nombre están fallando"
    )
