"""Fixtures compartidos por las pruebas.

Las pruebas son deliberadamente **independientes de la red y de los ZIP de
INEGI**: usan un conjunto sintético pequeño con el mismo esquema de columnas que
los microdatos reales. Así `pytest` corre en cualquier máquina sin descargar
287 MB, y las pruebas de exactitud numérica se pueden verificar a mano.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from enigh import CLAVE_SONORA  # noqa: E402


@pytest.fixture(scope="session")
def raiz_proyecto() -> Path:
    return RAIZ


def _escala_esperada(menores, p12_64, p65mas) -> float:
    """Misma escala que usa el proyecto, para poder verificarla en las pruebas."""
    return 0.7 * menores + 0.8 * p12_64 + 1.0 * p65mas


@pytest.fixture
def hogares_sinteticos() -> pd.DataFrame:
    """Hogares sintéticos con diseño muestral completo y datos verificables.

    Se construyen 40 hogares: 20 en Sonora (entidad 26) y 20 en otra entidad
    (entidad 01), con estratos y UPM repetidos para poder probar el bootstrap.
    El ingreso per cápita se asigna de forma controlada para que los deciles se
    puedan comprobar a mano.
    """
    filas = []
    for i in range(40):
        en_sonora = i < 20
        entidad = CLAVE_SONORA if en_sonora else "01"
        municipio = f"{(i % 5) + 1:03d}"
        integrantes = 2 + (i % 4)
        ingreso_pc = 1000.0 * (i + 1)
        # Los bloques de edad deben ser consistentes con tot_integ: en la ENIGH
        # real menores + p12_64 + p65mas == tot_integ. Se reparte cada integrante
        # exactamente una vez para que el fixture sea plausible.
        menores = 1
        p12_64 = integrantes - 2
        p65mas = integrantes - menores - p12_64  # exactamente 1
        filas.append(
            {
                "folioviv": f"{entidad}{municipio}{i:04d}",
                "foliohog": "1",
                "ubica_geo": f"{entidad}{municipio}",
                "entidad": entidad,
                "municipio": municipio,
                "est_dis": f"{(i // 4) + 1:03d}",
                "upm": f"{(i // 2) + 1:07d}",
                "factor": 100.0 + i,
                "tam_loc": "1",
                "est_socio": "3",
                "clase_hog": "2",
                "sexo_jefe": "1" if i % 3 else "2",
                "edad_jefe": 30 + (i % 40),
                "educa_jefe": "04" if i % 2 else "09",
                "tot_integ": integrantes,
                "hombres": integrantes // 2,
                "mujeres": integrantes - integrantes // 2,
                "menores": menores,
                "mayores": p65mas,
                "p12_64": p12_64,
                "p65mas": p65mas,
                "ocupados": 1 + (i % 2),
                "percep_ing": 1 + (i % 2),
                "perc_ocupa": 1,
                "ing_cor": ingreso_pc * integrantes,
                "ingtrab": ingreso_pc * integrantes * 0.7,
                "negocio": 0.0,
                "rentas": 0.0,
                "transfer": ingreso_pc * integrantes * 0.2,
                "estim_alqu": ingreso_pc * integrantes * 0.1,
                "otros_ing": 0.0,
                "remesas": ingreso_pc * integrantes * 0.05,
                "bene_gob": ingreso_pc * integrantes * 0.05,
                "jubilacion": 0.0,
                "gasto_mon": ingreso_pc * integrantes * 0.8,
                "alimentos": ingreso_pc * integrantes * 0.3,
                "ing_pc": ingreso_pc,
                "ing_ae": ingreso_pc * integrantes
                / _escala_esperada(menores, p12_64, p65mas),
                "share_ingtrab": 0.7,
                "share_transfer": 0.2,
                "share_remesas": 0.05,
                "share_bene_gob": 0.05,
                "share_jubilacion": 0.0,
                "share_rentas": 0.0,
                "share_negocio": 0.0,
                "share_estim_alqu": 0.1,
                "prop_gasto_alimentos": 0.375,
                "dependientes_calc": 2.0,
                "activos_calc": float(integrantes - 2),
                "smg": 22403.7,
                "desc_ent": "Sonora" if en_sonora else "Aguascalientes",
                "desc_mun": f"Municipio {municipio}",
            }
        )
    return pd.DataFrame(filas)


@pytest.fixture
def microdatos_para_limpieza(tmp_path: Path) -> Path:
    """Escribe un ``concentradohogar`` sintético en disco con su diccionario.

    Reproduce los detalles reales del archivo de INEGI que rompen un lector
    ingenuo: BOM al inicio, comillas en la descripción y ceros a la izquierda en
    las claves geográficas.
    """
    from enigh.extraccion import TablaExtraida

    directorio = tmp_path / "conjunto_de_datos_concentradohogar_enigh2099_ns"
    (directorio / "conjunto_de_datos").mkdir(parents=True)
    (directorio / "catalogos").mkdir(parents=True)

    columnas = [
        "folioviv", "foliohog", "ubica_geo", "tam_loc", "est_socio", "est_dis",
        "upm", "factor", "clase_hog", "sexo_jefe", "edad_jefe", "educa_jefe",
        "tot_integ", "hombres", "mujeres", "mayores", "menores", "p12_64",
        "p65mas", "ocupados", "percep_ing", "perc_ocupa", "ing_cor", "ingtrab",
        "rentas", "transfer", "estim_alqu", "otros_ing", "remesas", "bene_gob",
        "jubilacion", "gasto_mon", "alimentos", "smg",
    ]

    filas = [
        # Sonora, cuadra la identidad de ing_cor perfectamente.
        ["0100019001", "1", "01001", "1", "3", "001", "0000001", "207", "2", "1",
         "32", "06", "4", "2", "2", "0", "1", "3", "1", "2", "2", "2",
         "138232.38", "130518.10", "0", "7714.28", "0", "0", "0", "0", "0",
         "47478.66", "17858.49", "22403.7"],
        # Sonora, con BOM y comillas en los textos (INEGI los usa).
        ["2600100010", "1", "26001", "1", "3", "002", "0000002", "150", "2", "2",
         "48", "09", "5", "2", "3", "1", "1", "4", "1", "1", "1", "1",
         "50000.00", "40000.00", "0", "5000.00", "5000.00", "0", "2500.00",
         "0", "0", "40000.00", "12000.00", "22403.7"],
        # Otra entidad.
        ["0100200001", "1", "01002", "1", "2", "001", "0000003", "120", "1", "2",
         "55", "03", "3", "1", "2", "0", "2", "1", "0", "1", "1", "1",
         "30000.00", "20000.00", "1000.00", "5000.00", "2000.00", "2000.00",
         "0", "0", "0", "24000.00", "9000.00", "22403.7"],
        # Fila inválida a propósito: factor <= 0 (debe descartarse).
        ["2600100011", "1", "26001", "1", "3", "002", "0000002", "0", "2", "1",
         "30", "05", "3", "2", "1", "0", "1", "1", "0", "1", "1", "1",
         "10000.00", "8000.00", "0", "1000.00", "1000.00", "0", "0", "0", "0",
         "8000.00", "3000.00", "22403.7"],
    ]

    ruta_datos = directorio / "conjunto_de_datos" / "conjunto_de_datos_concentradohogar_enigh2099_ns.csv"
    with ruta_datos.open("w", encoding="utf-8-sig", newline="") as fh:
        fh.write(",".join(columnas) + "\n")
        for fila in filas:
            fh.write(",".join(str(v) for v in fila) + "\n")

    ruta_dicc = directorio / "diccionario_de_datos" / "diccionario_datos_concentradohogar_enigh2099_ns.csv"
    ruta_dicc.parent.mkdir(parents=True)
    with ruta_dicc.open("w", encoding="utf-8-sig", newline="") as fh:
        fh.write("nombre_campo,longitud,tipo,nemónico,catálogo,rango_claves,no_especificado\n")
        descripciones = {
            "folioviv": ("Identificador de la vivienda", "10", "C"),
            "foliohog": ("Identificador del hogar", "1", "C"),
            "ubica_geo": ("Ubicación geográfica", "5", "C"),
            "factor": ("Factor de expansión", "5", "N"),
            "tot_integ": ("Número de integrantes del hogar", "2", "N"),
            "ing_cor": ('Ingreso corriente "total"', "12,2", "N"),
            "ingtrab": ("Ingreso por trabajo", "12,2", "N"),
            "rentas": ("Renta de la propiedad", "12,2", "N"),
            "transfer": ("Transferencias", "12,2", "N"),
            "estim_alqu": ("Estimación del alquiler", "12,2", "N"),
            "otros_ing": ("Otros ingresos corrientes", "12,2", "N"),
        }
        for clave, (descripcion, longitud, tipo) in descripciones.items():
            fh.write(f'"{descripcion}",{longitud},{tipo},{clave},,,\n')

    ruta_catalogo = directorio / "catalogos" / "ubica_geo.csv"
    with ruta_catalogo.open("w", encoding="utf-8-sig", newline="") as fh:
        fh.write("ubica_geo,entidad,desc_ent,municipio,desc_mun\n")
        fh.write('"01001","01","Aguascalientes","001","Aguascalientes"\n')
        fh.write('"01002","01","Aguascalientes","002","Asientos"\n')
        fh.write('"26001","26","Sonora","001","Hermosillo"\n')

    return TablaExtraida(
        nombre="concentradohogar",
        anio=2099,
        directorio=directorio,
        csv_datos=ruta_datos,
        csv_diccionario=ruta_dicc,
        directorio_catalogos=directorio / "catalogos",
    )


@pytest.fixture
def sonora_sintetica(hogares_sinteticos: pd.DataFrame) -> pd.DataFrame:
    """Solo los hogares de Sonora del conjunto sintético."""
    datos = hogares_sinteticos
    return datos[datos["entidad"] == CLAVE_SONORA].copy()


@pytest.fixture
def rng() -> np.random.Generator:
    """Generador reproducible para las pruebas."""
    return np.random.default_rng(20240101)
