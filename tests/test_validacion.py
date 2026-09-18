"""Pruebas del módulo de validación.

Se verifica la lógica de comparación contra cifras de control y de detección de
invariantes violadas. Los datos son sintéticos: la validación real corre sobre
los Parquet procesados con ``make validate``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from enigh import validacion


def test_comparacion_dentro_de_tolerancia_pasa():
    reporte = validacion.ReporteValidacion(anio=2024)
    validacion._comparar_con_control(reporte, "x", 100.0, 100.0, 0.01)
    assert reporte.paso
    assert reporte.comprobaciones[0].simbolo == "OK  "


def test_comparacion_fuera_de_tolerancia_falla():
    reporte = validacion.ReporteValidacion(anio=2024)
    validacion._comparar_con_control(reporte, "x", 120.0, 100.0, 0.01)
    assert not reporte.paso
    assert len(reporte.fallidas) == 1


def test_comparacion_justo_en_el_borde_de_la_tolerancia_pasa():
    reporte = validacion.ReporteValidacion(anio=2024)
    validacion._comparar_con_control(reporte, "x", 101.0, 100.0, 0.01)
    assert reporte.paso


def test_comparacion_con_esperado_cero_exige_cero_exacto():
    """Un control de "debe ser exactamente 0" no admite tolerancia relativa."""
    reporte = validacion.ReporteValidacion(anio=2024)
    validacion._comparar_con_control(reporte, "identidad", 0.0, 0.0, 0.0)
    assert reporte.paso

    reporte_malo = validacion.ReporteValidacion(anio=2024)
    validacion._comparar_con_control(reporte_malo, "identidad", 0.5, 0.0, 0.0)
    assert not reporte_malo.paso


def test_metricas_de_hogares_cuenta_sonora_y_entidades(hogares_sinteticos):
    metricas = validacion._metricas_de_hogares(hogares_sinteticos)
    assert metricas["nacional_hogares_muestra"] == 40
    assert metricas["sonora_hogares_muestra"] == 20
    assert metricas["entidades_distintas"] == 2
    assert metricas["sonora_hogares_expandidos"] == pytest.approx(
        hogares_sinteticos.loc[hogares_sinteticos["entidad"] == "26", "factor"].sum()
    )


def test_leer_controles_ignora_filas_sin_anio(tmp_path):
    ruta = tmp_path / "controles.csv"
    ruta.write_text(
        "anio,ambito,valor_esperado,tolerancia_relativa\n"
        "2024,nacional_hogares_expandidos,38830230,0.03\n"
        ",fila_vacia,1,0.01\n",
        encoding="utf-8",
    )
    controles = validacion.leer_controles(ruta)
    assert len(controles) == 1
    assert controles[0]["anio"] == "2024"


def test_leer_controles_sin_archivo_devuelve_vacio(tmp_path):
    assert validacion.leer_controles(tmp_path / "no_existe.csv") == []


def test_controles_reales_declaran_los_anios_de_la_serie():
    """La tabla de control debe cubrir los años habilitados del proyecto."""
    from enigh import config

    controles = validacion.leer_controles()
    anios_control = {int(f["anio"]) for f in controles}
    for anio in config.cargar_configuracion().anios_habilitados:
        assert anio in anios_control, f"falta la cifra de control de {anio}"


def test_validar_anio_sin_datos_reporta_falla_con_instrucciones():
    """Un año sin procesar debe fallar de forma explícita, no silenciosa."""
    reporte = validacion.validar_anio(1899, controles=[])
    assert not reporte.paso
    assert "datos disponibles" in {c.nombre for c in reporte.fallidas}


def test_flujo_completo_sobre_datos_sinteticos(tmp_path, monkeypatch, hogares_sinteticos):
    """Recorre ``validar_anio`` con Parquet sintéticos en un directorio temporal.

    Comprueba que las invariantes se calculan y que una violación deliberada
    (participaciones que no suman 100) se detecta.
    """
    monkeypatch.setattr(
        validacion.limpieza, "cargar_hogares", lambda anio: hogares_sinteticos
    )
    monkeypatch.setattr(
        validacion.config, "ruta_agregados", lambda anio, nombre: tmp_path / f"{nombre}.parquet"
    )
    monkeypatch.setattr(
        validacion.config, "DIR_PROCESSED", tmp_path
    )

    deciles = pd.DataFrame(
        {
            "decil": list(range(1, 11)),
            "n_hogares": [2] * 10,
            "ing_pc": np.linspace(1000, 10000, 10),  # monótono
            "gini_dentro_del_decil": [0.17] * 10,
            "gini_global": [0.45] * 10,
            "participacion_ingreso": [10.0] * 10,  # suma 100: correcto
            "ing_pc_fiable": [True] * 10,
        }
    )
    deciles.to_parquet(tmp_path / "deciles_sonora.parquet", index=False)
    deciles.to_parquet(tmp_path / "deciles_nacionales.parquet", index=False)
    entidades = pd.DataFrame(
        {
            "entidad": ["26", "01"],
            "hogares_expandidos": [
                hogares_sinteticos.loc[hogares_sinteticos["entidad"] == "26", "factor"].sum(),
                hogares_sinteticos.loc[hogares_sinteticos["entidad"] == "01", "factor"].sum(),
            ],
        }
    )
    entidades.to_parquet(tmp_path / "entidades.parquet", index=False)

    reporte = validacion.validar_anio(2024, controles=[])
    nombres_ok = {c.nombre for c in reporte.comprobaciones if c.paso}
    assert "monotonicidad_ingreso_por_decil" in nombres_ok
    assert "participacion_total_deciles" in nombres_ok
    assert "gini_nacional_en_rango" in nombres_ok
    assert "gini_decil_no_confundido_con_global" in nombres_ok
    assert "ing_pc_no_mayor_que_ing_cor" in nombres_ok
    assert "coherencia_sonora_entre_archivos" in nombres_ok


def test_detecta_participaciones_que_no_suman_cien(tmp_path, monkeypatch, hogares_sinteticos):
    """Una tabla de deciles inconsistente debe hacer fallar la validación."""
    monkeypatch.setattr(
        validacion.limpieza, "cargar_hogares", lambda anio: hogares_sinteticos
    )
    monkeypatch.setattr(
        validacion.config, "ruta_agregados", lambda anio, nombre: tmp_path / f"{nombre}.parquet"
    )
    monkeypatch.setattr(validacion.config, "DIR_PROCESSED", tmp_path)

    deciles = pd.DataFrame(
        {
            "decil": list(range(1, 11)),
            "n_hogares": [2] * 10,
            "ing_pc": np.linspace(1000, 10000, 10),
            "gini_dentro_del_decil": [0.17] * 10,
            "gini_global": [0.45] * 10,
            "participacion_ingreso": [5.0] * 10,  # suma 50: incorrecto
            "ing_pc_fiable": [True] * 10,
        }
    )
    deciles.to_parquet(tmp_path / "deciles_sonora.parquet", index=False)
    deciles.to_parquet(tmp_path / "deciles_nacionales.parquet", index=False)
    pd.DataFrame({"entidad": ["26"], "hogares_expandidos": [1.0]}).to_parquet(
        tmp_path / "entidades.parquet", index=False
    )

    reporte = validacion.validar_anio(2024, controles=[])
    assert "participacion_total_deciles" in {c.nombre for c in reporte.fallidas}


def test_detecta_decil_no_monotono(tmp_path, monkeypatch, hogares_sinteticos):
    monkeypatch.setattr(
        validacion.limpieza, "cargar_hogares", lambda anio: hogares_sinteticos
    )
    monkeypatch.setattr(
        validacion.config, "ruta_agregados", lambda anio, nombre: tmp_path / f"{nombre}.parquet"
    )
    monkeypatch.setattr(validacion.config, "DIR_PROCESSED", tmp_path)

    ingresos = np.linspace(1000, 10000, 10).copy()
    ingresos[5], ingresos[6] = ingresos[6], ingresos[5]  # rompe la monotonía
    deciles = pd.DataFrame(
        {
            "decil": list(range(1, 11)),
            "n_hogares": [2] * 10,
            "ing_pc": ingresos,
            "gini_dentro_del_decil": [0.17] * 10,
            "gini_global": [0.45] * 10,
            "participacion_ingreso": [10.0] * 10,
            "ing_pc_fiable": [True] * 10,
        }
    )
    deciles.to_parquet(tmp_path / "deciles_sonora.parquet", index=False)
    deciles.to_parquet(tmp_path / "deciles_nacionales.parquet", index=False)
    pd.DataFrame({"entidad": ["26"], "hogares_expandidos": [1.0]}).to_parquet(
        tmp_path / "entidades.parquet", index=False
    )

    reporte = validacion.validar_anio(2024, controles=[])
    assert "monotonicidad_ingreso_por_decil" in {c.nombre for c in reporte.fallidas}
