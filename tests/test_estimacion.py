"""Pruebas del bootstrap con diseño muestral.

Lo que se verifica aquí no es que el intervalo sea exacto (eso depende de la
distribución), sino las propiedades que, de fallar, producirían conclusiones
falsas en el tablero:

* que el resultado sea reproducible con la misma semilla;
* que el error estándar crezca al reducir la muestra;
* que una celda sin diseño muestral (sin UPM) se marque como no fiable en lugar
  de devolver un intervalo inventado;
* que el CV y la bandera ``fiable`` se comporten como el tablero espera.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from enigh import estimacion


def _datos(n_estratos: int = 10, upm_por_estrato: int = 12, tamano_upm: int = 4):
    """Microdatos sintéticos con diseño muestral completo y verificable."""
    generador = np.random.default_rng(20240101)
    filas = []
    for estrato in range(1, n_estratos + 1):
        for upm in range(1, upm_por_estrato + 1):
            # Cada UPM tiene su propio nivel, que es lo que hace que el diseño
            # importe: hogares dentro de una UPM se parecen entre sí.
            nivel_upm = 1000 + estrato * 50 + upm * 400
            for _ in range(tamano_upm):
                filas.append(
                    {
                        "est_dis": f"{estrato:03d}",
                        "upm": f"{estrato:03d}{upm:04d}",
                        "factor": 10.0 + generador.uniform(0, 5),
                        "ing_pc": max(0.0, nivel_upm + generador.normal(0, 60)),
                    }
                )
    return pd.DataFrame(filas)


def test_estimacion_devuelve_valor_y_banda_de_confianza():
    datos = _datos()
    resultado = estimacion._estimar_una(
        datos,
        estimacion.estadistico_media("ing_pc"),
        replicas=120,
        semilla=1,
        cv_no_fiable=0.25,
        upm_min=20,
    )
    assert np.isfinite(resultado.valor)
    assert np.isfinite(resultado.error_estandar)
    assert resultado.ic_inferior < resultado.valor < resultado.ic_superior
    assert resultado.n_hogares == len(datos)
    assert resultado.n_upm == 120


def test_estimacion_es_reproducible_con_la_misma_semilla():
    datos = _datos()
    argumentos = dict(replicas=80, semilla=12345, cv_no_fiable=0.25, upm_min=20)
    primera = estimacion._estimar_una(datos, estimacion.estadistico_media("ing_pc"), **argumentos)
    segunda = estimacion._estimar_una(datos, estimacion.estadistico_media("ing_pc"), **argumentos)
    assert primera.error_estandar == pytest.approx(segunda.error_estandar, rel=1e-12)
    assert primera.ic_inferior == pytest.approx(segunda.ic_inferior, rel=1e-12)


def test_estimacion_depende_de_la_semilla_si_se_cambia():
    datos = _datos()
    base = dict(replicas=80, cv_no_fiable=0.25, upm_min=20)
    primera = estimacion._estimar_una(
        datos, estimacion.estadistico_media("ing_pc"), semilla=1, **base
    )
    segunda = estimacion._estimar_una(
        datos, estimacion.estadistico_media("ing_pc"), semilla=2, **base
    )
    assert primera.error_estandar != segunda.error_estandar


def test_error_estandar_crece_al_reducir_la_muestra():
    """La incertidumbre debe reflejar el tamaño: menos muestra, más error."""
    grande = _datos(n_estratos=12, upm_por_estrato=15, tamano_upm=5)
    chico = _datos(n_estratos=3, upm_por_estrato=4, tamano_upm=5)
    argumentos = dict(replicas=150, semilla=7, cv_no_fiable=0.25, upm_min=10)

    ee_grande = estimacion._estimar_una(
        grande, estimacion.estadistico_media("ing_pc"), **argumentos
    ).error_estandar
    ee_chico = estimacion._estimar_una(
        chico, estimacion.estadistico_media("ing_pc"), **argumentos
    ).error_estandar
    assert ee_chico > ee_grande


def test_celda_sin_variables_de_diseno_se_marca_no_fiable():
    datos = pd.DataFrame({"ing_pc": [1.0, 2.0, 3.0], "factor": [1.0, 1.0, 1.0]})
    resultado = estimacion._estimar_una(
        datos,
        estimacion.estadistico_media("ing_pc"),
        replicas=50, semilla=1, cv_no_fiable=0.25, upm_min=20,
    )
    assert resultado.fiable is False
    assert "diseño" in resultado.motivo
    # El valor puntual sí se reporta: es la incertidumbre la que no se puede medir.
    assert np.isfinite(resultado.valor)


def test_celda_con_pocas_upm_se_marca_no_fiable():
    datos = _datos(n_estratos=2, upm_por_estrato=2, tamano_upm=3)
    resultado = estimacion._estimar_una(
        datos,
        estimacion.estadistico_media("ing_pc"),
        replicas=50, semilla=1, cv_no_fiable=0.25, upm_min=20,
    )
    assert resultado.fiable is False
    assert "UPM" in resultado.motivo


def test_celda_sin_observaciones_no_revienta():
    vacio = pd.DataFrame({"ing_pc": [], "factor": [], "est_dis": [], "upm": []})
    resultado = estimacion._estimar_una(
        vacio,
        estimacion.estadistico_media("ing_pc"),
        replicas=20, semilla=1, cv_no_fiable=0.25, upm_min=1,
    )
    assert resultado.n_hogares == 0
    assert resultado.fiable is False
    assert not np.isfinite(resultado.valor)


def test_cv_marca_como_no_fiable_cuando_es_alto():
    """Con umbral muy estricto, todo debe quedar marcado como no fiable."""
    datos = _datos()
    resultado = estimacion._estimar_una(
        datos,
        estimacion.estadistico_media("ing_pc"),
        replicas=60, semilla=3, cv_no_fiable=1e-9, upm_min=20,
    )
    assert resultado.fiable is False
    assert resultado.cv > 0


def test_estimar_por_grupo_produce_una_fila_por_decil():
    datos = _datos()
    datos["decil"] = np.repeat(np.arange(1, 11), len(datos) // 10)
    resultado = estimacion.estimar_por_grupo(
        datos,
        "decil",
        {"ing_pc": estimacion.estadistico_media("ing_pc")},
        replicas=40,
    )
    assert len(resultado.tabla) == 10
    assert "ing_pc_valor" in resultado.tabla.columns
    assert "ing_pc_cv" in resultado.tabla.columns
    assert "ing_pc_fiable" in resultado.tabla.columns


def test_estimar_por_grupo_reporta_avisos_de_poca_muestra():
    datos = _datos(n_estratos=1, upm_por_estrato=1, tamano_upm=2)
    datos["decil"] = 1
    resultado = estimacion.estimar_por_grupo(
        datos, "decil", {"ing_pc": estimacion.estadistico_media("ing_pc")}, replicas=30
    )
    assert resultado.avisos, "debería avisar de que la muestra es insuficiente"


def test_estadistico_proporcion_calcula_la_fraccion_ponderada():
    datos = pd.DataFrame(
        {
            "tiene_internet": [1.0, 1.0, 0.0, 0.0],
            "factor": [3.0, 3.0, 1.0, 1.0],
        }
    )
    proporcion = estimacion.estadistico_proporcion("tiene_internet")(datos)
    # (3+3) / 8 = 0.75
    assert proporcion == pytest.approx(0.75)


def test_estadistico_proporcion_ignora_nulos_en_el_denominador():
    datos = pd.DataFrame(
        {"tiene_internet": [1.0, np.nan], "factor": [1.0, 9.0]}
    )
    proporcion = estimacion.estadistico_proporcion("tiene_internet")(datos)
    assert proporcion == pytest.approx(1.0)


def test_estadistico_mediana_usa_ponderadores():
    datos = pd.DataFrame({"x": [10.0, 20.0], "factor": [9.0, 1.0]})
    mediana = estimacion.estadistico_mediana("x")(datos)
    assert mediana == pytest.approx(10.0)
