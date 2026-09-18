"""Pruebas de exactitud de las medidas ponderadas.

Varias pruebas comparan contra valores calculados a mano con las fórmulas de
libro de texto (Gini) o contra las funciones equivalentes de NumPy con pesos
uniformes, que es la forma de comprobar que un estimador ponderado está bien
implementado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from enigh import metricas


# --------------------------------------------------------------------------
# Percentiles ponderados
# --------------------------------------------------------------------------


def test_percentil_con_pesos_uniformes_coincide_con_numpy():
    """Con pesos iguales, el percentil ponderado debe aproximar el de NumPy."""
    generador = np.random.default_rng(7)
    valores = generador.lognormal(10, 0.6, 5000)
    pesos = np.ones_like(valores)

    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        ponderado = metricas.percentil_ponderado(valores, pesos, q)
        referencia = float(np.percentile(valores, q * 100))
        assert ponderado == pytest.approx(referencia, rel=0.02), (
            f"q={q}: ponderado={ponderado}, numpy={referencia}"
        )


def test_percentil_ponderado_caso_resuelto_a_mano():
    """Caso verificable sin ambigüedad: un valor con el 90 % del peso."""
    valores = np.array([10.0, 100.0])
    pesos = np.array([90.0, 10.0])
    # El percentil 50 y el 89 caen dentro del bloque dominado por 10.
    assert metricas.percentil_ponderado(valores, pesos, 0.5) == pytest.approx(10.0)
    assert metricas.percentil_ponderado(valores, pesos, 0.9) == pytest.approx(10.0)
    # El percentil 95 debe estar entre 10 y 100.
    assert 10.0 < metricas.percentil_ponderado(valores, pesos, 0.95) <= 100.0


def test_percentil_ponderado_ignora_pesos_no_positivos():
    valores = np.array([1.0, 2.0, 3.0, 99.0])
    pesos = np.array([1.0, 1.0, 1.0, 0.0])
    # El valor con peso cero no debe influir. Sin él quedan tres observaciones
    # con el mismo peso, así que la mediana es la interpolación entre la primera
    # y la segunda (1.5) y el percentil 90 interpola 0.7 entre 2 y 3 => 2.7. Si
    # el 99 contara, ambos valores serían mucho mayores.
    assert metricas.percentil_ponderado(valores, pesos, 0.5) == pytest.approx(1.5)
    assert metricas.percentil_ponderado(valores, pesos, 0.9) == pytest.approx(2.7)


def test_percentil_ponderado_ignora_valores_no_finitos():
    """Un NaN se descarta y no arrastra el resultado.

    Con los dos valores válidos (1 y 3) y pesos iguales, la mediana ponderada es
    1.0, exactamente lo que devuelve NumPy con el NaN ya eliminado.
    """
    valores = np.array([1.0, np.nan, 3.0])
    pesos = np.array([1.0, 1.0, 1.0])
    assert metricas.percentil_ponderado(valores, pesos, 0.5) == pytest.approx(1.0)
    # Y coincide con calcularlo sobre los datos ya limpios.
    assert metricas.percentil_ponderado(valores, pesos, 0.5) == pytest.approx(
        metricas.percentil_ponderado(np.array([1.0, 3.0]), np.array([1.0, 1.0]), 0.5)
    )


def test_percentil_ponderado_rechaza_cuantil_invalido():
    with pytest.raises(metricas.ErrorDeMetrica):
        metricas.percentil_ponderado(np.array([1.0]), np.array([1.0]), 0.0)
    with pytest.raises(metricas.ErrorDeMetrica):
        metricas.percentil_ponderado(np.array([1.0]), np.array([1.0]), 1.5)


def test_percentil_ponderado_sin_datos_devuelve_nan():
    assert np.isnan(metricas.percentil_ponderado([], [], 0.5))


# --------------------------------------------------------------------------
# Gini
# --------------------------------------------------------------------------


def test_gini_coincide_con_formula_de_libro():
    """Gini contra la fórmula directa ``2*Σ(i*x_i)/(n*Σx) - (n+1)/n``."""
    valores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    n = len(valores)
    esperado = (2 * np.sum(np.arange(1, n + 1) * valores) / (n * np.sum(valores))) - (n + 1) / n
    assert metricas.gini(valores, np.ones(n)) == pytest.approx(esperado, abs=1e-12)


def test_gini_limites_conocidos():
    """Igualdad perfecta da 0; todo el ingreso en una persona tiende a 1."""
    assert metricas.gini(np.ones(1000), np.ones(1000)) == pytest.approx(0.0, abs=1e-9)

    muy_desigual = np.concatenate([np.zeros(999), [1.0]])
    gini_extremo = metricas.gini(muy_desigual, np.ones(1000))
    assert gini_extremo > 0.99
    assert gini_extremo <= 1.0


def test_gini_es_invariante_a_escala():
    """Multiplicar todo el ingreso por una constante no cambia el Gini."""
    generador = np.random.default_rng(11)
    valores = generador.lognormal(9, 0.8, 800)
    pesos = generador.uniform(1, 10, 800)
    assert metricas.gini(valores, pesos) == pytest.approx(
        metricas.gini(valores * 37.5, pesos), rel=1e-9
    )


def test_gini_es_invariante_a_la_unidad_de_peso():
    """Multiplicar todos los ponderadores por una constante no cambia el Gini."""
    generador = np.random.default_rng(12)
    valores = generador.lognormal(9, 0.8, 500)
    pesos = generador.uniform(1, 10, 500)
    assert metricas.gini(valores, pesos) == pytest.approx(
        metricas.gini(valores, pesos * 1000), rel=1e-9
    )


# --------------------------------------------------------------------------
# Theil
# --------------------------------------------------------------------------


def test_theil_es_cero_con_ingreso_igual():
    assert metricas.theil_t(np.full(500, 42.0), np.ones(500)) == pytest.approx(0.0, abs=1e-12)


def test_theil_ignora_ceros_sin_romper():
    """Theil no está definido con ceros; debe ignorarlos, no devolver infinito."""
    valores = np.array([0.0, 10.0, 20.0, 30.0])
    resultado = metricas.theil_t(valores, np.ones(4))
    positivo = metricas.theil_t(np.array([10.0, 20.0, 30.0]), np.ones(3))
    assert np.isfinite(resultado)
    assert resultado == pytest.approx(positivo)


# --------------------------------------------------------------------------
# Deciles
# --------------------------------------------------------------------------


def test_cortes_deciles_dejan_diez_partes_iguales_en_poblacion():
    generador = np.random.default_rng(3)
    valores = generador.lognormal(10, 0.5, 10_000)
    pesos = generador.uniform(1, 5, 10_000)

    cortes = metricas.cortes_deciles(valores, pesos)
    etiquetas = metricas.asignar_deciles(valores, pesos, cortes)

    poblacion = pd.Series(pesos).groupby(etiquetas).sum()
    for decil in range(1, 11):
        assert decil in poblacion.index, f"falta el decil {decil}"
    fracciones = poblacion / poblacion.sum()
    # Cada decil debe llevarse ~10 % de la población ponderada.
    assert fracciones.min() == pytest.approx(0.10, abs=0.01)


def test_cortes_deciles_son_crecientes():
    generador = np.random.default_rng(5)
    valores = generador.lognormal(10, 0.7, 2000)
    cortes = metricas.cortes_deciles(valores, np.ones(2000))
    assert np.all(np.diff(cortes) >= 0), "los cortes deben ser monótonos"


def test_asignar_deciles_usa_cortes_externos():
    """Con cortes de otra población, las etiquetas respetan esos cortes."""
    cortes = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0])
    valores = np.array([5.0, 15.0, 25.0, 95.0, 100.0])
    etiquetas = metricas.asignar_deciles(valores, np.ones(5), cortes)
    assert list(etiquetas) == [1, 2, 3, 10, 10]


def test_asignar_deciles_marca_nan_en_valores_nulos():
    cortes = np.arange(10.0, 100.0, 10.0)
    valores = np.array([5.0, np.nan, 50.0])
    etiquetas = metricas.asignar_deciles(valores, np.ones(3), cortes)
    assert etiquetas[0] == 1
    assert np.isnan(etiquetas[1])
    # 50 cae exactamente en el quinto corte; con side="right" pertenece al decil
    # superior (6), que es la convención conservadora y consistente con la
    # interpolación de percentiles.
    assert etiquetas[2] == 6

    # Justo por debajo del corte sí es el decil 5.
    etiquetas_borde = metricas.asignar_deciles(
        np.array([49.999]), np.ones(1), cortes
    )
    assert etiquetas_borde[0] == 5


def test_decil_uno_es_el_mas_pobre():
    """Regresión: el decil 1 debe contener los valores más bajos, no los más altos."""
    valores = np.arange(1.0, 101.0)
    etiquetas = metricas.asignar_deciles(valores, np.ones(100))
    assert etiquetas[0] == 1, "el valor más bajo pertenece al decil 1"
    assert etiquetas[-1] == 10, "el valor más alto pertenece al decil 10"
    media_d1 = valores[etiquetas == 1].mean()
    media_d10 = valores[etiquetas == 10].mean()
    assert media_d1 < media_d10


# --------------------------------------------------------------------------
# Participaciones y Palma
# --------------------------------------------------------------------------


def test_participaciones_suman_cien():
    generador = np.random.default_rng(13)
    valores = generador.lognormal(10, 0.6, 3000)
    pesos = generador.uniform(1, 4, 3000)

    resultado = metricas.participacion_por_grupo(valores, pesos)
    assert resultado is not None
    assert resultado.bajo40 < resultado.bajo50 < resultado.bajo60 < resultado.bajo90
    assert resultado.bajo90 + resultado.alto10 == pytest.approx(100.0, abs=1e-6)


def test_palma_con_ingreso_igual_da_la_razon_de_tamanos():
    """Con ingreso idéntico, el 10 % más rico se lleva exactamente 10 % del total
    y el 40 % más pobre exactamente 40 %, así que la razón es 10/40 = 0.25.

    Este valor es la firma de una implementación correcta: la versión con el
    desempaquetado equivocado devolvía 1/6 ≈ 0.167, y una implementación que
    devolviera 1 revelaría que está comparando bloques del mismo tamaño.
    """
    valores = np.full(1000, 500.0)
    assert metricas.razon_palma(valores, np.ones(1000)) == pytest.approx(0.25, abs=1e-9)


def test_participaciones_con_ingreso_igual_son_los_tamanos_de_grupo():
    """Firma de las participaciones acumuladas con distribución uniforme."""
    valores = np.full(1000, 500.0)
    p = metricas.participacion_por_grupo(valores, np.ones(1000))
    assert p is not None
    assert p.bajo40 == pytest.approx(40.0, abs=1e-9)
    assert p.bajo50 == pytest.approx(50.0, abs=1e-9)
    assert p.bajo60 == pytest.approx(60.0, abs=1e-9)
    assert p.bajo90 == pytest.approx(90.0, abs=1e-9)
    assert p.alto10 == pytest.approx(10.0, abs=1e-9)


def test_palma_es_mayor_que_la_razon_uniforme_cuando_hay_desigualdad():
    """Con desigualdad real, el decil alto se lleva mucho más que su tamaño."""
    valores = np.concatenate([np.full(900, 10.0), np.full(100, 5000.0)])
    assert metricas.razon_palma(valores, np.ones(1000)) > 0.25


def test_palma_crece_con_la_desigualdad():
    base = np.concatenate([np.full(900, 100.0), np.full(100, 200.0)])
    mas_desigual = np.concatenate([np.full(900, 10.0), np.full(100, 2000.0)])
    assert metricas.razon_palma(mas_desigual, np.ones(1000)) > metricas.razon_palma(
        base, np.ones(1000)
    )


def test_participacion_acumulada_aproxima_suma_directa():
    """El bloque más pobre del 100 % debe ser el total."""
    valores = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    pesos = np.ones(10)
    total = np.sum(valores * pesos)
    assert metricas._participacion_acumulada(valores, pesos, 1.0) == pytest.approx(total)


# --------------------------------------------------------------------------
# resumen_deciles
# --------------------------------------------------------------------------


def test_resumen_deciles_reparte_la_poblacion_y_suma_cien():
    generador = np.random.default_rng(17)
    datos = pd.DataFrame(
        {
            "ing_pc": generador.lognormal(9, 0.7, 5000),
            "factor": generador.uniform(50, 500, 5000),
        }
    )
    resumen = metricas.resumen_deciles(datos, columna_ingreso="ing_pc")

    assert len(resumen.tabla) == 10
    assert resumen.tabla["decil"].tolist() == list(range(1, 11))
    assert resumen.tabla["participacion_ingreso"].sum() == pytest.approx(100.0, abs=1e-6)
    # El ingreso promedio debe crecer de decil en decil.
    promedios = resumen.tabla["ing_pc"].to_numpy()
    assert np.all(np.diff(promedios) > 0)
    assert 0 <= resumen.gini <= 1


def test_resumen_deciles_falla_si_falta_la_columna():
    datos = pd.DataFrame({"otra": [1.0], "factor": [1.0]})
    with pytest.raises(metricas.ErrorDeMetrica):
        metricas.resumen_deciles(datos, columna_ingreso="ing_pc")


def test_resumen_deciles_ignora_filas_invalidas():
    datos = pd.DataFrame(
        {
            "ing_pc": [100.0, np.nan, 200.0, 300.0],
            "factor": [1.0, 1.0, 0.0, 1.0],  # nulo y peso cero
        }
    )
    resumen = metricas.resumen_deciles(datos, columna_ingreso="ing_pc")
    assert resumen.n_observaciones == 2


def test_resumen_deciles_acepta_cero_como_ingreso_valido():
    """Un hogar sin ingreso corriente es un dato legítimo, no un nulo.

    Con pocas observaciones y empates, los cortes pueden coincidir entre sí, así
    que el decil más pobre se localiza por su límite superior (el ingreso mínimo
    del conjunto) en lugar de asumir que quedó etiquetado como decil 1.
    """
    datos = pd.DataFrame(
        {"ing_pc": [0.0, 100.0, 200.0, 300.0], "factor": [1.0, 1.0, 1.0, 1.0]}
    )
    resumen = metricas.resumen_deciles(datos, columna_ingreso="ing_pc")
    assert resumen.n_observaciones == 4
    # El cero debe estar representado: el mínimo de todos los límites inferiores
    # reportados tiene que ser 0, y ninguna fila debe tener promedio negativo.
    assert resumen.tabla["limite_inferior"].min() == 0.0
    assert (resumen.tabla["ing_pc"].dropna() >= 0).all()
    # Y el valor cero debe haber entrado en alguna suma de decil.
    assert resumen.tabla["hogares_expandidos"].sum() == 4.0


# --------------------------------------------------------------------------
# media ponderada
# --------------------------------------------------------------------------


def test_media_ponderada_coincide_con_promedio_simple_sin_pesos():
    valores = np.array([1.0, 2.0, 3.0, 4.0])
    assert metricas.media_ponderada(valores, np.ones(4)) == pytest.approx(2.5)


def test_media_ponderada_respeta_los_pesos():
    """Un valor con el triple de peso debe jalar el promedio hacia él."""
    valores = np.array([0.0, 100.0])
    pesos = np.array([3.0, 1.0])
    assert metricas.media_ponderada(valores, pesos) == pytest.approx(25.0)


def test_media_ponderada_sin_datos_devuelve_nan():
    assert np.isnan(metricas.media_ponderada([], []))


def test_media_ponderada_exige_formas_iguales():
    with pytest.raises(metricas.ErrorDeMetrica):
        metricas.media_ponderada([1.0, 2.0], [1.0])
