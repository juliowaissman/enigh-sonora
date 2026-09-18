"""Lectura, validación y normalización de los microdatos de la ENIGH a Parquet.

Responsabilidades:

1. Leer los CSV detectando su codificación (INEGI no es consistente entre años:
   2020 viene en ``utf-8-sig`` y 2022 en ``latin-1``), además de comillas y
   valores nulos propios.
2. Validar contra el diccionario de datos y contra ``config/fuentes.yaml``,
   fallando con el **nombre exacto** de la columna faltante.
3. Derivar las columnas de trabajo: identificadores territoriales, ingreso per
   cápita, ingreso por adulto equivalente e ingreso real.
4. Agregar los indicadores de bienestar desde las tablas ``poblacion``,
   ``hogares`` y ``viviendas`` a nivel hogar.

Convención monetaria: la ENIGH reporta montos **trimestrales en pesos
corrientes** del año del levantamiento. Aquí no se anualiza nada.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .extraccion import TablaExtraida, leer_diccionario

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Columnas de trabajo
# --------------------------------------------------------------------------

# Identificadores y diseño muestral.
COLS_ID = ["folioviv", "foliohog", "ubica_geo", "factor", "est_dis", "upm", "tam_loc", "est_socio"]

# Composición del hogar y características del jefe.
COLS_HOGAR = [
    "tot_integ", "hombres", "mujeres", "menores", "mayores", "p12_64", "p65mas",
    "ocupados", "percep_ing", "perc_ocupa", "clase_hog",
    "sexo_jefe", "edad_jefe", "educa_jefe",
]

# Bloque de ingresos y su desglose.
COLS_INGRESO = [
    "ing_cor", "ingtrab", "trabajo", "sueldos", "horas_extr", "comisiones",
    "aguinaldo", "indemtrab", "otra_rem", "remu_espec", "negocio", "noagrop",
    "industria", "comercio", "servicios", "agrope", "agricolas", "pecuarios",
    "reproducc", "pesca", "otros_trab", "rentas", "utilidad", "arrenda",
    "transfer", "jubilacion", "becas", "donativos", "remesas", "bene_gob",
    "transf_hog", "trans_inst", "estim_alqu", "otros_ing",
]

# Bloque de gastos (solo los rubros que usa el tablero y el cálculo de bienestar).
COLS_GASTO = [
    "gasto_mon", "alimentos", "ali_dentro", "ali_fuera", "bebidas", "tabaco",
    "vesti_calz", "vivienda", "salud", "transporte", "educa_espa", "educacion",
    "esparci", "personales", "comunica", "limpieza", "energia", "agua",
    "pred_cons", "cuidados", "utensilios", "enseres", "otros_gas",
]

# Variables monetarias a las que se aplica la deflactación.
COLS_MONETARIAS = COLS_INGRESO + COLS_GASTO

# Indicadores de la tabla `poblacion` agregados a nivel hogar.
#
# ``valor_afirmativo`` es la clave que INEGI usa para "sí" en cada variable. El
# catálogo ``si_no`` de INEGI codifica **1 = Sí y 2 = No**, así que contar
# "cualquier valor mayor que cero" trataría el "no" como afirmativo. Ese error
# pasó inadvertido en la primera versión porque producía porcentajes plausibles
# a simple vista (cerca de 100 %), aunque la cifra real sea ~37 %.
#
# ``disc_ver`` es la excepción: usa el catálogo ``disc`` con rango [&,1-4], donde
# 1 a 4 son grados de dificultad y "&" es sin discapacidad; ahí sí se cuenta
# "mayor que cero", por eso su valor afirmativo es ``None``.
INDICADORES_POBLACION: dict[str, tuple[str, str, float | None]] = {
    # nombre_salida: (columna_origen, descripción, valor afirmativo o None si > 0)
    "asis_esc": ("asis_esc", "Integrantes que asisten a la escuela", 1.0),
    "alfabetism": ("alfabetism", "Integrantes alfabetas", 1.0),
    "segsoc": ("segsoc", "Integrantes con cobertura de seguridad social", 1.0),
    "etnia": ("etnia", "Integrantes que se autoidentifican como indígenas", 1.0),
    "afrod": ("afrod", "Integrantes afrodescendientes", 1.0),
    "disc_ver": ("disc_ver", "Integrantes con discapacidad visual", None),
    "aten_sal": ("aten_sal", "Integrantes que recibieron atención médica al enfermar", 1.0),
}

# Claves de INEGI para "no sabe" / "no aplica": se tratan como ausencia para no
# inflar el denominador de un indicador binario.
VALORES_NO_APLICA = (9.0, 99.0, 999.0)

# Indicadores de la tabla `hogares` (ya vienen por hogar, sin agregación).
INDICADORES_HOGARES: dict[str, str] = {
    "conex_inte": "El hogar cuenta con conexión a internet",
    "celular": "El hogar cuenta con teléfono celular",
    "telefono": "El hogar cuenta con línea telefónica fija",
    "num_compu": "Computadoras en el hogar",
    "num_auto": "Automóviles en el hogar",
    "num_bici": "Bicicletas en el hogar",
    "num_refri": "Refrigeradores en el hogar",
    "num_lavad": "Lavadoras en el hogar",
    "tarjeta": "El hogar cuenta con tarjeta de crédito",
    "af_empleo": "Afiliación laboral de algún integrante",
    "nr_viv": "Número de viviendas del hogar",
}

# Indicadores de la tabla `viviendas`.
INDICADORES_VIVIENDAS: dict[str, str] = {
    "tipo_viv": "Tipo de vivienda",
    "agua_ent": "Agua entubada en la vivienda",
    "drenaje": "Drenaje conectado",
    "excusado": "Excusado de uso exclusivo",
    "disp_elect": "Disponibilidad de energía eléctrica",
    "num_cuarto": "Número de cuartos de la vivienda",
    "cuart_dorm": "Número de cuartos usados para dormir",
    "tenencia": "Forma de tenencia de la vivienda",
    "mat_pisos": "Material de los pisos",
    "mat_techos": "Material de los techos",
    "combus": "Combustible usado para cocinar",
    "fogon_chi": "Tipo de fogón o chimenea al cocinar",
}

# Acceso a alimentos reportado por el hogar (16 reactivos + 1 general).
COLS_ACC_ALIM = [f"acc_alim{i}" for i in range(1, 17)] + ["acc_alim18"]

# Variables que deben leerse como texto aunque parezcan números (preservan ceros
# a la izquierda en claves geográficas y de diseño).
#
# ``educa_jefe`` y ``sexo_jefe`` también van como texto: sus claves son códigos
# ("01".."11"), no cantidades. Leerlas como número convertiría "01" en 1 y haría
# que cualquier comparación contra el catálogo de INEGI fallara en silencio.
COLS_TEXTO = ["folioviv", "foliohog", "ubica_geo", "est_dis", "upm", "clase_hog",
              "sexo_jefe", "educa_jefe", "tam_loc", "est_socio"]

VALORES_NULOS = ["", "NA", "N/A", " ", ".", "NULL", "null", "No especificado"]

# Codificaciones que usa INEGI, en orden de intento. El orden importa: probar
# latin-1 primero "funciona" siempre (decodifica cualquier byte) y destrozaría
# los acentos del UTF-8. Verificado sobre los archivos reales: la ENIGH 2020
# viene en utf-8-sig y la de 2022 en latin-1, así que no se puede fijar una sola.
CODIFICACIONES = ("utf-8-sig", "utf-8", "latin-1", "cp1252")


def detectar_codificacion(ruta: Path) -> str:
    """Devuelve la primera codificación de :data:`CODIFICACIONES` que decodifica.

    Como ``latin-1`` acepta cualquier secuencia de bytes, siempre hay respuesta;
    se usa como último recurso para nunca fallar por codificación.
    """
    crudo = ruta.read_bytes()
    for codificacion in CODIFICACIONES:
        try:
            crudo.decode(codificacion)
            return codificacion
        except (UnicodeDecodeError, LookupError):
            continue
    return "latin-1"


class ErrorDeDatos(RuntimeError):
    """Los microdatos no tienen la forma esperada."""


# --------------------------------------------------------------------------
# Lectura de la tabla principal
# --------------------------------------------------------------------------


def _catalogo_ubica_geo(tabla: TablaExtraida) -> pd.DataFrame:
    """Catálogo de ubicación geográfica con nombres de entidad y municipio."""
    if tabla.directorio_catalogos is None:
        return pd.DataFrame(columns=["ubica_geo", "desc_ent", "desc_mun"])
    ruta = tabla.directorio_catalogos / "ubica_geo.csv"
    if not ruta.exists():
        return pd.DataFrame(columns=["ubica_geo", "desc_ent", "desc_mun"])
    cat = pd.read_csv(
        ruta,
        dtype=str,
        encoding=detectar_codificacion(ruta),
        keep_default_na=False,
        na_values=VALORES_NULOS,
    )
    cat["ubica_geo"] = cat["ubica_geo"].astype(str).str.zfill(5)
    for columna in ("desc_ent", "desc_mun"):
        if columna not in cat.columns:
            cat[columna] = pd.NA
    return cat[["ubica_geo", "desc_ent", "desc_mun"]].drop_duplicates("ubica_geo")


def _validar_columnas(
    presentes: list[str],
    requeridas: list[str],
    diccionario: dict[str, dict[str, str]],
    anio: int,
) -> None:
    """Falla con el nombre exacto de la primera columna faltante."""
    faltantes = [c for c in requeridas if c not in presentes]
    if faltantes:
        conocidas = ", ".join(sorted(diccionario)[:40])
        raise ErrorDeDatos(
            f"[{anio}] Faltan columnas en concentradohogar: {faltantes}. "
            f"La tabla trae {len(presentes)} columnas. "
            f"Columnas del diccionario (primeras 40): {conocidas}"
        )


def _a_numero(serie: pd.Series) -> pd.Series:
    """Convierte una serie a numérico tolerando comas de miles, espacios y celdas vacías.

    Cuidado con ``pd.to_numeric`` sobre el dtype ``string`` de pandas: devuelve
    ``NaN`` para **todos** los elementos, sin lanzar error. Fue exactamente el
    origen de un bug real: los indicadores de personas (seguridad social,
    alfabetismo, asistencia escolar) quedaban como ``den == num`` y el porcentaje
    calculado daba 100 % constante. Por eso se normaliza a texto explícitamente y
    se verifica el resultado.
    """
    # Un dtype ya numérico se devuelve tal cual.
    if serie.dtype.kind in "if":
        return serie

    limpio = serie.astype("string").str.strip()
    # Normaliza variantes de ausencia antes de convertir.
    limpio = limpio.replace({"": None, " ": None, "NA": None, "N/A": None, ".": None})
    for codigo in VALORES_NULOS:
        limpio = limpio.mask(limpio == codigo, None)
    limpio = limpio.str.replace(",", "", regex=False)

    resultado = pd.to_numeric(limpio, errors="coerce")

    # Salvaguarda explícita: si la columna tiene valores de cadena interpretables
    # como números pero el resultado salió todo nulo, la conversión falló en
    # silencio. Es preferible enterarse aquí que publicar un indicador constante.
    if resultado.isna().all() and limpio.notna().any():
        muestra = limpio.dropna().head(5).tolist()
        raise ErrorDeDatos(
            f"La conversión a número devolvió todo nulo para una columna con "
            f"{int(limpio.notna().sum())} valores no vacíos (ejemplos: {muestra}). "
            f"Revise la codificación y el tipo de la columna."
        )
    return resultado


# Columnas que son códigos categóricos, no cantidades. Convertirlas a número
# destruiría los ceros a la izquierda ("01" -> 1) y haría fallar en silencio
# cualquier comparación contra el catálogo de INEGI.
COLS_CODIGO = ("educa_jefe", "sexo_jefe", "clase_hog")


def _a_numero_si_aplica(serie: pd.Series, nombre: str) -> pd.Series:
    """Numérica salvo para los códigos categóricos, que se conservan como texto."""
    if nombre in COLS_CODIGO and serie.dtype.kind in "O":
        return serie.astype("string").str.strip()
    return _a_numero(serie)


def cargar_concentradohogar(
    tabla: TablaExtraida,
    *,
    cfg: config.Configuracion | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Lee ``concentradohogar``, valida y deriva las columnas de trabajo.

    Returns:
        ``(hogares, catalogo_geo, reporte)`` donde ``reporte`` documenta
        cuántas filas se descartaron y por qué.
    """
    cfg = cfg or config.cargar_configuracion()
    anio = tabla.anio
    diccionario = leer_diccionario(tabla.csv_diccionario) if tabla.csv_diccionario else {}

    # Primera pasada: conocer las columnas reales para no depender del diccionario.
    muestra = pd.read_csv(
        tabla.csv_datos, nrows=5, dtype=str,
        encoding=detectar_codificacion(tabla.csv_datos),
    )
    presentes = list(muestra.columns)

    if diccionario:
        faltan_en_diccionario = [c for c in presentes if c not in diccionario]
        if faltan_en_diccionario:
            log.info(
                "[%s] %s columnas del CSV no están en el diccionario (se conservan igual): %s",
                anio, len(faltan_en_diccionario), faltan_en_diccionario[:10],
            )

    requeridas = cfg.columnas_requeridas_concentradohogar
    _validar_columnas(presentes, requeridas, diccionario, anio)

    # Seleccionamos solo lo necesario: el CSV completo pesa 46 MB y no hace falta.
    deseables = set(COLS_ID + COLS_HOGAR + COLS_INGRESO + COLS_GASTO + ["smg"])
    usar = [c for c in presentes if c in deseables] + [
        c for c in requeridas if c not in deseables and c in presentes
    ]
    usar = list(dict.fromkeys(usar))

    dtype = {c: "string" for c in usar if c in COLS_TEXTO}
    hogares = pd.read_csv(
        tabla.csv_datos,
        usecols=usar,
        dtype=dtype,
        encoding=detectar_codificacion(tabla.csv_datos),
        na_values=VALORES_NULOS,
        thousands=None,
    )

    reporte: dict[str, object] = {
        "anio": anio,
        "filas_crudas": int(len(hogares)),
        "columnas_leidas": len(usar),
        "columnas_documentadas": len(diccionario),
    }

    # --- Normalización de identificadores ---
    for columna in ("folioviv", "foliohog", "ubica_geo"):
        if columna in hogares.columns:
            hogares[columna] = hogares[columna].astype(str).str.strip()
    hogares["ubica_geo"] = hogares["ubica_geo"].str.zfill(5)
    hogares["entidad"] = hogares["ubica_geo"].str[:2]
    hogares["municipio"] = hogares["ubica_geo"].str[2:5]

    mal_ubicados = ~hogares["ubica_geo"].str.fullmatch(r"\d{5}")
    if mal_ubicados.any():
        log.warning("[%s] %s filas con ubica_geo inválido; se descartan.",
                    anio, int(mal_ubicados.sum()))
        reporte["descartadas_ubica_geo_invalido"] = int(mal_ubicados.sum())
        hogares = hogares.loc[~mal_ubicados].copy()

    # --- Numéricos ---
    numericas = [c for c in COLS_HOGAR + COLS_INGRESO + COLS_GASTO + ["factor", "smg"]
                 if c in hogares.columns]
    for columna in numericas:
        hogares[columna] = _a_numero_si_aplica(hogares[columna], columna)

    # --- Validaciones de diseño muestral ---
    factor_invalido = hogares["factor"].isna() | (hogares["factor"] <= 0)
    if factor_invalido.any():
        log.warning("[%s] %s filas con factor de expansión inválido; se descartan.",
                    anio, int(factor_invalido.sum()))
        reporte["descartadas_factor_invalido"] = int(factor_invalido.sum())
        hogares = hogares.loc[~factor_invalido].copy()

    integ_invalido = hogares["tot_integ"].isna() | (hogares["tot_integ"] < 1)
    if integ_invalido.any():
        log.warning("[%s] %s filas con tot_integ < 1; se descartan.",
                    anio, int(integ_invalido.sum()))
        reporte["descartadas_tot_integ_invalido"] = int(integ_invalido.sum())
        hogares = hogares.loc[~integ_invalido].copy()

    if "ing_cor" not in hogares.columns:
        raise ErrorDeDatos(f"[{anio}] concentradohogar no trae ing_cor.")

    # --- Identidad contable del ingreso corriente ---
    componentes = [c for c in cfg.componentes_ing_cor if c in hogares.columns]
    faltantes_identidad = [c for c in cfg.componentes_ing_cor if c not in hogares.columns]
    if faltantes_identidad:
        log.warning(
            "[%s] No se puede validar la identidad de ing_cor: faltan %s",
            anio, faltantes_identidad,
        )
        reporte["identidad_ing_cor"] = "no verificable"
    else:
        suma = hogares[componentes].fillna(0).sum(axis=1)
        diferencia = (hogares["ing_cor"].fillna(0) - suma).abs()
        tolerancia = 0.02
        inconsistentes = int((diferencia > tolerancia).sum())
        reporte["identidad_ing_cor"] = {
            "componentes": componentes,
            "discrepancias": inconsistentes,
            "peor_diferencia": float(diferencia.max()) if len(diferencia) else 0.0,
            "tolerancia": tolerancia,
        }
        if inconsistentes:
            log.warning(
                "[%s] ing_cor no cuadra en %s hogares (peor diferencia %.2f).",
                anio, inconsistentes, float(diferencia.max()),
            )

    # --- Columnas derivadas ---
    hogares["ing_pc"] = hogares["ing_cor"] / hogares["tot_integ"]
    hogares["ing_ae"] = hogares["ing_cor"] / _adultos_equivalentes(hogares)

    # Dependientes (menores de 12 o de 65 y más) y activos (12 a 64 años), en
    # las unidades que sí trae concentradohogar sin traslape: menores, p12_64
    # y p65mas. Se usan para la tasa de dependencia.
    hogares["dependientes_calc"] = (
        hogares["menores"].fillna(0) + hogares["p65mas"].fillna(0)
    )
    hogares["activos_calc"] = hogares["p12_64"].fillna(0)

    # Gasto en alimentos como proporción del gasto corriente monetario.
    if {"alimentos", "gasto_mon"}.issubset(hogares.columns):
        hogares["prop_gasto_alimentos"] = np.where(
            hogares["gasto_mon"] > 0, hogares["alimentos"] / hogares["gasto_mon"], np.nan
        )

    reporte["filas_finales"] = int(len(hogares))
    reporte["hogares_expandidos"] = float(hogares["factor"].sum())
    reporte["metricas_nulas"] = {
        "ing_pc": int(hogares["ing_pc"].isna().sum()),
        "ing_ae": int(hogares["ing_ae"].isna().sum()),
    }

    catalogo_geo = _catalogo_ubica_geo(tabla)
    if not catalogo_geo.empty:
        hogares = hogares.merge(catalogo_geo, on="ubica_geo", how="left")
    else:
        log.warning("[%s] Sin catálogo ubica_geo: no habrá nombres de entidad/municipio.", anio)
        hogares["desc_ent"] = pd.NA
        hogares["desc_mun"] = pd.NA

    return hogares, catalogo_geo, reporte


def _adultos_equivalentes(hogares: pd.DataFrame) -> pd.Series:
    """Escala de equivalencia declarada: 0.7 menores, 0.8 de 12-64, 1.0 de 65+.

    No es una equivalencia oficial (México no publica una canónica); es una
    elección de conveniencia, documentada en el tablero.

    Dos salvaguardas, ambas necesarias para que el indicador nunca se vuelva
    absurdo por un dato inconsistente:

    1. Si los bloques de edad no suman ``tot_integ``, los integrantes no
       clasificados se prorratean como adultos de 12 a 64 (peso 0.8), para no
       perder población.
    2. El denominador se acota a ``tot_integ``. Los ponderadores son todos
       menores o iguales a 1, así que un denominador mayor que el número de
       integrantes solo puede venir de bloques inconsistentes, y produciría un
       ingreso por adulto equivalente *menor* que el per cápita, que es
       exactamente lo contrario de lo que la escala significa.
    """
    menores = hogares["menores"].fillna(0)
    p12_64 = hogares["p12_64"].fillna(0)
    p65mas = hogares["p65mas"].fillna(0)
    total = hogares["tot_integ"].fillna(0)

    denominador = 0.7 * menores + 0.8 * p12_64 + 1.0 * p65mas
    suma_bloques = menores + p12_64 + p65mas

    # Salvaguarda 1: prorrateo de los integrantes no clasificados en los bloques.
    no_clasificados = (total - suma_bloques).clip(lower=0)
    denominador = denominador + no_clasificados * 0.8

    # Salvaguarda 2: acotar al número de integrantes.
    denominador = pd.Series(
        np.minimum(denominador.to_numpy(), total.to_numpy()), index=hogares.index
    )

    # Salvaguarda 3: nunca devolver 0 ni NaN en el denominador.
    return denominador.where(denominador > 0, total.where(total > 0, np.nan))


# --------------------------------------------------------------------------
# Indicadores de bienestar
# --------------------------------------------------------------------------


@dataclass
class ResultadoBienestar:
    """Indicadores por hogar e inventario de lo que sí se pudo construir."""

    dataframe: pd.DataFrame
    construidos: list[str] = field(default_factory=list)
    no_disponibles: dict[str, str] = field(default_factory=dict)


def agregar_bienestar(
    anio: int,
    tablas: dict[str, TablaExtraida],
    hogares: pd.DataFrame,
) -> ResultadoBienestar:
    """Agrega los indicadores de bienestar a nivel hogar.

    Cada indicador faltante se registra en ``no_disponibles`` con su motivo; un
    año al que le falte una tabla no rompe el pipeline, solo deja ese indicador
    como no disponible.
    """
    llaves = ["folioviv", "foliohog"]
    salida = hogares[llaves + ["tot_integ"]].copy()
    construidos: list[str] = []
    no_disponibles: dict[str, str] = {}

    # ---- poblacion: agregación de personas a hogar ----
    if "poblacion" in tablas:
        pob = _cargar_tabla_ligera(tablas["poblacion"])

        # Población efectiva del hogar según el archivo de personas. Se usa para
        # ponderar indicadores de personas; no se compara con tot_integ porque
        # el archivo de personas puede excluir a algún integrante.
        pob["_uno"] = 1
        salida = salida.merge(
            pob.groupby(llaves, as_index=False)["_uno"]
            .sum()
            .rename(columns={"_uno": "pob_personas"}),
            on=llaves,
            how="left",
        )

        for nombre, (columna, _descripcion, afirmativo) in INDICADORES_POBLACION.items():
            if columna not in pob.columns:
                no_disponibles[nombre] = f"poblacion.{columna} no existe en {anio}"
                continue
            serie = _a_numero(pob[columna])
            crudo = pob[columna].astype("string").str.strip()
            vacio = crudo.isna() | crudo.eq("") | crudo.isin(VALORES_NULOS)
            # "No sabe" / "no aplica" no cuentan como respuesta afirmativa.
            no_aplica = serie.isin(VALORES_NO_APLICA)

            if afirmativo is None:
                # Variable de grado (p. ej. disc_ver, catálogo [&,1-4]): 1..N son
                # la condición y cualquier otro código válido es "sin la
                # condición". Solo las celdas realmente vacías salen del
                # denominador.
                valido = ~vacio
                conteo = ((serie.fillna(0) > 0) & valido & ~no_aplica).astype(int)
            else:
                # Variable binaria (catálogo ``si_no``, 1 = Sí / 2 = No): la clave
                # afirmativa es "sí" y **cualquier otra respuesta válida es "no"**,
                # incluidos códigos no numéricos como "&". Descartarlos sesgaría el
                # denominador y el porcentaje se calcularía sobre una base menor.
                valido = ~vacio
                conteo = ((serie == afirmativo) & valido).astype(int)

            agg = pd.DataFrame(
                {**{k: pob[k].to_numpy() for k in llaves},
                 "_c": conteo.to_numpy(), "_d": valido.astype(int).to_numpy()}
            )
            agrupado = agg.groupby(llaves, as_index=False)[["_c", "_d"]].sum()
            agrupado = agrupado.rename(
                columns={"_c": f"{nombre}_num", "_d": f"{nombre}_den"}
            )
            salida = salida.merge(agrupado, on=llaves, how="left")
            construidos.append(f"{nombre}_num")

        # Indicadores derivados a nivel hogar.
        derivados = {
            "pct_asis_esc": "asis_esc",
            "pct_alfabeta": "alfabetism",
            "pct_segsoc": "segsoc",
            "pct_etnia": "etnia",
            "pct_afro": "afrod",
            "pct_discapacidad": "disc_ver",
            "pct_atencion_salud": "aten_sal",
        }
        for nombre_salida, base in derivados.items():
            col_den = f"{base}_den"
            if col_den in salida.columns:
                salida[nombre_salida] = _proporcion(
                    salida[f"{base}_num"], salida[col_den]
                )
                construidos.append(nombre_salida)
    else:
        no_disponibles["poblacion"] = f"tabla poblacion ausente en {anio}"

    # ---- hogares: variables ya por hogar ----
    if "hogares" in tablas:
        hg = _cargar_tabla_ligera(tablas["hogares"])
        columnas = list(INDICADORES_HOGARES)
        presentes = [c for c in columnas if c in hg.columns]
        for faltante in set(columnas) - set(presentes):
            no_disponibles[faltante] = f"hogares.{faltante} no existe en {anio}"
        if presentes:
            sub = hg[llaves + presentes].drop_duplicates(llaves).copy()
            for columna in presentes:
                sub[columna] = _a_numero(sub[columna])
            salida = salida.merge(sub, on=llaves, how="left")
            construidos.extend(presentes)

        # Dificultad alimentaria reportada por el hogar.
        #
        # Los reactivos ``acc_alim*`` son binarios con el catálogo ``si_no`` de
        # INEGI (1 = Sí, 2 = No), y los que solo aplican a hogares con menores
        # quedan vacíos en el resto. Aquí se construyen dos medidas escalonadas:
        #
        # * ``dificultad_alimentaria``: el hogar reportó **al menos una** de las
        #   16 situaciones. Es una medida **amplia**: el reactivo 1 ("preocupación
        #   de que la comida se acabe"), por sí solo, lo reporta ~30 % de los
        #   hogares, así que el indicador ronda el 95 % y NO equivale a la
        #   medición oficial de inseguridad alimentaria del INEGI.
        # * ``inseg_alimentaria_severa``: el hogar reportó **hambre y no comió**
        #   o **una o menos comidas al día** (reactivos 7, 8, 14 y 16, que son los
        #   que INEGI asocia a la escala más grave). Es la medida que conviene
        #   citar.
        #
        # La medición oficial completa depende de una escala de puntos y de
        # cortes que INEGI/CONEVAL no publican en los microdatos abiertos; no se
        # replica aquí y el tablero lo dice.
        acc = [c for c in COLS_ACC_ALIM if c in hg.columns]
        if acc:
            valores = pd.DataFrame({c: _a_numero(hg[c]) for c in acc})
            sin_dato = valores.isna().all(axis=1)
            alguna = ((valores == 1).any(axis=1) & ~sin_dato).astype(int)

            severos = [c for c in ("acc_alim7", "acc_alim8", "acc_alim14", "acc_alim16")
                       if c in valores.columns]
            if severos:
                severa = ((valores[severos] == 1).any(axis=1) & ~sin_dato).astype(int)
            else:
                severa = pd.Series(np.nan, index=valores.index)

            sub = pd.DataFrame(
                {**{k: hg[k].to_numpy() for k in llaves},
                 "dificultad_alimentaria": alguna.to_numpy(),
                 "inseg_alimentaria_severa": severa.to_numpy()}
            )
            sub = sub.groupby(llaves, as_index=False).max()
            salida = salida.merge(sub, on=llaves, how="left")
            construidos.extend(["dificultad_alimentaria", "inseg_alimentaria_severa"])

            # Alias histórico: versiones anteriores llamaban así a la medida
            # amplia. Se conserva para no romper consumidores, con el nombre
            # correcto al lado.
            salida["inseg_alimentaria"] = salida["dificultad_alimentaria"]
        else:
            no_disponibles["dificultad_alimentaria"] = f"reactivos acc_alim* ausentes en {anio}"
    else:
        no_disponibles["hogares"] = f"tabla hogares ausente en {anio}"

    # ---- viviendas ----
    if "viviendas" in tablas:
        vv = _cargar_tabla_ligera(tablas["viviendas"])
        if "foliohog" not in vv.columns:
            # Una fila por vivienda: se asigna al hogar principal.
            vv["foliohog"] = "1"
        columnas = list(INDICADORES_VIVIENDAS)
        presentes = [c for c in columnas if c in vv.columns]
        for faltante in set(columnas) - set(presentes):
            no_disponibles[faltante] = f"viviendas.{faltante} no existe en {anio}"
        if presentes:
            sub = vv[llaves + presentes].drop_duplicates(llaves).copy()
            for columna in presentes:
                sub[columna] = _a_numero(sub[columna])
            salida = salida.merge(sub, on=llaves, how="left")
            construidos.extend(presentes)
    else:
        no_disponibles["viviendas"] = f"tabla viviendas ausente en {anio}"

    # ---- indicadores derivados que cruzan tablas ----
    # Hacinamiento: integrantes por cuarto usado para dormir.
    if "cuart_dorm" in salida.columns:
        dorm = salida["cuart_dorm"].replace(0, np.nan)
        salida["hacinamiento"] = salida["tot_integ"] / dorm
        construidos.append("hacinamiento")

    return ResultadoBienestar(
        dataframe=salida, construidos=sorted(set(construidos)),
        no_disponibles=no_disponibles,
    )


def _proporcion(numerador: pd.Series, denominador: pd.Series) -> pd.Series:
    """Proporción segura: ``NaN`` cuando el denominador es cero o nulo."""
    den = pd.to_numeric(denominador, errors="coerce")
    num = pd.to_numeric(numerador, errors="coerce")
    return np.where((den > 0), num / den, np.nan)


def _cargar_tabla_ligera(tabla: TablaExtraida) -> pd.DataFrame:
    """Lee una tabla de personas/hogares/viviendas como texto, sin coercionar."""
    columnas_necesarias = (
        list(INDICADORES_POBLACION) + list(INDICADORES_HOGARES)
        + list(INDICADORES_VIVIENDAS) + COLS_ACC_ALIM
        + ["folioviv", "foliohog"]
    )
    muestra = pd.read_csv(
        tabla.csv_datos, nrows=5, dtype=str,
        encoding=detectar_codificacion(tabla.csv_datos),
    )
    usar = [c for c in muestra.columns if c in columnas_necesarias]
    return pd.read_csv(
        tabla.csv_datos,
        usecols=usar,
        dtype="string",
        encoding=detectar_codificacion(tabla.csv_datos),
        na_values=VALORES_NULOS,
        low_memory=False,
    )


# --------------------------------------------------------------------------
# Producto final
# --------------------------------------------------------------------------


def construir_hogares(
    anio: int,
    *,
    cfg: config.Configuracion | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Pipeline de limpieza completo de un año listo para guardar en Parquet."""
    from .extraccion import extraer_anio

    cfg = cfg or config.cargar_configuracion()
    tablas = extraer_anio(anio, cfg=cfg)

    hogares, catalogo_geo, reporte = cargar_concentradohogar(tablas["concentradohogar"], cfg=cfg)

    bienestar = agregar_bienestar(anio, tablas, hogares)
    # Solo se pegan las columnas nuevas: el marco de bienestar trae las llaves y
    # alguna columna auxiliar que ya existe en hogares, y pegarlas duplicadas
    # crearía sufijos _x/_y que romperían la lectura de columnas.
    nuevas = [
        c for c in bienestar.dataframe.columns
        if c not in ("folioviv", "foliohog") and c not in hogares.columns
    ]
    if nuevas:
        hogares = hogares.merge(
            bienestar.dataframe[["folioviv", "foliohog", *nuevas]],
            on=["folioviv", "foliohog"],
            how="left",
        )
    reporte["bienestar_construidos"] = len(bienestar.construidos)
    reporte["bienestar_columnas"] = nuevas
    reporte["bienestar_no_disponibles"] = bienestar.no_disponibles
    reporte["tablas_extraidas"] = sorted(tablas)

    # Deflactación: se agregan columnas reales solo si el año está en la tabla.
    factor = cfg.deflactor(anio)
    if factor is not None:
        base = cfg.anio_base_deflactor
        hogares[f"ing_pc_real_{base}"] = hogares["ing_pc"] * factor
        hogares[f"ing_ae_real_{base}"] = hogares["ing_ae"] * factor
        hogares[f"ing_cor_real_{base}"] = hogares["ing_cor"] * factor
        hogares[f"gasto_mon_real_{base}"] = hogares["gasto_mon"] * factor
        reporte["deflactor"] = {
            "anio_base": base,
            "factor": factor,
            "columnas": [c for c in hogares.columns if f"real_{base}" in c],
        }
    else:
        reporte["deflactor"] = {
            "anio_base": cfg.anio_base_deflactor,
            "factor": None,
            "motivo": f"El año {anio} no está en config/deflactores.csv",
        }

    # Proporción del ingreso por fuente (share), útil para el tablero.
    if "ing_cor" in hogares.columns:
        denominador = hogares["ing_cor"].where(hogares["ing_cor"] > 0, np.nan)
        for fuente in ("ingtrab", "negocio", "transfer", "remesas", "bene_gob",
                       "jubilacion", "rentas", "estim_alqu"):
            if fuente in hogares.columns:
                hogares[f"share_{fuente}"] = _proporcion(hogares[fuente], denominador)

    return hogares, reporte


def guardar_hogares(hogares: pd.DataFrame, anio: int) -> Path:
    """Escribe el Parquet curado de un año."""
    ruta = config.ruta_hogares(anio)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    hogares.to_parquet(ruta, index=False, compression="zstd")
    log.info("[%s] Guardado %s (%s filas, %.1f KB)", anio, ruta,
             len(hogares), ruta.stat().st_size / 1024)
    return ruta


def cargar_hogares(anio: int) -> pd.DataFrame:
    """Carga el Parquet curado de un año."""
    ruta = config.ruta_hogares(anio)
    if not ruta.exists():
        raise FileNotFoundError(
            f"No hay datos procesados para {anio}: falta {ruta}. "
            f"Ejecute: python -m enigh.pipeline --anios {anio}"
        )
    return pd.read_parquet(ruta)
