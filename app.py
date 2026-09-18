"""Tablero Streamlit: el ingreso de las familias en Sonora por decil.

Ejecución::

    streamlit run app.py

El tablero consume exclusivamente los datos procesados por ``enigh.pipeline``
(``data/processed``). No descarga nada en tiempo de ejecución: si faltan datos,
lo dice con instrucciones concretas en lugar de fallar.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

log = logging.getLogger("enigh.tablero")

# El paquete ``enigh`` vive junto a este archivo, así que la raíz del proyecto
# debe estar en el path de importación. Sin esto, el tablero solo funciona si se
# lanza desde la raíz del proyecto: al ejecutarlo por ruta absoluta (por ejemplo
# desde otra carpeta o desde las pruebas con AppTest), ``import enigh`` falla con
# un error confuso de "unknown location".
_RAIZ_PROYECTO = Path(__file__).resolve().parent
if str(_RAIZ_PROYECTO) not in sys.path:
    sys.path.insert(0, str(_RAIZ_PROYECTO))

# Las importaciones de ``enigh`` van aquí a propósito, después de ajustar el
# path: es la única forma de que el paquete sea importable sin instalarlo.
from enigh import CLAVE_SONORA, NOMBRE_SONORA, UNIDAD_INGRESO, bienestar, config  # noqa: E402
from enigh import geografia, graficos, limpieza, metricas  # noqa: E402
from enigh.metricas import asignar_deciles, resumen_deciles  # noqa: E402

st.set_page_config(
    page_title=f"ENIGH · Ingreso de las familias en {NOMBRE_SONORA}",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

CFG = config.cargar_configuracion()

# --- Identidad del proyecto y autoría ---------------------------------------
# Estos textos se definen una sola vez para que las pruebas puedan afirmarlos y
# para que el crédito no se desincronice entre las vistas.

PROGRAMA_MCD = "Maestría en Ciencia de Datos"
SITIO_MCD = "https://mcd.unison.mx/"

CREDITO = (
    "Desarrollado por Julio Waissman (julio.waissman@unison.mx) "
    "con DeepSeek Harness y Streamlit."
)

# Emblema de la MCD. Se versiona en ``assets/`` (19 KB) en lugar de descargarse
# en el arranque: así el tablero sigue funcionando sin red y sin depender de que
# el sitio de la MCD mantenga el archivo disponible.
RUTA_LOGO_MCD = _RAIZ_PROYECTO / "assets" / "logo_mcd.png"
LOGO_MCD_DISPONIBLE = RUTA_LOGO_MCD.is_file()

ETIQUETAS_METRICA = {
    "ing_pc": "Ingreso corriente per cápita",
    "ing_ae": "Ingreso corriente por adulto equivalente",
}

ETIQUETAS_CORTES = {
    "nacionales": "Deciles nacionales (mismos cortes para todo el país)",
    "propios": f"Deciles propios de {NOMBRE_SONORA}",
}

NOTA_2020 = (
    "El levantamiento de 2020 se hizo en condiciones atípicas por la pandemia de "
    "COVID-19. Léase como un año de referencia excepcional, no como un punto "
    "comparable en igualdad de circunstancias."
)


# --------------------------------------------------------------------------
# Carga de datos (cacheada)
# --------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def anios_disponibles() -> list[int]:
    """Años con microdatos ya procesados en disco."""
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


@st.cache_data(show_spinner=False)
def cargar_hogares(anio: int) -> pd.DataFrame:
    return limpieza.cargar_hogares(anio)


@st.cache_data(show_spinner=False)
def cargar_agregado(anio: int, nombre: str) -> pd.DataFrame:
    ruta = config.ruta_agregados(anio, nombre)
    if not ruta.exists():
        return pd.DataFrame()
    return pd.read_parquet(ruta)


@st.cache_data(show_spinner=False)
def cargar_serie(nombre: str) -> pd.DataFrame:
    ruta = config.DIR_PROCESSED / "agregados" / f"{nombre}.parquet"
    if not ruta.exists():
        return pd.DataFrame()
    return pd.read_parquet(ruta)


@st.cache_data(show_spinner=False)
def cargar_bienestar(anio: int) -> pd.DataFrame:
    """Perfil de bienestar por decil. Se recalcula si no está precomputado."""
    ruta = config.DIR_PROCESSED / "agregados" / f"ano={anio}" / "bienestar_sonora.parquet"
    if ruta.exists():
        return pd.read_parquet(ruta)
    # Recálculo bajo demanda: los indicadores son baratos sin bootstrap.
    ruta_hogares = config.ruta_hogares(anio)
    if not ruta_hogares.exists():
        return pd.DataFrame()
    hogares = cargar_hogares(anio)
    sonora = hogares[hogares["entidad"] == CLAVE_SONORA].copy()
    if sonora.empty:
        return pd.DataFrame()
    perfil, _omitidos, _avisos = bienestar.perfil_por_decil(
        sonora, grupo="decil_nacional", replicas=0
    )
    return perfil


@st.cache_data(show_spinner=False)
def cargar_manifiesto() -> dict:
    if not config.ARCHIVO_MANIFIESTO.exists():
        return {}
    with config.ARCHIVO_MANIFIESTO.open(encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data(show_spinner=False)
def cargar_geojson_estatal() -> dict | None:
    return geografia.cargar_geojson(geografia.NOMBRE_ENTIDADES)


@st.cache_data(show_spinner=False)
def cargar_geojson_municipal() -> dict | None:
    return geografia.cargar_geojson(geografia.NOMBRE_MUNICIPIOS_SONORA)


# --------------------------------------------------------------------------
# Recálculo de deciles bajo demanda
# --------------------------------------------------------------------------


def preparar_sonora(
    hogares: pd.DataFrame,
    *,
    metrica: str,
    cortes: str,
) -> tuple[pd.DataFrame, "resumen_deciles"]:
    """Filtra Sonora y asigna los deciles según la métrica y los cortes elegidos."""
    sonora = hogares[hogares["entidad"] == CLAVE_SONORA].copy()
    if metrica not in sonora.columns:
        raise KeyError(f"La métrica {metrica!r} no está en los microdatos de {NOMBRE_SONORA}.")

    valores = sonora[metrica].to_numpy(dtype=float)
    pesos = sonora["factor"].to_numpy(dtype=float)

    if cortes == "propios":
        resumen = resumen_deciles(
            sonora, columna_ingreso=metrica, columna_poblacion="tot_integ",
            nombre_metrica=metrica,
        )
        sonora["decil"] = asignar_deciles(valores, pesos, resumen.cortes)
    else:
        # Cortes nacionales: se recalculan sobre todo el país con la métrica activa.
        resumen_nac = resumen_deciles(
            hogares, columna_ingreso=metrica, columna_poblacion="tot_integ",
            nombre_metrica=metrica,
        )
        sonora["decil"] = asignar_deciles(valores, pesos, resumen_nac.cortes)
        resumen = resumen_deciles(
            sonora, columna_ingreso=metrica, cortes=resumen_nac.cortes,
            columna_poblacion="tot_integ", nombre_metrica=metrica,
        )

    return sonora, resumen


def aplicar_deflactor(hogares: pd.DataFrame, anio: int, metrica: str, expresion: str) -> tuple[pd.DataFrame, str, str]:
    """Agrega la columna de la métrica ya expresada en la unidad elegida.

    Returns:
        ``(datos, nombre_columna, etiqueta_unidad)``.
    """
    datos = hogares.copy()
    if expresion == "real":
        factor = CFG.deflactor(anio)
        if factor is None:
            return datos, metrica, UNIDAD_INGRESO
        columna = f"{metrica}_realbase"
        datos[columna] = datos[metrica] * factor
        return datos, columna, f"pesos constantes de {CFG.anio_base_deflactor} por trimestre"
    if expresion == "smg" and "smg" in datos.columns:
        columna = f"{metrica}_smg"
        datos[columna] = datos[metrica] / datos["smg"].replace(0, np.nan)
        return datos, columna, "veces el salario mínimo general trimestral"
    return datos, metrica, UNIDAD_INGRESO


# --------------------------------------------------------------------------
# Branding (Maestría en Ciencia de Datos)
# --------------------------------------------------------------------------


def mostrar_logo_mcd(destino, *, ancho: int) -> bool:
    """Muestra el emblema de la MCD si está disponible.

    Devuelve ``True`` si dibujó la imagen. Nunca lanza: un recurso ausente no debe
    tumbar el tablero completo, porque ``st.image`` levanta ``RuntimeError``
    cuando el archivo no existe y el branding es lo menos importante de la página.
    """
    if not LOGO_MCD_DISPONIBLE:
        log.warning(
            "No se encontró el emblema en %s; se muestra solo el texto.",
            RUTA_LOGO_MCD,
        )
        return False
    try:
        destino.image(str(RUTA_LOGO_MCD), width=ancho)
        return True
    except (RuntimeError, OSError) as exc:  # pragma: no cover - defensivo
        log.warning("No se pudo mostrar el emblema de la MCD: %s", exc)
        return False


def creditos(*, compacto: bool = False) -> None:
    """Identidad del programa y autoría.

    ``compacto`` usa tipografía pequeña, para la barra lateral; en la vista de
    metodología se muestra en tamaño normal porque ahí sí es contenido.
    """
    etiqueta = st.caption if compacto else st.markdown
    etiqueta(f"**{PROGRAMA_MCD}** · [UNISON]({SITIO_MCD})")
    etiqueta(CREDITO)


# --------------------------------------------------------------------------
# Barra lateral
# --------------------------------------------------------------------------


def barra_lateral(anios: list[int]) -> dict:
    """Controles globales del tablero."""
    # Identidad del programa: emblema y texto siempre visibles, para que el
    # contexto académico del tablero quede claro en cualquier vista.
    mostrar_logo_mcd(st.sidebar, ancho=110)
    st.sidebar.caption(f"**{PROGRAMA_MCD}** · [UNISON]({SITIO_MCD})")
    st.sidebar.divider()

    st.sidebar.title("ENIGH · Sonora")
    st.sidebar.caption(
        "Encuesta Nacional de Ingresos y Gastos de los Hogares, INEGI."
    )

    pagina = st.sidebar.radio(
        "Vista",
        [
            "Panorama",
            "Deciles",
            "Mapas",
            "Serie de tiempo",
            "Bienestar",
            "Metodología y calidad",
        ],
        index=0,
    )

    st.sidebar.divider()
    anio = st.sidebar.selectbox(
        "Año del levantamiento", options=list(reversed(anios)), index=0,
        help="Años con microdatos ya descargados y procesados localmente.",
    )
    metrica = st.sidebar.radio(
        "Métrica que ordena los deciles",
        options=list(ETIQUETAS_METRICA),
        format_func=lambda k: ETIQUETAS_METRICA[k],
        index=0,
        help=(
            "Per cápita divide el ingreso entre todos los integrantes. "
            "Adulto equivalente pondera por edad (0.7 menores, 0.8 de 12 a 64 "
            "años, 1.0 de 65 y más), y es más sensible al tamaño del hogar."
        ),
    )

    opciones_expresion = ["nominal", "real"]
    if (CFG.deflactor(anio) is None) and "real" in opciones_expresion:
        opciones_expresion.remove("real")

    expresion = st.sidebar.radio(
        "Expresión monetaria",
        options=opciones_expresion,
        format_func=lambda k: {
            "nominal": f"Pesos corrientes de {anio}",
            "real": f"Pesos constantes de {CFG.anio_base_deflactor} (INPC)",
        }[k],
        index=0,
    )

    cortes = st.sidebar.radio(
        "Definición de los deciles",
        options=list(ETIQUETAS_CORTES),
        format_func=lambda k: ETIQUETAS_CORTES[k],
        index=0,
        help=(
            "Con cortes nacionales, 'decil 1' es el 10 % más pobre de México, "
            "así que Sonora puede no tener hogares en algún decil extremo. "
            "Con deciles propios, se reparte la población sonorense en diez partes."
        ),
    )

    st.sidebar.divider()
    if anio == 2020:
        st.sidebar.warning(NOTA_2020)
    st.sidebar.caption(
        f"Años procesados: {', '.join(str(a) for a in anios)}. "
        "Los montos de la ENIGH son trimestrales."
    )

    # Autoría, presente en todas las vistas.
    st.sidebar.divider()
    st.sidebar.caption(CREDITO)

    return {
        "pagina": pagina, "anio": anio, "metrica": metrica,
        "expresion": expresion, "cortes": cortes,
    }


# --------------------------------------------------------------------------
# Helpers de presentación
# --------------------------------------------------------------------------


def formato_moneda(valor: float) -> str:
    if valor is None or not np.isfinite(valor):
        return "s/d"
    return f"${valor:,.0f}"


def formato_pct(valor: float, decimales: int = 1) -> str:
    if valor is None or not np.isfinite(valor):
        return "s/d"
    return f"{valor:.{decimales}f} %"


def aviso_muestra(n_hogares: int, n_upm: int, etiqueta: str) -> None:
    """Aviso proporcional al tamaño de muestra de la celda."""
    if n_hogares == 0:
        st.warning(f"{etiqueta}: muestra insuficiente.")
    elif n_hogares < 60 or n_upm < 20:
        st.info(
            f"{etiqueta}: {n_hogares:,} hogares en {n_upm:,} UPM. "
            "Muestra pequeña; interprete con cautela."
        )


def tabla_descargable(tabla: pd.DataFrame, nombre: str, etiqueta: str) -> None:
    st.dataframe(tabla, use_container_width=True, hide_index=True)
    st.download_button(
        f"Descargar {etiqueta} (CSV)",
        data=tabla.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{nombre}.csv",
        mime="text/csv",
    )


# --------------------------------------------------------------------------
# Vistas
# --------------------------------------------------------------------------


def vista_panorama(estado: dict, datos: dict) -> None:
    sonora = datos["sonora"]
    resumen = datos["resumen"]
    nacional = datos["nacional"]

    st.title(f"¿Cómo está el ingreso de las familias en {NOMBRE_SONORA}?")
    st.markdown(
        f"**{estado['anio']}** · {ETIQUETAS_METRICA[estado['metrica']]} · "
        f"{datos['unidad']} · {ETIQUETAS_CORTES[estado['cortes']]}"
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        ETIQUETAS_METRICA[estado["metrica"]],
        formato_moneda(resumen.tabla[datos["columna"]].mean()),
        help="Promedio ponderado de los hogares de Sonora en el año seleccionado.",
    )
    col2.metric(
        "Hogares en Sonora",
        f"{sonora['factor'].sum():,.0f}",
        help=f"Expandidos desde {len(sonora):,} hogares en la muestra.",
    )
    col3.metric(
        "Gini",
        f"{resumen.gini:.3f}",
        help="0 = igualdad perfecta, 1 = desigualdad total. Ponderado.",
    )
    col4.metric(
        "Razón de Palma",
        f"{resumen.palma:.2f}",
        help="Cuántas veces el ingreso del 10 % más rico supera al del 40 % más pobre.",
    )

    aviso_muestra(len(sonora), sonora["upm"].nunique(), "Sonora")

    columna_metrica = datos["columna"]
    mediana_sonora = resumen.tabla[f"{columna_metrica}_mediana"].median()
    st.subheader("Los tres datos que resumen la situación")
    izq, der = st.columns(2)
    with izq:
        st.markdown(
            f"""
- El **40 % más pobre** de Sonora concentra el
  **{formato_pct(resumen.participacion_bajo40)}** del ingreso corriente total.
- El **10 % más rico** concentra el
  **{formato_pct(resumen.participacion_alto10)}**.
- La brecha entre el decil más rico y el más pobre es de
  **{datos['brecha_d10_d1']:.1f} veces**.
"""
        )
    with der:
        st.markdown(
            f"""
- Mediana del ingreso: **{formato_moneda(mediana_sonora)}**
  (la mitad de los hogares está por debajo de este monto).
- Sonora se ubica en el lugar **{datos['lugar_sonora']} de 32** entidades por
  ingreso per cápita.
- Comparado con el promedio nacional
  ({formato_moneda(nacional['ing_pc'])}), Sonora está
  **{datos['vs_nacional_pct']:+.1f} %**.
"""
        )

    st.divider()
    st.subheader(f"Ingreso per cápita por decil en {NOMBRE_SONORA}")
    izquierda, derecha = st.columns([3, 2])
    with izquierda:
        st.plotly_chart(
            graficos.barras_deciles(
                resumen.tabla,
                columna=datos["columna"],
                titulo=f"Ingreso per cápita por decil · {estado['anio']}",
                unidad=datos["unidad"],
            ),
            use_container_width=True,
        )
    with derecha:
        st.plotly_chart(
            graficos.curva_lorenz(
                sonora[datos["columna"]], sonora["factor"],
                titulo="Curva de Lorenz · Sonora",
            ),
            use_container_width=True,
        )

    st.caption(
        "Cada barra es el promedio ponderado de los hogares de Sonora que caen en "
        "ese decil. Los bigotes son intervalos de confianza al 95 % calculados por "
        "bootstrap respetando el diseño muestral (UPM dentro de estrato)."
    )


def vista_deciles(estado: dict, datos: dict) -> None:
    st.title("Deciles: el detalle")
    resumen = datos["resumen"]
    sonora = datos["sonora"]
    columna = datos["columna"]

    tabla = resumen.tabla.copy()
    tabla = tabla.rename(
        columns={
            "decil": "Decil", "n_hogares": "Hogares en muestra",
            "hogares_expandidos": "Hogares expandidos",
            "ing_pc": "Ingreso per cápita",
            "ing_pc_mediana": "Mediana del decil",
            "ing_cor": "Ingreso corriente del hogar",
            "participacion_ingreso": "Participación del ingreso (%)",
            "ing_pc_ic_inferior": "IC 95 % inferior",
            "ing_pc_ic_superior": "IC 95 % superior",
            "ing_pc_cv": "Coeficiente de variación",
            "ing_pc_fiable": "Estimación fiable",
        }
    )
    columnas = [
        c for c in [
            "Decil", "Hogares en muestra", "Hogares expandidos",
            "Ingreso per cápita", "IC 95 % inferior", "IC 95 % superior",
            "Mediana del decil", "Ingreso corriente del hogar",
            "Participación del ingreso (%)", "Coeficiente de variación",
            "Estimación fiable",
        ] if c in tabla.columns
    ]
    tabla_visible = tabla[columnas]

    st.subheader("Tabla por decil")
    st.caption(
        "El ingreso se expresa en " + datos["unidad"] + ". "
        "'Estimación fiable' es falso cuando el coeficiente de variación supera "
        f"{CFG.umbrales.cv_no_fiable:.0%}, señal de muestra insuficiente."
    )
    tabla_descargable(tabla_visible, f"deciles_sonora_{estado['anio']}", "la tabla")

    st.divider()
    st.subheader("Sonora frente al país, decil por decil")
    nacional = datos["nacional"]
    st.plotly_chart(
        graficos.barras_comparadas(
            {
                NOMBRE_SONORA: _reindexar_deciles(resumen.tabla, columna),
                "Nacional": _reindexar_deciles(nacional["deciles"], columna),
            },
            columna=columna,
            titulo=f"Ingreso per cápita por decil · {estado['anio']}",
            unidad=datos["unidad"],
            colores={NOMBRE_SONORA: graficos.COLOR_SONORA, "Nacional": graficos.COLOR_NACIONAL},
        ),
        use_container_width=True,
    )

    st.divider()
    st.subheader("Composición del ingreso por decil")
    fuentes = [c for c in sonora.columns if c.startswith("share_")]
    if fuentes:
        agrupado = (
            sonora.groupby("decil", dropna=False)
            .apply(
                lambda sub: pd.Series(
                    {
                        f"share_{f.split('share_')[1]}": metricas.media_ponderada(
                            sub[f], sub["factor"]
                        )
                        for f in fuentes
                    }
                ),
                include_groups=False,
            )
            .reset_index()
            .rename(columns={"decil": "decil"})
        )
        agrupado = agrupado.rename(columns={"index": "decil"})
        st.plotly_chart(
            graficos.barras_composicion(agrupado), use_container_width=True
        )
    else:
        st.info("Los microdatos no traen el desglose de fuentes de ingreso.")

    st.divider()
    st.subheader("¿Los deciles son distintos entre sí?")
    st.caption(
        "Si los intervalos de dos deciles se traslapan, la diferencia entre ellos "
        "no es estadísticamente distinguible con esta muestra."
    )
    if "ing_pc_ic_inferior" in resumen.tabla.columns:
        traslapes = _contar_traslapes(resumen.tabla, "ing_pc")
        st.metric(
            "Pares de deciles con intervalos traslapados",
            f"{traslapes} de 45",
            help="Un número alto indica que la muestra no distingue bien deciles vecinos.",
        )


def vista_mapas(estado: dict, datos: dict) -> None:
    st.title("Dónde: mapas de Sonora y del país")

    entidades = datos["entidades"]
    geojson = cargar_geojson_estatal()

    st.subheader("Sonora en el contexto de las 32 entidades")
    if entidades.empty:
        st.info("Sin agregados por entidad. Ejecute el pipeline para generarlos.")
    else:
        columna_mapa = "ing_pc"
        izquierda, derecha = st.columns([3, 2])
        with izquierda:
            st.plotly_chart(
                graficos.mapa_entidades(
                    entidades, geojson or {},
                    columna=columna_mapa,
                    titulo=f"Ingreso per cápita por entidad · {estado['anio']}",
                    unidad=UNIDAD_INGRESO,
                ),
                use_container_width=True,
            )
        with derecha:
            st.plotly_chart(
                graficos.puntos_entidades(
                    entidades, columna=columna_mapa,
                    titulo="Todas las entidades, ordenadas",
                ),
                use_container_width=True,
            )

    st.divider()
    st.subheader("Municipios de Sonora")
    municipal = datos["municipal"]
    geojson_mun = cargar_geojson_municipal()

    if municipal.empty:
        st.info("Sin desglose municipal para este año.")
        return

    col1, col2 = st.columns([3, 2])
    with col1:
        st.plotly_chart(
            graficos.mapa_municipios(
                municipal, geojson_mun,
                columna="ing_pc",
                titulo=f"Ingreso per cápita municipal · {estado['anio']}",
                hogares_minimos=CFG.umbrales.hogares_min_municipio,
            ),
            use_container_width=True,
        )
    with col2:
        st.markdown(
            f"""
**Cómo leer este mapa**

- Solo se publican los municipios con al menos
  **{CFG.umbrales.hogares_min_municipio} hogares** en la muestra.
- Los municipios con muestra insuficiente se agrupan en
  "resto del estado" o se omiten; nunca se publica un promedio frágil.
- El tamaño de muestra de la ENIGH no permite estimaciones municipales
  precisas: Sonora tiene {len(datos['sonora']):,} hogares en la muestra para
  todo el estado.
"""
        )

    st.subheader("Tabla municipal")
    columnas = [
        c for c in ["nom_municipio", "n_hogares", "n_upm", "hogares_expandidos",
                    "ing_pc", "ing_cor", "gini", "fiable"]
        if c in municipal.columns
    ]
    if columnas:
        visible = municipal[columnas].copy()
        visible = visible.rename(columns={
            "nom_municipio": "Municipio", "n_hogares": "Hogares en muestra",
            "n_upm": "UPM", "hogares_expandidos": "Hogares expandidos",
            "ing_pc": "Ingreso per cápita", "ing_cor": "Ingreso del hogar",
            "gini": "Gini", "fiable": "Estimación fiable",
        })
        tabla_descargable(
            visible.sort_values("Ingreso per cápita", ascending=False),
            f"municipios_sonora_{estado['anio']}",
            "la tabla municipal",
        )


def vista_serie(estado: dict, datos: dict) -> None:
    st.title("¿Mejora o empeora? Serie de tiempo")
    serie = cargar_serie("serie_deciles_sonora")
    desigualdad = cargar_serie("serie_desigualdad")

    # La métrica elegida también cambia la serie: los deciles por adulto
    # equivalente agrupan a los hogares de otra manera (la escala pondera por
    # edad), así que no basta con cambiar la etiqueta del eje.
    if estado["metrica"] == "ing_ae":
        serie_ae = cargar_serie("serie_deciles_sonora_ae")
        if not serie_ae.empty:
            serie = serie_ae
            st.caption(
                "Deciles formados con el **ingreso por adulto equivalente**. "
                "Cambia la métrica en la barra lateral para ver los deciles "
                "ordenados por ingreso per cápita."
            )

    if serie.empty:
        st.info(
            "Se necesitan al menos dos años procesados para construir la serie. "
            "Ejecute: `python -m enigh.pipeline`"
        )
        return

    anios_serie = sorted(serie["anio"].unique())
    st.markdown(
        f"Levantamientos disponibles: **{', '.join(str(a) for a in anios_serie)}**. "
        "La ENIGH no es anual: los puntos se muestran en un eje categórico a "
        "propósito, sin interpolar, porque una línea suave entre 2020 y 2024 "
        "sugeriría una tendencia continua que los datos no sostienen."
    )
    if 2020 in anios_serie:
        st.warning(NOTA_2020)

    # La serie se guarda en pesos corrientes de cada levantamiento. Se deflacta
    # aquí si el usuario pidió pesos constantes, en lugar de ignorar su elección:
    # comparar 2020 con 2024 en pesos corrientes exagera el crecimiento, porque
    # los precios subieron ~26 % en el periodo.
    serie = _serie_en_pesos_elegidos(serie, estado["expresion"])
    con_ic = estado["expresion"] == "nominal"
    # La serie siempre trae ``ing_pc``: es el ingreso per cápita de los hogares de
    # cada decil. Cuando el usuario elige adulto equivalente, lo que cambia es la
    # *agrupación* de los deciles (se carga la serie _ae), no el nombre de la
    # columna.
    columna = "ing_pc"
    unidad_serie = (
        UNIDAD_INGRESO
        if estado["expresion"] == "nominal"
        else f"pesos constantes de {CFG.anio_base_deflactor} por trimestre"
    )
    if not con_ic and estado["expresion"] == "real":
        st.caption(
            f"Serie deflactada con el INPC a pesos de {CFG.anio_base_deflactor}. "
            "En esta vista no se muestran intervalos de confianza: se calcularon "
            "sobre los valores corrientes de cada año."
        )

    st.subheader("Evolución por decil")
    st.plotly_chart(
        graficos.linea_serie_deciles(
            serie, columna=columna,
            titulo=f"Ingreso per cápita por decil en {NOMBRE_SONORA}",
            unidad=unidad_serie, con_ic=con_ic,
        ),
        use_container_width=True,
    )

    st.subheader("¿Creció más el ingreso de los pobres o el de los ricos?")
    col1, col2 = st.columns([3, 2])
    with col1:
        st.plotly_chart(
            graficos.cambio_porcentual_por_decil(
                serie, columna=columna, unidad=unidad_serie,
            ),
            use_container_width=True,
        )
    with col2:
        pivote = serie.pivot_table(
            index="decil", columns="anio", values=columna, aggfunc="first"
        )
        if len(anios_serie) >= 2 and anios_serie[0] in pivote.columns:
            inicial, final = anios_serie[0], anios_serie[-1]
            if pivote[final].notna().any() and pivote[inicial].notna().any():
                cambio_bajos = (
                    pivote.loc[1:3, final].mean() / pivote.loc[1:3, inicial].mean() - 1
                ) * 100
                cambio_altos = (
                    pivote.loc[8:10, final].mean() / pivote.loc[8:10, inicial].mean() - 1
                ) * 100
                signo = "pro-pobres" if cambio_bajos > cambio_altos else "pro-ricos"
                st.markdown(
                    f"""
**Lectura {inicial} → {final}**

- Deciles bajos (D1-D3): **{cambio_bajos:+.1f} %**
- Deciles altos (D8-D10): **{cambio_altos:+.1f} %**

El crecimiento fue **{signo}**: el ingreso creció más
en los deciles {"bajos" if cambio_bajos > cambio_altos else "altos"}.
"""
                )

    st.divider()
    st.subheader("Desigualdad a lo largo del tiempo")
    if desigualdad.empty:
        st.info("Sin serie de desigualdad.")
    else:
        izquierda, derecha = st.columns(2)
        for columna_dato, titulo, eje in (
            ("gini", "Coeficiente de Gini", "índice 0-1"),
            ("palma", "Razón de Palma", "veces"),
        ):
            with (izquierda if columna_dato == "gini" else derecha):
                st.plotly_chart(
                    _linea_simple(
                        desigualdad, x="anio", y=columna_dato, color="ambito",
                        titulo=titulo, eje_y=eje,
                    ),
                    use_container_width=True,
                )
        st.caption(
            "El Gini y la razón de Palma se calculan con los ponderadores de la "
            "encuesta sobre el ingreso corriente **per cápita de todo Sonora** "
            "(no dentro de un decil). Un Gini más alto significa más desigualdad."
        )


def _serie_en_pesos_elegidos(serie: pd.DataFrame, expresion: str) -> pd.DataFrame:
    """Convierte la serie de deciles a la expresión monetaria elegida.

    ``serie_deciles_sonora.parquet`` guarda los valores **corrientes** de cada
    levantamiento. Si el usuario pidió pesos constantes se deflacta aquí con el
    INPC del año base; de lo contrario la comparación 2020 → 2024 mezclaría pesos
    de distinto poder adquisitivo y exageraría el crecimiento.
    """
    if expresion != "real" or serie.empty:
        return serie

    datos = serie.copy()
    datos["factor_deflactor"] = datos["anio"].map(
        {int(anio): CFG.deflactor(int(anio)) for anio in datos["anio"].unique()}
    )
    for columna in (c for c in ("ing_pc", "ing_cor", "ing_pc_mediana") if c in datos.columns):
        datos[columna] = datos[columna] * datos["factor_deflactor"]
    return datos.drop(columns="factor_deflactor")


def vista_bienestar(estado: dict, datos: dict) -> None:
    st.title("Bienestar más allá del ingreso")
    perfil = cargar_bienestar(estado["anio"])

    if perfil.empty:
        st.info(
            "Sin indicadores de bienestar. Ejecute el pipeline para generarlos."
        )
        return

    st.markdown(
        "Todos los indicadores se calculan por hogar y se ponderan con el factor "
        "de expansión. El valor mostrado es el del decil, con su intervalo de "
        "confianza al 95 %."
    )

    dominios = ["Todos"] + list(dict.fromkeys(perfil["dominio"]))
    dominio = st.selectbox("Dominio", options=dominios, index=0)
    visible = perfil if dominio == "Todos" else perfil[perfil["dominio"] == dominio]

    st.subheader("Panorama por decil")
    st.plotly_chart(
        graficos.heatmap_indicadores(visible),
        use_container_width=True,
    )
    st.caption(
        "Cada fila se colorea según la posición relativa del decil dentro de ese "
        "indicador (rojo = valor más bajo, azul = más alto). El número es el valor "
        "real, así que columnas con unidades distintas siguen siendo legibles."
    )

    st.divider()
    st.subheader("Detalle de un indicador")
    indicadores = sorted(visible["indicador"].unique())
    if indicadores:
        clave = st.selectbox(
            "Indicador",
            options=indicadores,
            format_func=lambda k: visible.loc[
                visible["indicador"] == k, "etiqueta"
            ].iloc[0],
        )
        sub = visible[visible["indicador"] == clave].sort_values("grupo")
        unidad = sub["unidad"].iloc[0]
        mayor_mejor = bool(sub["mayor_es_mejor"].iloc[0])
        descripcion = sub["descripcion"].iloc[0]
        if descripcion:
            st.caption(descripcion)

        izquierda, derecha = st.columns([3, 2])
        with izquierda:
            st.plotly_chart(
                _barras_indicador(
                    sub, etiqueta=sub["etiqueta"].iloc[0], unidad=unidad,
                    color=graficos.COLOR_NACIONAL if mayor_mejor else graficos.COLOR_ALERTA,
                ),
                use_container_width=True,
            )
        with derecha:
            bajo = sub[sub["grupo"].isin([1, 2])]["valor"].mean()
            alto = sub[sub["grupo"].isin([9, 10])]["valor"].mean()
            st.markdown(
                f"""
**Entre el 20 % más pobre (D1-D2) y el 20 % más rico (D9-D10)**

- D1-D2: **{_formato_indicador(bajo, unidad)}**
- D9-D10: **{_formato_indicador(alto, unidad)}**
- Brecha: **{_formato_indicador(abs(alto - bajo), unidad)}**
"""
            )
            no_fiables = int((~sub["fiable"].fillna(False)).sum())
            if no_fiables:
                st.warning(
                    f"{no_fiables} de {len(sub)} deciles tienen muestra insuficiente "
                    "para este indicador."
                )

    st.divider()
    st.subheader("Tabla completa")
    columnas = [
        c for c in ["grupo", "etiqueta", "dominio", "valor", "ic_inferior",
                    "ic_superior", "cv", "fiable", "n_hogares", "n_upm"]
        if c in perfil.columns
    ]
    tabla_descargable(
        perfil[columnas],
        f"bienestar_sonora_{estado['anio']}",
        "los indicadores",
    )


def vista_metodologia(estado: dict, datos: dict) -> None:
    st.title("Metodología, fuentes y calidad de los datos")

    manifiesto = cargar_manifiesto()

    st.subheader("De dónde salen los datos")
    st.markdown(
        f"""
Microdatos de la **Encuesta Nacional de Ingresos y Gastos de los Hogares
(ENIGH) Nueva Serie**, INEGI, descargados de su catálogo de datos abiertos.
Años procesados: **{', '.join(str(a) for a in datos['anios'])}**.

- Panorama general de la encuesta:
  [INEGI · ENIGH 2024](https://www.inegi.org.mx/programas/enigh/nc/2024/)
- Boleta de resultados:
  [Comunicado ENIGH 2024](https://www.inegi.org.mx/contenidos/saladeprensa/boletines/2025/enigh/ENIGH2024.pdf)
- Aviso: la URL con el identificador `tit=2848200` que circula para la descarga
  de la ENIGH 2024 **no existe**; devuelve una página de "archivo no
  encontrado". El proyecto usa las rutas reales del catálogo de datos abiertos.
"""
    )

    st.subheader("Cómo se calculan los deciles")
    st.markdown(
        f"""
1. La unidad de análisis es el **hogar**; el ingreso es el *ingreso corriente*
   (`ing_cor`), tal como lo publica INEGI.
2. La métrica por omisión es el **ingreso corriente per cápita**
   (`ing_cor / integrantes`). La alternativa es el **ingreso por adulto
   equivalente**, con una escala de conveniencia
   (0.7 menores, 0.8 de 12 a 64 años, 1.0 de 65 y más). México no tiene una
   escala de equivalencia oficial; esta se declara explícitamente.
3. Los deciles se forman ordenando a los hogares por la métrica elegida y
   cortando la **población ponderada** en diez partes iguales. Todo cálculo usa
   el factor de expansión (`factor`): aquí no hay promedios simples.
4. Con **cortes nacionales**, el decil 1 es el 10 % más pobre del país. Con
   **deciles propios**, se reparte la población de Sonora en diez partes. Las dos
   vistas se etiquetan de forma distinta en el tablero porque no son comparables
   entre sí.
5. Los montos son **trimestrales en pesos corrientes** del año del
   levantamiento. La opción "pesos constantes de {CFG.anio_base_deflactor}"
   aplica el **INPC de INEGI** (serie 865586, índice general, base 2ª quincena de
   julio 2018 = 100), usando el promedio de los 12 meses de cada año.
6. A partir de 2024, la ENIGH adoptó periodicidad **bienal**: no hay
   levantamiento de 2023 ni de 2025.
"""
    )

    st.subheader("Incertidumbre: por qué hay intervalos de confianza")
    st.markdown(
        f"""
La ENIGH es una encuesta con diseño complejo (estratos y unidades primarias de
muestreo). El tablero calcula intervalos al 95 % por **bootstrap**, remuestreando
UPM completas dentro de cada estrato, no hogares sueltos: remuestrear hogares
subestimaría la varianza real.

Con **{len(datos['sonora']):,} hogares en la muestra de {NOMBRE_SONORA}**
(aproximadamente {len(datos['sonora']) / 10:.0f} por decil), algunas celdas son
genuinamente frágiles. Las estimaciones con coeficiente de variación superior a
**{CFG.umbrales.cv_no_fiable:.0%}** se marcan como no fiables y se atenúan en las
gráficas.
"""
    )

    st.subheader("Limitaciones que conviene tener presentes")
    st.markdown(
        """
- **2020 no es un año normal.** El levantamiento se realizó en condiciones
  atípicas por la pandemia; la comparación con 2022 y 2024 debe leerse con
  cuidado.
- **La muestra municipal es pequeña.** La ENIGH está diseñada para estimaciones
  estatales y nacionales, no municipales. El detalle municipal es indicativo.
- **Comparar 2022 con 2024 está bien; comparar 2020 con 2024 exige cautela**
  por el cambio en el operativo y por la inflación acumulada.
- **Los deciles son cortes de población ponderada**, no grupos de hogares
  iguales en número: un decil puede tener más hogares pequeños que otro.
- **No se aplica la metodología oficial de pobreza multidimensional** de
  CONEVAL. Los indicadores de bienestar son descriptivos y no equivalen a una
  medición de pobreza.
- **El deflactor es el índice general del INPC** (serie 865586 de INEGI, base
  2ª quincena de julio 2018 = 100), promediando los 12 meses de cada año. INEGI
  no publica un campo rotulado "promedio anual", así que ese promedio es una
  media calculada sobre la serie mensual oficial. No se usa la variación anual
  de diciembre, que es un dato distinto.
- **La cartografía municipal proviene del Marco Geoestadístico del INEGI**
  (72 municipios de Sonora con clave ``cvegeo``), redistribuido por un tercero
  con un commit fijado y verificado por SHA-256. Atribuya al INEGI, no al
  repositorio. Para un producto en producción conviene descargar el MGN oficial
  desde https://www.inegi.org.mx/temas/mg/ .
"""
    )

    st.divider()
    st.subheader("Trazabilidad de esta ejecución")
    if not manifiesto:
        st.info("No hay manifiesto. Ejecute el pipeline para generarlo.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Generado", str(manifiesto.get("generado_en", "s/d")))
        col2.metric("Duración", f"{manifiesto.get('segundos_totales', 0):,.0f} s")
        col3.metric("pandas", str(manifiesto.get("pandas", "s/d")))
        for entrada in manifiesto.get("anios", []):
            with st.expander(
                f"Año {entrada['anio']} · {entrada['hogares_en_muestra']:,} hogares "
                f"({entrada['hogares_sonora_muestra']:,} en {NOMBRE_SONORA})"
            ):
                st.json(entrada)
        st.caption(f"Python: {manifiesto.get('python', 's/d')}")
        st.caption(f"Plataforma: {manifiesto.get('plataforma', 's/d')}")

    st.divider()
    _seccion_creditos()


def _seccion_creditos() -> None:
    """Identidad del programa y autoría, en tamaño legible y texto copiable."""
    st.subheader("Sobre este tablero")

    izquierda, derecha = st.columns([1, 4], gap="large")
    with izquierda:
        mostrar_logo_mcd(st, ancho=140)
    with derecha:
        st.markdown(
            f"""
**{PROGRAMA_MCD}** · Universidad de Sonora

{CREDITO}

Este tablero forma parte del trabajo de la [{PROGRAMA_MCD}]({SITIO_MCD}) y se
apoya en software abierto: [Streamlit](https://streamlit.io/) para la interfaz,
[Plotly](https://plotly.com/python/) para las gráficas, [pandas](https://pandas.pydata.org/)
y [NumPy](https://numpy.org/) para el cálculo.
"""
        )
        st.caption(
            "Los microdatos son del INEGI. El emblema identifica al programa "
            "académico y no implica que el INEGI ni la Universidad de Sonora "
            "respalden las cifras aquí presentadas."
        )


# --------------------------------------------------------------------------
# Utilidades de gráficas específicas de la app
# --------------------------------------------------------------------------


def _reindexar_deciles(tabla: pd.DataFrame, columna: str) -> pd.DataFrame:
    """Normaliza una tabla de deciles para poder compararla con otra."""
    if tabla.empty:
        return tabla
    columna_decil = "decil" if "decil" in tabla.columns else "grupo"
    salida = tabla[[columna_decil, columna]].copy()
    return salida.rename(columns={columna_decil: "decil"})


def _contar_traslapes(tabla: pd.DataFrame, columna: str) -> int:
    """Cuenta pares de deciles cuyos intervalos de confianza se traslapan."""
    inferior = tabla.get(f"{columna}_ic_inferior")
    superior = tabla.get(f"{columna}_ic_superior")
    if inferior is None or superior is None:
        return 0
    pares = 0
    valores = list(zip(inferior.tolist(), superior.tolist()))
    for i in range(len(valores)):
        for j in range(i + 1, len(valores)):
            a_inf, a_sup = valores[i]
            b_inf, b_sup = valores[j]
            if any(pd.isna(x) for x in (a_inf, a_sup, b_inf, b_sup)):
                continue
            if a_inf <= b_sup and b_inf <= a_sup:
                pares += 1
    return pares


def _linea_simple(
    datos: pd.DataFrame, *, x: str, y: str, color: str, titulo: str, eje_y: str
) -> go.Figure:
    """Línea temporal por categoría, en eje categórico (años no continuos)."""
    figura = go.Figure()
    for nombre, sub in datos.groupby(color, sort=True):
        sub = sub.sort_values(x)
        figura.add_trace(
            go.Scatter(
                x=[str(v) for v in sub[x]], y=sub[y],
                mode="lines+markers", name=str(nombre),
                line=dict(width=2.6), marker=dict(size=9),
                hovertemplate="%{fullData.name}<br>%{x}: %{y:,.3f}<extra></extra>",
            )
        )
    figura.update_layout(
        template=graficos.PLANTILLA, title=titulo,
        xaxis_title="Año", yaxis_title=eje_y, height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(t=80),
    )
    figura.update_xaxes(type="category")
    return figura


def _barras_indicador(
    sub: pd.DataFrame, *, etiqueta: str, unidad: str, color: str
) -> go.Figure:
    """Barras por decil de un indicador, con IC y atenuación de lo no fiable."""
    colores = [
        color if bool(f) else "rgba(150,150,150,0.45)"
        for f in sub["fiable"].fillna(False)
    ]
    barras = go.Bar(
        x=[f"D{int(g)}" for g in sub["grupo"]],
        y=sub["valor"],
        marker_color=colores,
        error_y=dict(
            type="data", symmetric=False,
            array=(sub["ic_superior"] - sub["valor"]).fillna(0).clip(lower=0),
            arrayminus=(sub["valor"] - sub["ic_inferior"]).fillna(0).clip(lower=0),
            color="#444444", thickness=1, width=3,
        ),
        hovertemplate="%{x}<br>%{y:,.3f} " + unidad + "<extra></extra>",
    )
    figura = go.Figure(barras)
    figura.update_layout(
        template=graficos.PLANTILLA, title=f"{etiqueta} por decil",
        xaxis_title="Decil", yaxis_title=unidad, height=440, showlegend=False,
        margin=dict(t=60, b=50),
    )
    return figura


def _formato_indicador(valor: float, unidad: str) -> str:
    if valor is None or not np.isfinite(valor):
        return "s/d"
    if "proporción" in unidad:
        return f"{valor:.1%}"
    if "pesos" in unidad:
        return f"${valor:,.0f}"
    return f"{valor:,.2f}"


# --------------------------------------------------------------------------
# Punto de entrada
# --------------------------------------------------------------------------


def main() -> None:
    anios = anios_disponibles()

    if not anios:
        st.title("ENIGH · Sonora")
        st.error(
            "Todavía no hay datos procesados en `data/processed`."
        )
        st.markdown(
            """
Ejecute primero el pipeline, que descarga los microdatos de INEGI y los
procesa (unos 25 minutos la primera vez, ~287 MB de descarga):

```bash
make data
# o bien
python -m enigh.pipeline
```

Si solo quiere ver el tablero sin recalcular intervalos de confianza:

```bash
python -m enigh.pipeline --sin-bootstrap
```
"""
        )
        return

    estado = barra_lateral(anios)
    hogares_crudos = cargar_hogares(estado["anio"])

    # Métrica y expresión monetaria elegidas.
    metrica_base = estado["metrica"]
    datos_metrica, columna, unidad = aplicar_deflactor(
        hogares_crudos, estado["anio"], metrica_base, estado["expresion"]
    )

    sonora, resumen = preparar_sonora(
        datos_metrica, metrica=columna, cortes=estado["cortes"]
    )

    # Contexto nacional del mismo año.
    resumen_nacional = resumen_deciles(
        datos_metrica, columna_ingreso=columna, columna_poblacion="tot_integ",
        nombre_metrica=columna,
    )
    entidades = cargar_agregado(estado["anio"], "entidades")
    municipal = _agregado_municipal(estado["anio"], sonora)

    tabla_resumen = resumen.tabla
    promedio_sonora = metricas.media_ponderada(sonora[columna], sonora["factor"])
    promedio_nacional = metricas.media_ponderada(
        datos_metrica[columna], datos_metrica["factor"]
    )

    brecha = _brecha_d10_d1(tabla_resumen)
    lugar = _lugar_nacional(entidades, promedio_sonora)

    datos_vista = {
        "sonora": sonora,
        "resumen": resumen,
        "nacional": {"deciles": resumen_nacional.tabla, "ing_pc": promedio_nacional},
        "entidades": entidades,
        "municipal": municipal,
        "columna": columna,
        "unidad": unidad,
        "anios": anios,
        "brecha_d10_d1": brecha,
        "lugar_sonora": lugar,
        "vs_nacional_pct": (promedio_sonora / promedio_nacional - 1) * 100
        if promedio_nacional
        else 0.0,
    }

    paginas = {
        "Panorama": vista_panorama,
        "Deciles": vista_deciles,
        "Mapas": vista_mapas,
        "Serie de tiempo": vista_serie,
        "Bienestar": vista_bienestar,
        "Metodología y calidad": vista_metodologia,
    }
    paginas[estado["pagina"]](estado, datos_vista)


def _agregado_municipal(anio: int, sonora: pd.DataFrame) -> pd.DataFrame:
    """Agregado municipal de Sonora, con n y fiabilidad por municipio.

    Se prefiere la tabla que dejó el pipeline (que además trae intervalos de
    confianza). Si no existe, se calcula al vuelo para no dejar la vista vacía.
    """
    precomputado = cargar_agregado(anio, "municipios_sonora")
    if not precomputado.empty:
        return precomputado

    if sonora.empty:
        return pd.DataFrame()
    if "nom_municipio" not in sonora.columns:
        sonora = geografia.agregar_municipio(sonora)

    agrupado = (
        sonora.groupby("nom_municipio", dropna=False)
        .apply(
            lambda sub: pd.Series(
                {
                    "n_hogares": len(sub),
                    "n_upm": sub["upm"].nunique() if "upm" in sub else 0,
                    "hogares_expandidos": sub["factor"].sum(),
                    "ing_pc": metricas.media_ponderada(sub["ing_pc"], sub["factor"]),
                    "ing_cor": metricas.media_ponderada(sub["ing_cor"], sub["factor"]),
                    "gini": metricas.gini(sub["ing_pc"], sub["factor"]),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    agrupado["fiable"] = agrupado["n_hogares"] >= CFG.umbrales.hogares_min_municipio
    return agrupado.sort_values("ing_pc", ascending=False)


def _brecha_d10_d1(tabla: pd.DataFrame, columna: str = "ing_pc") -> float:
    """Cuántas veces el ingreso del decil 10 supera al del decil 1."""
    if tabla.empty or columna not in tabla.columns:
        return float("nan")
    bajo = tabla.loc[tabla["decil"] == 1, columna]
    alto = tabla.loc[tabla["decil"] == 10, columna]
    if bajo.empty or alto.empty:
        return float("nan")
    valor_bajo = float(bajo.iloc[0])
    if not np.isfinite(valor_bajo) or valor_bajo <= 0:
        return float("nan")
    return float(alto.iloc[0]) / valor_bajo


def _lugar_nacional(entidades: pd.DataFrame, promedio_sonora: float) -> str:
    """Posición de Sonora entre las 32 entidades (1 = mayor ingreso)."""
    if entidades.empty or "ing_pc" not in entidades.columns:
        return "s/d"
    ordenado = entidades.sort_values("ing_pc", ascending=False).reset_index(drop=True)
    coincidencias = ordenado.index[ordenado["entidad"] == CLAVE_SONORA]
    if len(coincidencias):
        return str(int(coincidencias[0]) + 1)
    # Si el agregado no trae Sonora, se cuenta cuántas entidades la superan.
    return str(int((entidades["ing_pc"] > promedio_sonora).sum()) + 1)


if __name__ == "__main__":
    main()
