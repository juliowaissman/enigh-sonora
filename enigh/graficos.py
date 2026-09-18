"""Constructores de figuras Plotly reutilizables por el tablero.

Regla de honestidad visual: toda estimación que traiga intervalo de confianza se
dibuja **con** su intervalo, y las estimaciones marcadas como poco fiables se
atenúan. Nunca se presenta un promedio municipal frágil como si fuera firme.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Paleta del proyecto.
COLOR_SONORA = "#C1440E"       # terracota, el color del estado de interés
COLOR_NACIONAL = "#1F4E79"     # azul institucional
COLOR_REFERENCIA = "#8C8C8C"   # gris para series de contexto
COLOR_ALERTA = "#B00020"       # rojo para "poco fiable"
SECUENCIA_ENTIDADES = "YlOrBr"

PLANTILLA = "plotly_white"

MENSAJE_SIN_DATOS = "No hay datos suficientes para esta vista."


def _figura_vacia(mensaje: str = MENSAJE_SIN_DATOS) -> go.Figure:
    """Figura con un aviso centrado, para no romper la página."""
    figura = go.Figure()
    figura.add_annotation(
        text=mensaje, showarrow=False, xref="paper", yref="paper",
        x=0.5, y=0.5, font=dict(size=15, color=COLOR_REFERENCIA),
    )
    figura.update_layout(
        template=PLANTILLA, height=320,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return figura


# --------------------------------------------------------------------------
# Deciles
# --------------------------------------------------------------------------


def barras_deciles(
    tabla: pd.DataFrame,
    *,
    columna: str = "ing_pc",
    titulo: str = "Ingreso corriente per cápita por decil",
    unidad: str = "pesos por trimestre",
    color: str = COLOR_SONORA,
    con_ic: bool = True,
) -> go.Figure:
    """Barras por decil con bigotes de intervalo de confianza al 95 %."""
    if tabla.empty or columna not in tabla.columns:
        return _figura_vacia()

    datos = tabla.sort_values("decil")
    etiquetas = [f"D{int(d)}" for d in datos["decil"]]

    barras = go.Bar(
        x=etiquetas,
        y=datos[columna],
        marker_color=color,
        name=titulo,
        hovertemplate="%{x}<br>%{y:,.0f} " + unidad + "<extra></extra>",
    )

    if con_ic:
        inferior = datos.get(f"{columna}_ic_inferior")
        superior = datos.get(f"{columna}_ic_superior")
        if inferior is not None and superior is not None and superior.notna().any():
            barras.error_y = dict(
                type="data",
                symmetric=False,
                array=(superior - datos[columna]).fillna(0).clip(lower=0),
                arrayminus=(datos[columna] - inferior).fillna(0).clip(lower=0),
                color="#444444",
                thickness=1.2,
                width=4,
            )

    figura = go.Figure(barras)
    figura.update_layout(
        template=PLANTILLA,
        title=titulo,
        xaxis_title="Decil (D1 = 20 % más pobre del país · D10 = 20 % más rico)",
        yaxis_title=unidad,
        height=460,
        showlegend=False,
        margin=dict(t=60, b=60),
    )
    figura.update_yaxes(rangemode="tozero")
    return figura


def barras_comparadas(
    series: dict[str, pd.DataFrame],
    *,
    columna: str = "ing_pc",
    titulo: str = "Ingreso per cápita por decil: Sonora frente al país",
    unidad: str = "pesos por trimestre",
    colores: dict[str, str] | None = None,
    con_ic: bool = False,
) -> go.Figure:
    """Barras agrupadas por decil para comparar varios ámbitos."""
    colores = colores or {}
    figura = go.Figure()
    agregado = False

    for nombre, tabla in series.items():
        if tabla is None or tabla.empty or columna not in tabla.columns:
            continue
        datos = tabla.sort_values("decil")
        barras = go.Bar(
            x=[f"D{int(d)}" for d in datos["decil"]],
            y=datos[columna],
            name=nombre,
            marker_color=colores.get(nombre),
            hovertemplate=f"{nombre}<br>%{{x}}<br>%{{y:,.0f}} {unidad}<extra></extra>",
        )
        if con_ic:
            inferior = datos.get(f"{columna}_ic_inferior")
            superior = datos.get(f"{columna}_ic_superior")
            if inferior is not None and superior is not None and superior.notna().any():
                barras.error_y = dict(
                    type="data", symmetric=False,
                    array=(superior - datos[columna]).fillna(0).clip(lower=0),
                    arrayminus=(datos[columna] - inferior).fillna(0).clip(lower=0),
                    color="#444444", thickness=1.0, width=3,
                )
        figura.add_trace(barras)
        agregado = True

    if not agregado:
        return _figura_vacia()

    figura.update_layout(
        template=PLANTILLA, title=titulo, barmode="group",
        xaxis_title="Decil", yaxis_title=unidad, height=470,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(t=80),
    )
    figura.update_yaxes(rangemode="tozero")
    return figura


def linea_serie_deciles(
    serie: pd.DataFrame,
    *,
    columna: str = "ing_pc",
    titulo: str = "Evolución del ingreso per cápita por decil en Sonora",
    unidad: str = "pesos por trimestre",
    groupby: str = "decil",
    con_ic: bool = True,
) -> go.Figure:
    """Serie temporal por decil.

    Los años se colocan en eje categórico a propósito: la ENIGH no es continua
    (2020, 2022, 2024) e interpolarla daría una falsa impresión de tendencia
    suave.
    """
    if serie.empty or columna not in serie.columns:
        return _figura_vacia()

    figura = go.Figure()
    anios = sorted(serie["anio"].unique())
    paleta = _paleta(len(serie[groupby].unique()))

    for i, (grupo, sub) in enumerate(serie.groupby(groupby, sort=True)):
        sub = sub.sort_values("anio")
        color = paleta[i % len(paleta)]
        traza = go.Scatter(
            x=[str(a) for a in sub["anio"]],
            y=sub[columna],
            mode="lines+markers",
            name=f"D{int(grupo)}" if groupby == "decil" else str(grupo),
            line=dict(color=color, width=2.2),
            marker=dict(size=8),
            hovertemplate=(
                "%{fullData.name}<br>Año %{x}<br>%{y:,.0f} " + unidad + "<extra></extra>"
            ),
        )
        if con_ic:
            inferior = sub.get(f"{columna}_ic_inferior")
            superior = sub.get(f"{columna}_ic_superior")
            if inferior is not None and superior is not None:
                traza.error_y = dict(
                    type="data", symmetric=False, color=color, thickness=1,
                    width=3,
                    array=(superior - sub[columna]).fillna(0).clip(lower=0),
                    arrayminus=(sub[columna] - inferior).fillna(0).clip(lower=0),
                )
        figura.add_trace(traza)

    figura.update_layout(
        template=PLANTILLA, title=titulo,
        xaxis_title=f"Año del levantamiento ({', '.join(str(a) for a in anios)})",
        yaxis_title=unidad, height=500,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, x=0),
        margin=dict(t=60, b=110),
    )
    figura.update_xaxes(type="category")
    return figura


def cambio_porcentual_por_decil(
    serie: pd.DataFrame,
    *,
    columna: str = "ing_pc",
    anio_inicial: int | None = None,
    anio_final: int | None = None,
    titulo: str | None = None,
    unidad: str = "en pesos por trimestre",
) -> go.Figure:
    """Variación porcentual entre dos levantamientos, por decil.

    Es la vista que responde si el crecimiento fue pro-pobres o pro-ricos: si las
    barras de los deciles bajos son más altas que las de los altos, el ingreso
    creció de forma más que proporcional abajo.
    """
    if serie.empty or columna not in serie.columns:
        return _figura_vacia()

    anios = sorted(serie["anio"].unique())
    if len(anios) < 2:
        return _figura_vacia(
            "Se necesitan al menos dos levantamientos para calcular una variación."
        )
    inicial = anio_inicial or anios[0]
    final = anio_final or anios[-1]
    if inicial not in anios or final not in anios or inicial == final:
        return _figura_vacia("Los años seleccionados no están disponibles.")

    pivote = serie.pivot_table(
        index="decil", columns="anio", values=columna, aggfunc="first"
    )
    if inicial not in pivote.columns or final not in pivote.columns:
        return _figura_vacia()

    variacion = (pivote[final] / pivote[inicial] - 1) * 100
    variacion = variacion.replace([np.inf, -np.inf], np.nan).dropna()
    if variacion.empty:
        return _figura_vacia()

    colores = [COLOR_SONORA if v >= 0 else COLOR_ALERTA for v in variacion.values]
    figura = go.Figure(
        go.Bar(
            x=[f"D{int(d)}" for d in variacion.index],
            y=variacion.values,
            marker_color=colores,
            hovertemplate="%{x}<br>%{y:+.1f} %<extra></extra>",
        )
    )
    media = float(variacion.mean())
    figura.add_hline(
        y=media, line_dash="dot", line_color=COLOR_NACIONAL,
        annotation_text=f"Promedio {media:+.1f} %",
        annotation_position="top left",
    )
    figura.update_layout(
        template=PLANTILLA,
        title=titulo or f"Cambio del ingreso per cápita {inicial} → {final}, por decil",
        xaxis_title="Decil", yaxis_title="Variación porcentual (%)",
        height=440, showlegend=False,
    )
    return figura


# --------------------------------------------------------------------------
# Mapa y entidades
# --------------------------------------------------------------------------


def mapa_entidades(
    entidades: pd.DataFrame,
    geojson: dict,
    *,
    columna: str = "ing_pc",
    titulo: str = "Ingreso corriente per cápita por entidad federativa",
    unidad: str = "pesos por trimestre",
    entidad_destacada: str | None = "26",
) -> go.Figure:
    """Coroplético nacional por entidad, con Sonora resaltada con línea gruesa."""
    if entidades.empty or not geojson:
        return _figura_vacia(
            "Sin cartografía estatal. Ejecute: python -m enigh.pipeline --solo-geografia"
        )
    if columna not in entidades.columns:
        return _figura_vacia()

    # El empate se hace por nombre normalizado contra las variantes del nombre
    # del GeoJSON, que incluyen los alias: sin ellos, tres entidades con nombre
    # oficial largo (Coahuila de Zaragoza, Michoacán de Ocampo, Veracruz de
    # Ignacio de la Llave) no empatarían y el mapa mostraría 29 de 32.
    from .geografia import alias_entidad, normalizar_nombre

    mapa_nombres = {
        variante: str((f.get("properties") or {}).get("name", ""))
        for f in geojson.get("features", [])
        for variante in alias_entidad((f.get("properties") or {}).get("name"))
    }
    datos = entidades.copy()
    datos["_nombre_geojson"] = datos["nom_entidad"].map(
        lambda n: next(
            (mapa_nombres[v] for v in alias_entidad(n) if v in mapa_nombres), None
        )
    )
    datos = datos[datos["_nombre_geojson"].notna()].copy()
    datos["_nombre_norm"] = datos["_nombre_geojson"].map(normalizar_nombre)
    if datos.empty:
        return _figura_vacia(
            "Los nombres de entidad no empataron con la cartografía; "
            "el detalle está en el log del pipeline."
        )

    figura = go.Figure(
        go.Choropleth(
            geojson=geojson,
            locations=datos["_nombre_norm"],
            featureidkey="properties.name",
            z=datos[columna],
            colorscale=SECUENCIA_ENTIDADES,
            marker_line_color="white",
            marker_line_width=0.6,
            colorbar=dict(title=unidad),
            hovertemplate=(
                "%{customdata[0]}<br>%{z:,.0f} " + unidad + "<extra></extra>"
            ),
            customdata=datos[["nom_entidad"]].to_numpy(),
        )
    )

    if entidad_destacada:
        destacada = entidades[entidades["entidad"] == entidad_destacada]
        if not destacada.empty:
            nombre = destacada["nom_entidad"].iloc[0]
            # El nombre que entiende el GeoJSON puede ser un alias del oficial.
            nombre_en_geojson = next(
                (mapa_nombres[v] for v in alias_entidad(nombre) if v in mapa_nombres),
                nombre,
            )
            figura.add_trace(
                go.Choropleth(
                    geojson=geojson,
                    locations=[normalizar_nombre(nombre_en_geojson)],
                    featureidkey="properties.name",
                    z=[None],
                    colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                    marker_line_color=COLOR_SONORA,
                    marker_line_width=3.5,
                    showscale=False,
                    hoverinfo="skip",
                )
            )

    figura.update_geos(
        fitbounds="locations", visible=False,
        bgcolor="rgba(0,0,0,0)", showcountries=False,
    )
    figura.update_layout(
        template=PLANTILLA, title=titulo, height=560,
        margin=dict(t=60, b=10, l=10, r=10),
    )
    return figura


def mapa_municipios(
    datos: pd.DataFrame,
    geojson: dict | None,
    *,
    columna: str = "ing_pc",
    columna_nombre: str = "nom_municipio",
    titulo: str = "Ingreso corriente per cápita por municipio de Sonora",
    unidad: str = "pesos por trimestre",
    hogares_minimos: int = 30,
) -> go.Figure:
    """Coroplético municipal de Sonora con supresión de celdas frágiles.

    Si no hay GeoJSON municipal, degrada a un mapa de puntos donde el tamaño
    representa los hogares expandidos; el título lo indica para que el lector
    sepa qué está viendo.
    """
    if datos is None or datos.empty or columna not in datos.columns:
        return _figura_vacia("Sin datos municipales suficientes.")

    datos = datos.copy()
    datos["_fiable"] = datos["n_hogares"] >= hogares_minimos
    fiables = datos[datos["_fiable"]]
    suprimidos = datos[~datos["_fiable"]]

    if geojson and fiables.empty:
        return _figura_vacia(
            f"Ningún municipio alcanza los {hogares_minimos} hogares mínimos "
            "para publicar un promedio. Vea la tabla municipal."
        )

    if geojson:
        from .geografia import poner_clave_cvegeo

        # El empate es por clave INEGI de municipio (``cvegeo``), nunca por
        # nombre: 22 de los 72 municipios de Sonora llevan acento y las fuentes
        # geográficas no son consistentes entre sí ("Álamos" vs "Alamos").
        mapa = poner_clave_cvegeo(datos)
        claves_geo = {
            str((f.get("properties", {}) or {}).get("cvegeo", "")).strip()
            for f in geojson.get("features", [])
        }
        empata = mapa["clave_cvegeo"].isin(claves_geo)
        if empata.any():
            figura = go.Figure(
                go.Choropleth(
                    geojson=geojson,
                    locations=mapa.loc[empata, "clave_cvegeo"],
                    featureidkey="properties.cvegeo",
                    z=mapa.loc[empata, columna],
                    colorscale=SECUENCIA_ENTIDADES,
                    marker_line_color="white",
                    marker_line_width=0.5,
                    colorbar=dict(title=unidad),
                    hovertemplate=(
                        "%{customdata[0]}<br>%{z:,.0f} " + unidad
                        + "<br>hogares en muestra: %{customdata[1]}<extra></extra>"
                    ),
                    customdata=mapa.loc[empata, [columna_nombre, "n_hogares"]].to_numpy(),
                )
            )
            figura.update_geos(fitbounds="locations", visible=False)
            figura.update_layout(
                template=PLANTILLA, title=titulo, height=560,
                margin=dict(t=60, b=10, l=10, r=10),
            )
            return figura

    # Degradación: mapa de puntos de municipios (sin geometría disponible).
    if {"lat", "lon"}.issubset(datos.columns) and not datos.empty:
        trazas = []
        if not fiables.empty:
            trazas.append(
                go.Scattergeo(
                    lat=fiables["lat"], lon=fiables["lon"],
                    text=fiables[columna_nombre],
                    mode="markers",
                    marker=dict(
                        size=np.sqrt(fiables["hogares_expandidos"].astype(float)) / 60,
                        color=fiables[columna],
                        colorscale=SECUENCIA_ENTIDADES,
                        colorbar=dict(title=unidad),
                        line=dict(width=0.5, color="white"),
                    ),
                    customdata=fiables[[columna_nombre, columna, "n_hogares"]].to_numpy(),
                    hovertemplate=(
                        "%{customdata[0]}<br>%{customdata[1]:,.0f} " + unidad
                        + "<br>hogares en muestra: %{customdata[2]}<extra></extra>"
                    ),
                    name="Muestra suficiente",
                )
            )
        if not suprimidos.empty:
            # Los municipios frágiles se dibujan en gris y sin valor: se indica
            # que existen, pero no se publica un promedio inestable.
            trazas.append(
                go.Scattergeo(
                    lat=suprimidos["lat"], lon=suprimidos["lon"],
                    text=suprimidos[columna_nombre],
                    mode="markers",
                    marker=dict(size=6, color="#BDBDBD",
                                line=dict(width=0.5, color="white")),
                    customdata=suprimidos[[columna_nombre, "n_hogares"]].to_numpy(),
                    hovertemplate=(
                        "%{customdata[0]}<br>muestra insuficiente "
                        "(%{customdata[1]} hogares)<extra></extra>"
                    ),
                    name="Muestra insuficiente",
                )
            )
        if trazas:
            figura = go.Figure(trazas)
            figura.update_geos(
                scope="north america", fitbounds="locations",
                showland=True, landcolor="#F2F2F2",
            )
            figura.update_layout(
                template=PLANTILLA,
                title=titulo + " (sin geometría municipal: se muestran puntos)",
                height=560, margin=dict(t=60, b=10, l=10, r=10),
            )
            return figura

    return _figura_vacia(
        "No hay cartografía municipal ni coordenadas. Se muestra el detalle en "
        "la tabla municipal."
    )


def puntos_entidades(
    entidades: pd.DataFrame,
    *,
    columna: str = "ing_pc",
    titulo: str = "Ingreso per cápita por entidad: Sonora en el contexto nacional",
    unidad: str = "pesos por trimestre",
) -> go.Figure:
    """Barras ordenadas de entidades, con Sonora resaltada.

    Es el complemento accesible del mapa: comunica el mismo dato sin depender de
    la percepción de color.
    """
    if entidades.empty or columna not in entidades.columns:
        return _figura_vacia()

    datos = entidades.sort_values(columna)
    colores = [COLOR_SONORA if e == "26" else COLOR_REFERENCIA for e in datos["entidad"]]
    mediana = float(datos[columna].median())

    figura = go.Figure(
        go.Bar(
            x=datos[columna], y=datos["nom_entidad"], orientation="h",
            marker_color=colores,
            hovertemplate="%{y}<br>%{x:,.0f} " + unidad + "<extra></extra>",
        )
    )
    figura.add_vline(
        x=mediana, line_dash="dash", line_color=COLOR_NACIONAL,
        annotation_text=f"Mediana nacional {mediana:,.0f}",
        annotation_position="top",
    )
    figura.update_layout(
        template=PLANTILLA, title=titulo, height=760,
        xaxis_title=unidad, yaxis_title="",
        margin=dict(t=60, l=10, r=10, b=40),
    )
    return figura


# --------------------------------------------------------------------------
# Desigualdad
# --------------------------------------------------------------------------


def curva_lorenz(
    valores: pd.Series,
    pesos: pd.Series,
    *,
    titulo: str = "Curva de Lorenz del ingreso corriente per cápita",
) -> go.Figure:
    """Curva de Lorenz con la línea de igualdad perfecta."""
    v = np.asarray(valores, dtype=float)
    w = np.asarray(pesos, dtype=float)
    valido = np.isfinite(v) & np.isfinite(w) & (w > 0) & (v >= 0)
    v, w = v[valido], w[valido]
    if v.size == 0 or v.sum() == 0:
        return _figura_vacia()

    orden = np.argsort(v, kind="stable")
    v, w = v[orden], w[orden]
    peso_acum = np.cumsum(w) / w.sum()
    ingreso_acum = np.cumsum(v * w) / (v * w).sum()

    figura = go.Figure()
    figura.add_trace(
        go.Scatter(
            x=peso_acum, y=ingreso_acum, mode="lines", name="Sonora",
            line=dict(color=COLOR_SONORA, width=3),
            fill="tozeroy", fillcolor="rgba(193,68,14,0.12)",
            hovertemplate="%{x:.1%} de hogares<br>%{y:.1%} del ingreso<extra></extra>",
        )
    )
    figura.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Igualdad perfecta",
            line=dict(color=COLOR_REFERENCIA, dash="dash", width=1.5),
            hoverinfo="skip",
        )
    )
    figura.update_layout(
        template=PLANTILLA, title=titulo,
        xaxis_title="Proporción acumulada de hogares (ordenados por ingreso)",
        yaxis_title="Proporción acumulada del ingreso",
        height=460,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(t=80),
    )
    figura.update_xaxes(tickformat=".0%")
    figura.update_yaxes(tickformat=".0%")
    return figura


def heatmap_indicadores(
    perfil: pd.DataFrame,
    *,
    titulo: str = "Indicadores de bienestar por decil en Sonora",
    indicadores: list[str] | None = None,
) -> go.Figure:
    """Mapa de calor decil × indicador, normalizado por fila.

    Cada fila se estandariza a 0-1 para que indicadores con unidades muy
    distintas (proporciones, pesos, personas por cuarto) sean comparables en un
    solo golpe de vista. El valor real se muestra en el hover.
    """
    if perfil is None or perfil.empty:
        return _figura_vacia()

    datos = perfil.copy()
    if indicadores:
        datos = datos[datos["indicador"].isin(indicadores)]
    if datos.empty:
        return _figura_vacia()

    pivote = datos.pivot_table(
        index="etiqueta", columns="grupo", values="valor", aggfunc="first"
    ).sort_index()
    if pivote.empty:
        return _figura_vacia()

    # Normalización por fila, robusta a filas constantes.
    minimo = pivote.min(axis=1)
    maximo = pivote.max(axis=1)
    rango = (maximo - minimo).replace(0, np.nan)
    normalizado = pivote.sub(minimo, axis=0).div(rango, axis=0)
    normalizado = normalizado.fillna(0.5)

    texto = pivote.map(lambda x: f"{x:,.2f}" if pd.notna(x) else "")

    figura = go.Figure(
        go.Heatmap(
            z=normalizado.values,
            x=[f"D{int(c)}" for c in pivote.columns],
            y=list(pivote.index),
            colorscale="RdYlBu_r",
            zmin=0, zmax=1,
            text=texto.values,
            texttemplate="%{text}",
            textfont=dict(size=9),
            hovertemplate="%{y}<br>%{x}<br>valor: %{text}<extra></extra>",
            colorbar=dict(title="Posición relativa<br>dentro del indicador"),
        )
    )
    figura.update_layout(
        template=PLANTILLA, title=titulo,
        height=max(420, 34 * len(pivote) + 160),
        xaxis_title="Decil de ingreso (cortes nacionales)",
        margin=dict(t=60, l=10, r=10, b=50),
    )
    return figura


def barras_composicion(
    tabla: pd.DataFrame,
    *,
    titulo: str = "Composición del ingreso corriente por decil",
) -> go.Figure:
    """Barras apiladas al 100 % de las fuentes del ingreso por decil."""
    if not tabla.empty:
        tabla = tabla.copy()

    # Se detecta si la tabla ya trae columnas de participación por fuente.
    fuentes = [c for c in tabla.columns if c.startswith("share_")]
    etiqueta = "decil" if "decil" in tabla.columns else "grupo"
    if tabla.empty or not fuentes or etiqueta not in tabla.columns:
        return _figura_vacia(
            "Sin desglose de fuentes de ingreso para esta selección."
        )

    nombres = {
        "share_ingtrab": "Trabajo",
        "share_negocio": "Negocio propio",
        "share_transfer": "Transferencias",
        "share_remesas": "Remesas",
        "share_bene_gob": "Beneficios gubernamentales",
        "share_jubilacion": "Jubilaciones",
        "share_rentas": "Rentas de la propiedad",
        "share_estim_alqu": "Alquiler imputado",
    }
    figura = go.Figure()
    for fuente in fuentes:
        figura.add_trace(
            go.Bar(
                x=[f"D{int(d)}" for d in tabla[etiqueta]],
                y=tabla[fuente],
                name=nombres.get(fuente, fuente),
                hovertemplate="%{fullData.name}<br>%{x}<br>%{y:.1%}<extra></extra>",
            )
        )
    figura.update_layout(
        template=PLANTILLA, title=titulo, barmode="stack",
        xaxis_title="Decil", yaxis_title="Proporción del ingreso corriente",
        height=470,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, x=0),
        margin=dict(t=60, b=130),
    )
    figura.update_yaxes(tickformat=".0%")
    return figura


def _paleta(n: int) -> list[str]:
    """Paleta categórica reproducible para hasta ``n`` series."""
    base = [
        "#C1440E", "#1F4E79", "#2E7D32", "#6A1B9A", "#EF6C00",
        "#00838F", "#4E342E", "#AD1457", "#558B2F", "#37474F",
    ]
    if n <= len(base):
        return base[: max(n, 1)]
    repetida = (base * (n // len(base) + 1))[:n]
    return repetida
