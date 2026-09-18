"""Paquete de análisis de la ENIGH (INEGI) con foco en el estado de Sonora.

Módulos:
    config      -- carga de configuración y rutas del proyecto
    descarga    -- descarga robusta de los ZIP de microdatos de INEGI
    extraccion  -- descompresión selectiva de tablas
    limpieza    -- lectura y normalización de los CSV a Parquet
    metricas    -- deciles, Gini, Palma, Theil y demás medidas ponderadas
    estimacion  -- intervalos de confianza por bootstrap con diseño muestral
    bienestar   -- indicadores de bienestar agregados a nivel hogar
    geografia   -- etiquetado territorial y carga de GeoJSON
    graficos    -- constructores de figuras Plotly
    pipeline    -- orquestador con interfaz de línea de comandos
"""

__version__ = "0.1.0"

# Entidad de interés del proyecto.
CLAVE_SONORA = "26"
NOMBRE_SONORA = "Sonora"

# Los montos de la ENIGH son trimestrales y en pesos corrientes del año del
# levantamiento. Se etiquetan así en todo el tablero; nunca se anualizan.
UNIDAD_INGRESO = "pesos corrientes por trimestre"
