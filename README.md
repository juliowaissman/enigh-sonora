# ENIGH · Ingreso de las familias en Sonora por decil

Aplicación de análisis de datos que descarga la **Encuesta Nacional de Ingresos y
Gastos de los Hogares (ENIGH)** del INEGI, la procesa y sirve un tablero
interactivo en Streamlit sobre cómo está el ingreso de los hogares en **Sonora**,
por decil, con comparación nacional, mapas, serie de tiempo 2020-2022-2024 e
indicadores de bienestar más allá del ingreso.

---

## Arranque rápido

```bash
make install     # crea el entorno virtual e instala dependencias
make data        # descarga la ENIGH y procesa los tres años (~5 min, 287 MB)
make app         # levanta el tablero en http://localhost:8501
```

Si tienes prisa, `make data-fast` omite el cálculo de intervalos de confianza
(bootstrap) y tarda una fracción del tiempo; el tablero funciona igual, solo que
las gráficas no muestran bigotes de incertidumbre.

> **`make data` es obligatorio antes de `make app`.** El repositorio **no incluye
> datos**: son ~911 MB y se regeneran desde las fuentes públicas del INEGI con el
> pipeline incluido. Si abres el tablero sin haber corrido el pipeline, no falla:
> te muestra en pantalla los comandos exactos que faltan.

Sin `make`:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m enigh.pipeline
.venv/bin/streamlit run app.py
```

---

## Qué responde el tablero

| Vista | Qué muestra |
|---|---|
| **Panorama** | Los indicadores que resumen la situación de Sonora, con su intervalo de confianza, y la curva de Lorenz. |
| **Deciles** | Tabla completa por decil (ingreso, mediana, participación, CV, fiabilidad), comparación Sonora vs. país decil por decil, y composición del ingreso por fuente. |
| **Mapas** | Coroplético de las 32 entidades con Sonora resaltada y desglose municipal de Sonora (72 municipios) con supresión de celdas frágiles. |
| **Serie de tiempo** | Evolución 2020 → 2022 → 2024 por decil, cambio porcentual por decil (¿creció más el ingreso de los pobres o el de los ricos?), y evolución del Gini y la razón de Palma. |
| **Bienestar** | 37 indicadores por decil: alimentación, educación, salud, vivienda, servicios básicos, conectividad, bienes, demografía y vulnerabilidad. |
| **Metodología y calidad** | Fuentes, definiciones exactas, tamaño de muestra por celda y limitaciones. |

Todo el texto está en español y cada gráfica indica sus unidades. Los montos de
la ENIGH son **trimestrales en pesos corrientes** del año del levantamiento; el
tablero lo etiqueta así siempre y nunca anualiza.

---

## Decisiones metodológicas

Las decisiones que más afectan la lectura de los resultados, y por qué:

**Unidad y métrica.** La unidad de análisis es el hogar; el ingreso es el
*ingreso corriente* (`ing_cor`). La métrica por omisión es el **ingreso corriente
per cápita** (`ing_cor / integrantes`). El tablero permite cambiarla por el
**ingreso por adulto equivalente** (0.7 menores, 0.8 de 12 a 64 años, 1.0 de 65 y
más). México no publica una escala de equivalencia oficial; esta es una elección
de conveniencia, declarada como tal en la interfaz. El denominador está acotado al
número de integrantes: si los bloques de edad del archivo fueran inconsistentes,
sin esa cota el indicador podría quedar por debajo del per cápita, que es lo
contrario de lo que la escala significa.

**Deciles.** Se ordenan los hogares por la métrica elegida y se corta la
**población ponderada** en diez partes iguales. Dos variantes, nunca mezcladas:

- **Cortes nacionales** (por omisión): el decil 1 es el 10 % más pobre de México.
  Así Sonora se compara contra el país con el mismo rasero.
- **Deciles propios**: se reparte la población de Sonora en diez partes. Útil para
  ver la forma interna de la distribución estatal, pero no comparable con la
  anterior.

**Ponderadores.** Todo cálculo usa el factor de expansión (`factor`). No hay
promedios simples en ningún punto del código.

**Incertidumbre.** Con 2,642 hogares en la muestra de Sonora (~264 por decil),
algunas celdas son genuinamente frágiles. Los intervalos al 95 % se calculan por
**bootstrap remuestreando UPM completas dentro de cada estrato**, no hogares
sueltos: remuestrear hogares subestimaría la varianza porque los hogares de una
misma UPM no son independientes. Las estimaciones con coeficiente de variación
superior a 25 % se marcan como no fiables y se atenúan en las gráficas.

**Deflactación.** El INPC es el índice general de INEGI (serie 865586, base 2ª
quincena de julio de 2018 = 100), promediando los 12 meses de cada año. INEGI no
publica un campo rotulado "promedio anual", así que ese promedio es una media
calculada sobre la serie mensual oficial, y así está documentado. No se usa la
variación anual de diciembre, que es un dato distinto.

**Supresión municipal.** La ENIGH está diseñada para estimaciones estatales y
nacionales, no municipales. Los municipios con menos de 30 hogares en muestra no
publican promedio: se agrupan o se dibujan en gris como "muestra insuficiente".

---

## Fuentes de datos

| Qué | De dónde |
|---|---|
| Microdatos ENIGH 2020, 2022, 2024 | Catálogo de datos abiertos del INEGI. Las URLs exactas viven en `config/fuentes.yaml`. |
| Cartografía de entidades | GeoJSON público de 32 features; empate por nombre normalizado contra el catálogo `ubica_geo.csv` del propio ZIP. |
| Cartografía municipal de Sonora | Marco Geoestadístico del INEGI (72 municipios, clave `cvegeo`), vía un repositorio redistribuidor con **commit fijado y verificados por SHA-256** (`7d9dd414…`). Se atribuye al INEGI. Para producción conviene bajar el MGN oficial desde <https://www.inegi.org.mx/temas/mg/>. |
| INPC | INEGI, serie 865586 (índice general, base 2ª quincena de julio 2018 = 100). Valores en `config/deflactores.csv`; la evidencia de la verificación (boletines de INEGI y la serie descargada) está en `docs/`. |

**Nota sobre la URL del enunciado.** La dirección
`https://www.inegi.org.mx/app/descarga/ficha.html?tit=2848200&ag=0&f=csv` **no
existe**: INEGI responde "El archivo que estás buscando no existe o no cumple con
el estándar de datos abiertos". El proyecto usa las rutas reales del catálogo de
datos abiertos.

**Trampa de INEGI que el código maneja explícitamente.** Una ruta inexistente en
`inegi.org.mx` responde **HTTP 200 con un HTML de ~2,263 bytes**, no un 404. Por
eso la descarga valida el `Content-Type`, la firma ZIP (`PK\x03\x04`) y el tamaño,
y además verifica la integridad con `zipfile.testzip()`; confiar en el código de
estado guardaría un archivo de error como si fueran microdatos.

**Dos trampas más de los archivos de INEGI**, ambas descubiertas al procesar los
datos reales y cubiertas por pruebas:

- **La codificación cambia entre años.** La ENIGH 2020 viene en `utf-8-sig` y la
  de 2022 en `latin-1`. Leerla con una sola codificación falla con
  `UnicodeDecodeError` (o peor, produce texto corrupto). `limpieza.detectar_codificacion`
  prueba las candidatas en orden y se queda con la primera que decodifica.
- **Los códigos binarios son 1 = Sí y 2 = No**, no 1/0. Contar "cualquier valor
  mayor que cero" trataría el *no* como afirmativo; el error es especialmente
  traicionero porque produce porcentajes plausibles a simple vista (un 100 %
  constante) en lugar de un fallo visible.

---

## Estructura del proyecto

```
enigh/              paquete de análisis (sin dependencia de Streamlit)
  config.py         rutas, fuentes y umbrales
  descarga.py       descarga robusta y validada de los ZIP
  extraccion.py     descompresión selectiva de tablas
  limpieza.py       lectura, validación y derivación de columnas
  metricas.py       deciles, Gini, Theil, Palma y percentiles ponderados
  estimacion.py     bootstrap con diseño muestral (UPM dentro de estrato)
  bienestar.py      catálogo declarativo de 37 indicadores
  geografia.py      etiquetado territorial y cartografía
  graficos.py       figuras Plotly reutilizables
  validacion.py     comprobaciones contra cifras de control
  pipeline.py       orquestador con CLI
app.py              tablero Streamlit
config/             fuentes.yaml, deflactores.csv, validacion.csv
data/               NO versionado: se genera con `make data` (solo .gitkeep)
docs/               evidencia de la verificación del INPC (boletines y serie)
tests/              197 pruebas (unidad, integración y humo del tablero)
```

El catálogo de indicadores de bienestar se declara en `enigh/bienestar.py` (una
sola lista con dominio, etiqueta, unidad y forma de cálculo) en lugar de en YAML:
cada indicador referencia una columna concreta del DataFrame, así que tenerlo en
Python permite que el editor y las pruebas detecten un nombre mal escrito, cosa
que una cadena en YAML no haría.

### Sobre los datos en el control de versiones

`data/` está en `.gitignore` y **no** forma parte del repositorio:

| Carpeta | Tamaño | Qué es |
|---|---|---|
| `data/raw/` | ~276 MB | ZIP originales descargados de INEGI |
| `data/interim/` | ~575 MB | CSV extraídos de esos ZIP |
| `data/processed/` | ~60 MB | Parquet curado, agregados y cartografía |

Se versionan solo el código, la configuración, las pruebas y `docs/` (1.6 MB de
evidencia del INPC), de modo que el repositorio pesa ~1.5 MB. La razón es que los
datos son **enteramente reproducibles** desde fuentes públicas con el pipeline
incluido: versionarlos inflaría cada clon con ~911 MB y duplicaría en git algo que
ya está en línea en el INEGI.

`docs/` sí se versiona a propósito: son los boletines del INPC que respaldan los
deflactores, y sin ellos no se podría auditar la serie de tiempo en pesos
constantes.

---

## Comandos

```bash
python -m enigh.pipeline                     # años habilitados, proceso completo
python -m enigh.pipeline --anios 2024        # un solo año
python -m enigh.pipeline --sin-bootstrap     # sin intervalos (rápido)
python -m enigh.pipeline --solo-descarga     # solo baja los ZIP
python -m enigh.pipeline --solo-geografia    # solo la cartografía
python -m enigh.pipeline --forzar            # ignora cachés
python -m enigh.pipeline --muestra 3000      # modo demostración (ver abajo)
python -m enigh.validacion                   # valida contra cifras de control
python -m pytest tests/ -v                   # pruebas
python -m pytest tests/ -m integration       # solo las que usan datos reales
```

El pipeline deja `data/processed/manifiesto.json` con las URLs, la fecha de
descarga, los conteos, los cortes decílicos y las versiones de las dependencias,
para que cualquier resultado del tablero sea trazable.

**Modo demostración.** `--muestra N` conserva todos los hogares de Sonora y
recorta el resto del país, de modo que el tablero se pueda revisar en una fracción
del tiempo y del espacio en disco. Los factores de expansión **no** se recalculan
a propósito: hacerlo mantendría los totales poblacionales pero produciría
intervalos de confianza demasiado estrechos, que es justo el error que se busca
evitar. Por eso el manifiesto marca la ejecución como reducida y el tablero lo
advierte. Para el análisis real, vuelva a ejecutar sin `--muestra`.

**Validación.** `python -m enigh.validacion` comprueba las condiciones sobre los
datos procesados: las cifras de control de `config/validacion.csv` (conteos de
hogares, número de entidades), invariantes internas (monotonía del ingreso por
decil, participaciones que suman 100 %, Gini en rango, ingreso per cápita nunca
mayor que el del hogar) y coherencia entre archivos. Incluye una comprobación
específica de que el Gini de la tabla de deciles no se confunda con el Gini
global: el primero mide la dispersión *dentro* de un decil (~0.17) y el segundo la
desigualdad de toda la distribución (~0.45).

---

## Limitaciones que conviene tener presentes

- **2020 no es un año normal.** El levantamiento se hizo en condiciones atípicas
  por la pandemia de COVID-19. Comparar 2022 con 2024 es razonable; comparar 2020
  con 2024 exige cautela.
- **La muestra municipal es pequeña.** El detalle municipal es indicativo, no
  concluyente.
- **La ENIGH es bienal desde 2024.** No hay levantamiento de 2023 ni de 2025; la
  serie tiene tres puntos y no se interpola, porque una línea suave entre 2020 y
  2024 sugeriría una tendencia continua que los datos no sostienen.
- **No se aplica la metodología oficial de pobreza multidimensional** de CONEVAL.
  Los indicadores de bienestar son descriptivos y **no equivalen a una medición
  de pobreza**.
- **La inseguridad alimentaria se reporta en dos tramos, no con la escala
  oficial.** La medida amplia ("alguna dificultad alimentaria") ronda el 95 % de
  los hogares porque incluye la simple preocupación de que la comida se acabe; la
  medida severa ("hambre o una sola comida al día") es la que conviene citar. La
  escala oficial de puntos y sus cortes no están en los microdatos abiertos.
- **Los deciles son cortes de población ponderada**, no grupos de hogares iguales
  en número: un decil puede concentrar más hogares pequeños que otro.

---

## Entorno

Desarrollado y probado con **Python 3.12.8**, `pandas` 2.2.3, `numpy` 2.1.3,
`streamlit` 1.41.1, `plotly` 5.24.1 y `pyarrow` 18.1.0. Deliberadamente **no se
usa `geopandas` ni `shapely`**: requieren compilación frágil y los mapas se
construyen con Plotly sobre GeoJSON crudo en memoria.

El intérprete de Python del proyecto vive en `.tools/` (no versionado) porque el
entorno de desarrollo no permite escribir en `/opt/homebrew` ni en `~/.local`.
