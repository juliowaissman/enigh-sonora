# INPC México — índice general anual promedio, base 2ª quincena de julio de 2018 = 100
## Años 2019, 2020, 2021, 2022, 2023 y 2024

**Fecha de elaboración:** 17 de septiembre (datos vigentes según INEGI al último corte consultado).
**Serie oficial usada:** INEGI, INPC Nacional, serie/indicador **865586**, nombre **"Índice general"**, unidad
**"Índice base segunda quincena de julio 2018 = 100"**, región **Nacional**, frecuencia **Mensual**.
(Fuente primaria: API interna de indicadores de INEGI — ver sección 2.)

---

## 1. Resultado

El "índice general anual promedio" se calcula, como es estándar, como el **promedio aritmético de los
12 índices mensuales oficiales del año**. INEGI no devuelve por esta vía un campo ya precalculado de
"promedio anual", por lo que el promedio se obtuvo a partir de los 12 valores mensuales oficiales
(mostrados en la sección 1b). No se inventó ni interpoló ningún dato.

| Año | Dato reportado | Valor | Fuente |
|-----|----------------|-------|--------|
| 2019 | Índice general anual promedio (media de los 12 índices mensuales oficiales) | **103.901** | INEGI, serie 865586 |
| 2020 | Índice general anual promedio (media de los 12 índices mensuales oficiales) | **107.430** | INEGI, serie 865586 |
| 2021 | Índice general anual promedio (media de los 12 índices mensuales oficiales) | **113.542** | INEGI, serie 865586 |
| 2022 | Índice general anual promedio (media de los 12 índices mensuales oficiales) | **122.508** (media exacta 122.5075) | INEGI, serie 865586 |
| 2023 | Índice general anual promedio (media de los 12 índices mensuales oficiales) | **129.280** (media exacta 129.279667) | INEGI, serie 865586 |
| 2024 | Índice general anual promedio (media de los 12 índices mensuales oficiales) | **135.385** (media exacta 135.384583) | INEGI, serie 865586 |

### 1a. Dato complementario: índice de diciembre de cada año (nivel, no variación)
Este dato sí está publicado de forma explícita (verificable uno a uno) y sirve como control:

| Año | Índice general de diciembre (base 2Q-jul-2018 = 100) | Variación anual de diciembre (publicada) |
|-----|------------------------------------------------------|-------------------------------------------|
| 2019 | 105.934 | 2.83 % |
| 2020 | 109.271 | 3.15 % |
| 2021 | 117.308 | 7.36 % |
| 2022 | 126.478 | 7.82 % |
| 2023 | 132.373 | 4.66 % |
| 2024 | 137.949 | 4.21 % |

### 1b. Índices mensuales oficiales usados para el promedio (INEGI, serie 865586)

| Año | Ene | Feb | Mar | Abr | May | Jun | Jul | Ago | Sep | Oct | Nov | Dic |
|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| 2019 | 103.108 | 103.079 | 103.476 | 103.531 | 103.233 | 103.299 | 103.687 | 103.670 | 103.942 | 104.503 | 105.346 | 105.934 |
| 2020 | 106.447 | 106.889 | 106.838 | 105.755 | 106.162 | 106.743 | 107.444 | 107.867 | 108.114 | 108.774 | 108.856 | 109.271 |
| 2021 | 110.210 | 110.907 | 111.824 | 112.190 | 112.419 | 113.018 | 113.682 | 113.899 | 114.601 | 115.561 | 116.884 | 117.308 |
| 2022 | 118.002 | 118.981 | 120.159 | 120.809 | 121.022 | 122.044 | 122.948 | 123.803 | 124.571 | 125.276 | 125.997 | 126.478 |
| 2023 | 127.336 | 128.046 | 128.389 | 128.363 | 128.084 | 128.214 | 128.832 | 129.545 | 130.120 | 130.609 | 131.445 | 132.373 |
| 2024 | 133.555 | 133.681 | 134.065 | 134.336 | 134.087 | 134.594 | 136.003 | 136.013 | 136.080 | 136.828 | 137.424 | 137.949 |

(Valores copiados tal cual de la respuesta de la API de INEGI; se conserva la serie completa en
`evidencia_inpc/inpc_serie_865586_inegi.json`.)

### 1c. Comprobación independiente (Banxico)
El Informe Anual / apéndice estadístico de Banxico reproduce el INPC con la **misma base 2Q-jul-2018 = 100**
y publica la variación "Promedio Anual" y el "Promedio Móvil de 12 meses", que coinciden exactamente con
la razón entre los promedios anuales calculados arriba:

| Año | Variación "Promedio Anual" (Banxico, cuadro A16) | Ratio de los promedios calculados |
|-----|--------------------------------------------------|------------------------------------|
| 2019 | 3.64 % | 103.9007 / 100.2554 − 1 = 3.64 % |
| 2020 | 3.40 % | 107.4300 / 103.9007 − 1 = 3.40 % |
| 2021 | 5.69 % | 113.5419 / 107.4300 − 1 = 5.69 % |
| 2022 | 7.90 % | 122.5075 / 113.5419 − 1 = 7.90 % |
| 2023 | 5.53 % | 129.2797 / 122.5075 − 1 = 5.53 % |
| 2024 | 4.72 % | 135.3846 / 129.2797 − 1 = 4.72 % |

Además, el cuadro A17 de Banxico lista el **nivel mensual del índice** (base 2Q-jul-2018 = 100) para
Dic-2018 a Dic-2024 y coincide **valor por valor** con la serie de INEGI usada (p. ej. Dic-2019 105.934,
Dic-2021 117.308, Dic-2023 132.373, Dic-2024 137.949).

---

## 2. URLs exactas consultadas y resultado

| # | URL | ¿Respondió bien? | Qué se obtuvo |
|---|-----|------------------|----------------|
| 1 | `https://www.inegi.org.mx/app/api/indicadores/interna_v1_3/API.svc/indicador/865586/0700/es/false/inp/json/96fbd1bf-21e6-28e3-6e64-2b15999d2c89?callback=cb` | **Sí (HTTP 200, JSON con datos)** | Serie mensual completa del "Índice general", base 2Q-jul-2018=100, 1970/01–2026/08. **Fuente primaria del reporte.** |
| 2 | `https://www.inegi.org.mx/app/indicesdeprecios/Estructura.aspx?ST=INPC+Nacional+%28mensual%29&T=%C3%8Dndices+de+Precios+al+Consumidor&idEstructura=112001700030` | Sí (HTTP 200; app JS) | Página de estructura "INPC Nacional (mensual)". De aquí, vía su servicio `ArbolAjaxInteraccion.asmx/EstructuraInicialV2`, se identificó la serie **865586 = "Índice general"**. |
| 3 | `https://www.inegi.org.mx/app/indicesdeprecios/servicios/ArbolAjaxInteraccion.asmx/EstructuraInicialV2` (POST) | Sí | Devuelve el árbol y las claves de serie; confirmó `CBox_Serie_865586` = "Índice general". |
| 4 | `https://www.banxico.org.mx/publicaciones-y-prensa/informes-anuales/%7B89C6FBCA-9A51-A6F4-D55A-D58E340B5CA9%7D.pdf` | Sí (PDF, 38.9 MB) | Cuadro A16 (variación "Promedio Anual" 2019–2024) y Cuadro A17 (nivel mensual del INPC base 2Q-jul-2018, Dic-2004 a Dic-2024). Coincide con INEGI. |
| 5 | `https://www.inegi.org.mx/contenidos/saladeprensa/boletines/2025/inpc/inpc_2q2025_01.pdf` | Sí (HTTP 200, PDF) | Boletín INPC **diciembre 2024**: "En diciembre de 2024, el INPC presentó un nivel de **137.949**… la inflación general anual se ubicó en 4.21 por ciento". |
| 6 | `https://www.inegi.org.mx/contenidos/saladeprensa/boletines/2024/inpc_2q/inpc_2q2024_01.pdf` | Sí (PDF) | Boletín INPC **diciembre 2023**: inflación anual 4.66 %; niveles quincenales 132.058 / 132.688. |
| 7 | `https://www.inegi.org.mx/contenidos/saladeprensa/boletines/2023/inpc_2q/inpc_2q2023_01.pdf` | Sí (PDF) | Boletín INPC **diciembre 2022**: inflación anual 7.82 %; quincenales 126.417 / 126.539. |
| 8 | `https://www.inegi.org.mx/contenidos/saladeprensa/boletines/2022/inpc_2q/inpc_2q2022_01.pdf` | Sí (PDF) | Boletín INPC **diciembre 2021**: inflación anual 7.36 %; quincenales 117.301 / 117.314. |
| 9 | `https://www.inegi.org.mx/contenidos/saladeprensa/boletines/2021/inpc_2q/inpc_2q2021_01.pdf` | Sí (PDF) | Boletín INPC **diciembre 2020**: inflación anual 3.15 %; quincenales 109.168 / 109.374. |
| 10 | `https://www.inegi.org.mx/contenidos/saladeprensa/boletines/2020/inpc_2q/inpc_2q2020_01.pdf` | Sí (PDF) | Boletín INPC **diciembre 2019**: inflación anual 2.83 %; quincenales 105.763 / 106.105. |
| 11 | `https://www.inegi.org.mx/rnm/index.php/catalog/1015` | Sí (HTTP 200) | Catálogo RNM "INPC 2024, base 2Q-jul-2018 = 100". **Solo metadatos/documentación metodológica; no contiene la serie de valores.** |
| 12 | `https://micrs.sct.gob.mx/images/DireccionesGrales/DGP/estadistica/Indicador-Mensual/INDI-2025/CI_Agosto-2025.pdf` | Sí (PDF) | Cuadro 5 (SCT, con fuente INEGI): INPC Dic-2022 = 126.5, Dic-2023 = 132.4, Dic-2024 = 137.9. Coincide. |

### URLs que NO sirvieron (para que no se reintenten)
| URL | Resultado |
|-----|-----------|
| `https://www.inegi.org.mx/contenidos/saladeprensa/boletines/2025/inpc/inpc_2025_01.pdf` | HTTP 200 pero página "no encontrada" (2263 bytes de HTML). |
| `https://www.inegi.org.mx/contenidos/saladeprensa/boletines/2025/inpc_2q/inpc_2q2025_01.pdf` | Página "no encontrada". El boletín de dic-2024 está en la carpeta `inpc` (no `inpc_2q`) — ver fila 5. |
| `https://www.inegi.org.mx/app/api/indicadores/desarrolladores/jsonxml/INDICATOR/.../BIE/2.0/{token}?type=json` | Responde, pero devuelve `ErrorCode:100 "No se encontraron resultados"` para todos los identificadores probados. No se pudo usar. |
| `https://www.inegi.org.mx/contenidos/programas/inpc/2018/datosabiertos/*.zip` (y variantes) | Todas devolvieron la página "no encontrada"; no existen con esos nombres. |
| Descarga del Reporte Económico del CEFP (`cefp.gob.mx/.../252-250109.pdf`) | No descargó desde este entorno; no se usó. |

---

## 3. Notas explícitas sobre verificación y límites

1. **El promedio anual es una media calculada, no un campo publicado como tal.** INEGI no entrega, en el
   servicio consultado (serie 865586), un valor ya rotulado "promedio anual". Los seis promedios de la
   tabla principal son la media aritmética de los 12 índices mensuales oficiales de cada año (sección 1b).
   No se inventó ni se interpoló ningún valor.
2. **El nivel de diciembre de cada año sí quedó verificado contra publicación oficial:** Dic-2024 (137.949)
   está escrito textualmente en el boletín de INEGI de diciembre de 2024; Dic-2019/2020/2021/2022/2023
   (105.934 / 109.271 / 117.308 / 126.478 / 132.373) coinciden **exactamente** con los cuadros A16/A17 del
   Informe Anual de Banxico (que cita a INEGI) y con los niveles quincenales y variaciones anuales de los
   boletines de INEGI de cada diciembre.
3. **Base y vigencia.** Todos los valores están en la base **2ª quincena de julio de 2018 = 100**. Para
   2019–2023 se usó la serie oficial vigente publicada por INEGI (posterior a la Actualización del INPC 2024
   de agosto de 2024, que mantuvo fija esa base e introdujo índices encadenados); esta serie reproduce
   exactamente los niveles mensuales que Banxico publica para Dic-2019 a Dic-2024. No se detectaron
   discrepancias.
4. **Ningún valor de esta tabla quedó sin verificar.** Los seis promedios anuales provienen de los 72
   índices mensuales oficiales; los 6 índices de diciembre se confirmaron además contra una segunda fuente
   oficial independiente (Banxico) y 1 de ellos (2024) contra el texto del boletín de INEGI.
5. **Advertencia sobre la variación porcentual:** la "variación anual de diciembre" (2.83 %, 3.15 %, 7.36 %,
   7.82 %, 4.66 %, 4.21 %) es un dato distinto del índice anual promedio y se incluye solo como control;
   no debe confundirse con la columna de promedios.
6. La fecha de corte de la serie descargada es la del último periodo disponible en la API (agosto 2026);
   los años 2019–2024 son periodos cerrados y no cambian.

---

## 4. Archivos de respaldo en el espacio de trabajo
- `evidencia_inpc/inpc_serie_865586_inegi.json` — respuesta JSON completa de la API de INEGI (serie mensual 1970/01–2026/08).
- `evidencia_inpc/inpc_diciembre2019_boletin.pdf` … `inpc_diciembre2024_boletin.pdf` — boletines de prensa de INEGI de diciembre de cada año.
