---
tags: [tfg, redaccion, deposito, defensa, ods, referencia]
---

# Seminarios TFG — cosas a tener en cuenta

Resumen accionable de la carpeta `TrabajoFindeGradoGCD/` (seminarios oficiales ETSINF-UPV:
redacción, estructura, citas, revisión, depósito, defensa, ODS, competencias transversales,
emprendimiento). Destilado para **redactar / depositar / defender** este TFG de **Ciencia de
Datos (GCD)**. Las `.URL` de las carpetas son solo enlaces a vídeos internos (tortuga.inf.upv.es).

> **Aviso transversal:** todo el material está hecho para el **Grado en Ingeniería Informática
> (GII)**. El 99 % vale igual, pero la **estructura de la memoria** y la **estructura de la
> presentación** difieren para GCD. Para esas dos cosas manda la **plantilla específica de GCD**,
> no los ejemplos del GII. Ver §0.

---

## 0. Lo que difiere para GCD (leer primero)

- **No copies la estructura de capítulos del GII.** El ejemplo del seminario (transparencia 12 de
  "Estructura y contenidos") trae *estudio de mercado*, *especificación de requisitos*, etc.: es
  perfil de ingeniería de software, **no aplica a Ciencia de Datos**. Usa el esqueleto genérico
  (§1) adaptado a un proyecto de datos.
- **Plantilla y estructura oficial de GCD:** <https://www.inf.upv.es/www/etsinf/es/plantilla-tfg/>.
  La clase LaTeX `tfgetsinf.cls` que ya usamos es la común de la ETSINF y sirve; lo que hay que
  seguir de GCD es la **organización de capítulos** y el énfasis en datos (§5), no otra clase.
- **Transparencia 8 de la Parte III** (guion de diapositivas para la defensa) **cambia para GCD** —
  refleja secciones de un proyecto de datos (dataset, EDA, modelado, métricas/evaluación) en vez
  del esquema software-céntrico. Toma la versión GCD.
- **Énfasis propio de Ciencia de Datos** (lo que los vídeos no recalcan): uso continuado de
  **visualizaciones, tablas y gráficas**; **narrativa de datos** (transmitir la curiosidad por
  encontrar patrones); **sesgo / fairness** en datos y modelos (detectarlo o paliarlo); y
  **lenguaje inclusivo**. Esto puntúa.
- Los comentarios sobre **másteres** (Parte III) no aplican al GCD; consultar al director académico
  del grado si interesa cursar uno.

---

## 1. Estructura de la memoria (esqueleto recomendado)

Genérico y transferible a GCD (el ejemplo GII se descarta). Extensión recomendada **50–100 páginas**.

**Preliminares (no van en el índice):** portada · agradecimientos (≤1 página) · resumen/abstract en
3 idiomas + palabras clave · listados de siglas y de figuras/tablas · índice general (+ índice de
figuras y de tablas).

**Cuerpo:**
1. **Introducción** — el *qué* y el *por qué*: motivación, justificación, objetivos, alcance y
   límites; cierra describiendo la estructura de la memoria. Redacción impersonal o plural; frases
   cortas; **sin notas al pie ni referencias** en la introducción.
2. **Estado del arte / de la cuestión** — síntesis selectiva, ordenada y **crítica** de la
   literatura reciente. (Para nosotros: las dos familias de métricas, mapean a H3/H5.)
3. **Fundamentos teóricos / marco teórico** — contexto teórico bien definido.
4. **Metodología** — población/muestra, variables, instrumentos y **procedimiento** por fases.
5. **Desarrollo** — análisis del problema, diseño de la solución, propuesta. No volcar grandes
   bloques de código, solo lo relevante/novedoso.
6. **Resultados** — texto corto + tablas/figuras numeradas y nombradas. (Bloque que espera a los
   runs de septiembre.)
7. **Conclusiones** — **no es un resumen**: interpreta, generaliza, **una conclusión por objetivo**,
   coherencia objetivos↔conclusiones. Cerradas (lógicas) y abiertas (líneas futuras).
8. **Trabajos futuros** — p. ej. la componente de intervención (early stopping) fuera de alcance.
9. **Referencias bibliográficas** — solo lo citado, una sola norma.
10. **Anexos** — incluido el **anexo ODS** (§6) y la reflexión que pida la plantilla.

**Qué se puede escribir ya (sin resultados):** introducción, estado del arte, marco teórico y
metodología. Resultados, discusión y conclusiones esperan a los experimentos.

---

## 2. Formato, resumen y figuras

- **Resumen/abstract:** **200–500 palabras**, en **castellano, valenciano e inglés** (es público en
  RiuNet). Da tema, relevancia, qué se hizo y cómo, herramientas, resultados y conclusiones.
  Escribir **al final**. Presente / pretérito perfecto / impersonal.
- **Palabras clave:** **3–10**, en los mismos tres idiomas.
- **Tipografía** (la maneja la clase LaTeX, pero como referencia): Arial o Times New Roman;
  epígrafes 16 pt, cuerpo 10–11 pt, notas/figuras 9–10 pt; interlineado sencillo o 1,5.
- **Figuras y tablas:** solo las que aclaran; **toda** ilustración con numeración, leyenda y
  fuente/autoría; coméntalas en el texto. *Cursiva* para palabras extranjeras y resaltes.
- **Título:** conciso y analítico; **sin** punto final, comillas, subrayado, negrita, acrónimos ni
  inicial de cada palabra en mayúscula. (Nuestro `\title{}` sigue en placeholder.)

---

## 3. Estilo y lengua académica

- **Impersonalidad:** plural de modestia ("proponemos"), construcciones con "se", infinitivo
  integrado ("cabe destacar", "hay que…"). Evita el "yo".
- **Precisión cuantitativa** (clave en Ciencia de Datos): nada de "muchos"/"la mayoría" →
  "17 experimentos", "56 % de los casos". Sustituye vaguedades por cifras.
- **Errores frecuentes a cazar:** *sobre todo* vs *sobretodo*, *en torno* vs *entorno*, *a parte*
  vs *aparte*, "en base a", "a nivel de", redundancias, repetición de conectores; tras dos puntos,
  minúscula; números bajos en letra.
- **Extranjerismos:** comprueba en el DLE si van en cursiva (*hardware*) o adaptados en redonda
  (*clicar*). Fuentes: DLE, Diccionario panhispánico de dudas, Fundéu, WordReference.
- La ortografía/puntuación correcta es condición *sine qua non*: corrector automático **y** revisión
  personal.

---

## 4. Citas y referencias

- **Elige una norma y sé consistente.** Para ciencia/tecnología encaja **ISO 690** (o un estilo
  numérico tipo IEEE — es lo que ya genera nuestro `biblatex` con `style=numeric`).
- Dos sistemas: **autor-fecha** (Apellido, año, p. X) o **numérico** ([1] entre corchetes). No los
  mezcles.
- Distingue **cita** (marca en el texto) de **referencia** (entrada completa al final). Cita corta
  (<40 palabras) entre comillas en el cuerpo; cita larga (≥40) en párrafo aparte, sin comillas,
  cuerpo menor; omisiones con `[...]`.
- **Gestor:** Zotero o Mendeley (cuenta institucional UPV). Biblioguías UPV:
  `biblioguias.webs.upv.es/bg` (gestores y "cómo citar").
- **Todas** las referencias deben estar citadas en el texto (recordatorio: quitar `\nocite{*}` de
  `main.tex` cuando el cuerpo cite de verdad).

---

## 5. Énfasis específico de Ciencia de Datos (puntúa)

- Apóyate de forma continuada en **visualizaciones, tablas y gráficas** de datos y modelos.
- **Narrativa de datos:** cuenta el proceso de exploración, transmite curiosidad por los patrones y
  los hallazgos reseñables, no solo el resultado final.
- **Sesgo / equidad:** discute si hay *bias* en datos o modelos y cómo se mitiga (en nuestro estudio
  encaja con la paradoja de Simpson entre datasets y el análisis por condición).
- **Lenguaje inclusivo** en toda la memoria y la presentación.

---

## 6. Anexo ODS (obligatorio — hay que escribirlo)

- **Qué es:** reflexión sobre la relación del TFG con los **17 Objetivos de Desarrollo Sostenible**
  (Agenda 2030). **Obligatorio** desde las convocatorias de 2022.
- **Dónde:** como **anexo** al final de la memoria **y** como **fichero suelto** que se sube a Ebrón
  al depositar.
- **Plantilla oficial** (Word/ODT) en el sitio de PoliformaT de TFG y en la web de la ETSINF.
  Úsala tal cual y **borra el texto de instrucciones** antes de entregar.
- **Estructura:** (1) **tabla de los 17 ODS** marcando para cada uno **Alto / Medio / Bajo / No
  procede**; (2) **reflexión en prosa de ~500–1500 palabras** sobre el/los ODS más relacionados (o
  por qué no procede).
- **Para este TFG**, candidatos naturales a justificar: **ODS 9** (industria, innovación e
  infraestructuras), **ODS 12** (consumo responsable — eficiencia/coste de cómputo del
  entrenamiento), **ODS 4** (educación) y, si se mide huella de cómputo, **ODS 13/7** (clima/energía).
  Encaja bien con el ángulo "predecir eficiencia del entrenamiento para gastar menos cómputo".

---

## 7. Competencias transversales (las evalúan)

13 competencias UPV (CT1–CT13), evaluadas por **tutor y tribunal** en escala **A/B/C/D**. Cómo
demostrarlas en memoria/defensa, en corto:

- **CT1 Comprensión/integración** — objetivos claros + diagramas.
- **CT2 Aplicación práctica** — estado del arte con fuentes de calidad, indicadores de éxito.
- **CT3 Análisis y resolución** — abordaje sistemático, decisiones justificadas, validación.
- **CT4 Innovación/emprendeduría** — valor, mejora sobre lo existente, limitaciones.
- **CT5 Diseño y proyecto** — alcance y límites (≈300–360 h), riesgos; cronograma de dedicación.
- **CT6 Trabajo en equipo** — qué hiciste tú si fue colaborativo.
- **CT7 Ética/medioambiental** — normas (protección de datos), procedencia de todo material ajeno,
  deontología; **se enlaza con los ODS**.
- **CT8 Comunicación efectiva** — memoria bien estructurada; en defensa **no leer** las diapositivas.
- **CT9 Pensamiento crítico** — juicio sobre el estado de la técnica; conclusiones que sinteticen.
- **CT10 Problemas contemporáneos** — impactos sociales/económicos/sostenibilidad (ODS).
- **CT11 Aprendizaje permanente** — autonomía; aplicar algo nuevo.
- **CT12 Planificación/tiempo** — plan con fechas, entregas intermedias, terminar con antelación.
- **CT13 Instrumentación específica** — herramientas usadas y por qué; demo/vídeo si procede.

---

## 8. Revisión antes de entregar (checklist)

- Cumple normas y formato; encabezados y pies correctos; **paginación** (preliminares se cuentan
  pero no se numeran, o números romanos).
- Ortografía, puntuación y gramática (corrector + revisión personal); registro formal; léxico
  variado.
- **Coherencia interna:** cada apartado coherente con su título; introducción clara; conclusión con
  contenido propio (no resumen).
- **Citas/referencias:** todas las del texto están en la bibliografía y viceversa; una sola norma;
  procedencia de toda cita/fuente indicada.
- Notas al pie correctas (9–10 pt); cifras vs palabras; remisiones internas (`v.`/`véase`).

---

## 9. Depósito (procedimiento)

- **TFG = 12 ECTS**, ~300–360 h. Trabajo original con tutor.
- **Antiplagio TURNITIN obligatorio** antes del depósito: tarea en **PoliformaT** (sitio "Tfg").
  % aceptable ETSINF **0–25 %**. Hazlo con antelación.
- **Depósito por EBRÓN.** Se sube **memoria en PDF + informe de TURNITIN + fichero ODS**.
- **La portada la genera EBRÓN automáticamente** — no la hagas tú.
- Consigue el **visto bueno del tutor** sobre la versión final antes de subir. (Se puede presentar
  sin informe favorable, pero el tutor emite un informe al tribunal.)
- **Plazos** (aprobación de título, depósito, defensa):
  <https://www.inf.upv.es/www/etsinf/es/fechas-importantes-tfg/>.

---

## 10. Defensa

- **Tiempo:** ~**20 min** de exposición (máx. ~45 min con preguntas del tribunal). *Confirmar el
  tiempo exacto en la convocatoria.* Si te pasas, salta a **conclusiones**: nunca dejes la
  presentación sin cerrar.
- **Diapositivas:** ≈**1 por minuto**, concisas, contraste fondo/letra, tipografía constante,
  resalta lo esencial. Indica en cada una el epígrafe del índice. Demo/vídeo: reserva los últimos
  ~5 min.
- **Guion (versión GCD):** portada (con autor y tutor) → índice → introducción → motivación →
  estado del arte → metodología/datos → modelado → resultados/evaluación → conclusiones y futuro.
- **Tribunal:** 3 profesores, posiblemente de otra especialidad → **introduce fundamentos básicos**.
  Mira al tribunal (no a la pantalla); evita muletillas ("bueno", "vale", brazos cruzados, jugar con
  objetos). **Ensaya** y ajusta el tiempo; comprueba sala/proyector/TEAMS antes.
- Tras la defensa el tribunal levanta acta y puedes solicitar el título.

---

## 11. Enlaces oficiales

- Plantilla y estructura GCD: <https://www.inf.upv.es/www/etsinf/es/plantilla-tfg/>
- Fechas importantes TFG: <https://www.inf.upv.es/www/etsinf/es/fechas-importantes-tfg/>
- Competencias transversales UPV: <https://www.upv.es/contenidos/COMPTRAN/>
- Biblioguías (citar/gestores): `biblioguias.webs.upv.es/bg`
- TFGs de ejemplo: **RiuNet** (filtrar por titulación/tutor/idioma)

---

## Acciones inmediatas

1. Descargar la **plantilla/estructura de GCD** y confirmar el esqueleto de capítulos (§0–§1).
2. Empezar a redactar lo independiente de resultados: **introducción, estado del arte, marco
   teórico, metodología** (ver plan en `pending/`).
3. Bajar la **plantilla oficial del anexo ODS** y redactarlo (~500–1500 palabras, tabla de 17).
4. Fijar la **norma de citación** (numérico/IEEE ya activo en `biblatex`) y mantenerla.
5. Anotar los **plazos** de depósito/defensa de la convocatoria objetivo.
