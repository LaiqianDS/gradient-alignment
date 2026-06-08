## Hub
- [[Planificacion TFG]] — plan 10 semanas + estado actual + cola de lectura + backlog
- [[Estado TFG]] — pregunta, hipótesis, diseño cerrado, riesgos, decisiones abiertas
- [[EBRON]] — propuesta

## Recordatorios
- Anexo: reflexión ODS
- Tutor evalúa parte de la nota
- [Notas redacción TFG (UPV)](https://poliformat.upv.es/access/content/group/GRA_14056_2025/Seminario%20Redacción%20y%20Defensa%20del%20TFG/3_Trabajo%20Final%20de%20Grado.pdf)

## Pregunta de investigación
![[Estado TFG#Pregunta de investigación]]

## Hipótesis operativa
![[Estado TFG#Hipótesis operativa]]

## Decisiones cerradas
Fuente de verdad única: [[Estado TFG]]. Se muestran aquí por transclusión (no se copian).

![[Estado TFG#Diseño experimental]]

![[Estado TFG#Protocolo de análisis]]

## Procedimiento
![[Estado TFG#Procedimiento]]

## Papers
```dataview
TABLE authors, year, status, relevance, tfg_role AS "Rol TFG", reading_order AS "Orden", thesis_rank AS "Rank", link(url, "Link") AS "Paper"
FROM "docs/research/Papers"
SORT reading_order ASC
```

## Conceptos
```dataview
TABLE file.mtime AS "Editado"
FROM "docs/research/Conceptos"
SORT file.mtime DESC
```

## Papers relacionados
```dataview
LIST
FROM "docs/research/Papers"
SORT reading_order ASC
```
