# TFG — Métricas de gradiente como predictores de eficiencia

**Empieza aquí.** Este vault tiene *un documento por trabajo*; léelos en orden:

1. **[[1 - Diseño]]** — el qué y el porqué: pregunta de investigación, hipótesis (H1–H6), diseño experimental, baselines, protocolo de análisis y confusores.
2. **[[2 - Decisiones]]** — lo que se va decidiendo: decisiones pendientes + log cronológico de las tomadas.
3. **[[3 - Progreso]]** — dónde estamos: plan semanal, estado actual, pasos inmediatos, cola de lectura y backlog.

Referencia (cambia poco): [[metrics]] · [[datasets]] · [[models]] · [[EBRON]] (propuesta).

## Recordatorios
- Anexo: reflexión ODS
- Tutor evalúa parte de la nota
- [Notas redacción TFG (UPV)](https://poliformat.upv.es/access/content/group/GRA_14056_2025/Seminario%20Redacción%20y%20Defensa%20del%20TFG/3_Trabajo%20Final%20de%20Grado.pdf)

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
