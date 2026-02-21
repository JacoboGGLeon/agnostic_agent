---
name: "skills_reader"
description: "Explora skills: listar y leer definiciones completas."
tools:
  - list_skills
  - read_skill
knowledge: []
---

# Skills Reader

Objetivo: permitir al usuario listar skills y leer una skill en detalle (descripción, tools/knowledge permitidas e instrucciones).

Reglas:
- Cuando el usuario pida un listado, usa `list_skills`.
- Cuando el usuario pida el detalle de una skill, usa `read_skill` con el nombre exacto.
