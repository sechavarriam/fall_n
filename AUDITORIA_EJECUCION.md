# Ejecución de la auditoría de publicación — estado por ítem

Actualizado: 2026-07-19. Ejecuta el plan de `AUDITORIA_PUBLICACION.md`.
Rama de trabajo `feature/algorithms-metaheuristics`.

## Bloque 1 — P0 no destructivos

| Ítem | Estado | Commit / nota |
|---|---|---|
| Ruta personal en `VTKdataContainer.hh:176` | **HECHO** | `0b2634b` retira el bloque comentado completo. El archivo sigue siendo candidato a eliminación (ver tabla de muertos). |
| `box.geo.opt` trackeado | **HECHO** | `a562a3f` lo destrackea junto con `box1.msh` e `Ibeam_default.msh` (mallas generadas sin referencia en src/tests/scripts). `.gitignore` ahora excluye `*.geo.opt`. Los archivos siguen en disco. |
| README 287 KB con mojibake | **HECHO** | `f35e3a6`. README de publicación nuevo; historial íntegro en `CHANGELOG.md` con 87 clústeres de mojibake reparados. |
| PDFs Ko_Bathe trackeados | **PENDIENTE DE APROBACIÓN** | Plan de purga abajo. Sin ejecutar nada. |

## Bloque 2 — Documentación de configuración

| Ítem | Estado | Commit / nota |
|---|---|---|
| Variables `KOBATHE_*` sin documentar | **HECHO** | `f7b0211` crea `docs/kobathe_env_vars.md`: 97 entradas (3 macros + 94 env) con tipo, default, efecto y módulo lector. |

## Bloque 3 — Estructura (P1)

| Ítem | Estado | Nota |
|---|---|---|
| Namespaces núcleo → `fall_n` | pendiente | |
| Dependencia invertida reconstruction→validation | pendiente | análisis y propuesta, sin movimientos grandes |
| ~17 archivos muertos | pendiente | verificación uno a uno antes de proponer |
| Mutación global de opciones PETSc | pendiente | |
| Plan de corte del monolito (5.4k líneas) | pendiente | solo plan, sin ejecutar |

## Decisiones que quedan del autor

1. **Purga de historia** (plan abajo): aprobar y ejecutar una sola vez, al
   final de toda la limpieza.
2. **Submódulo `PhD_Thesis`** (P0.2): verificar visibilidad/licencia del
   repo remoto antes de publicar; retirarlo si es privado o embargado. No
   se tocó (el autor trabaja en él en paralelo).
3. **`doc/ch82_cyclic_validation.tex.old`** y figuras PDF bajo `doc/`:
   dentro de `doc/`, intocado por regla de esta sesión. El `.tex.old` es
   candidato obvio a `git rm` cuando el autor lo confirme.
4. **Licencia**: GPLv3 sin línea de copyright del autor. Confirmar copyleft
   fuerte y añadir `Copyright (C) <años> Sebastián Echavarría Montaña` si
   es la intención.
5. **JSONs force-added** en `data/output/validation_reboot/`: decidir si
   son evidencia canónica o se retiran del índice.

---

## Plan de purga de historia (P0.1) — PREPARADO, NO EJECUTADO

### Qué purgar

| Ruta | Motivo | Entró en |
|---|---|---|
| `doc/ref/Ko_Bathe-2026.pdf` (7.4 MB) | copyright de terceros | `e2c8e17` |
| `doc/ref/Ko_Bathe-2026.docx` (8.9 MB) | copyright de terceros | `d8038fb` |
| `doc/ref/Ko_Bathe-2026.extracted.txt` (92 KB) | texto extraído de la obra | — |
| `data/input/box.geo.opt` | rutas personales absolutas | `51a2276` |
| `data/input/box1.msh`, `data/input/Ibeam_default.msh` | opcional: solo bulk (54k líneas), sin problema legal | — |

### Datos del repo relevantes

- Remoto único: `origin = github.com/sechavarriam/fall_n.git`.
- Pack actual: **224.8 MiB**; la purga de `doc/ref` recorta ~16.5 MB.
- Hay múltiples ramas locales y remotas; `git filter-repo` reescribe TODAS
  las refs del clon donde corre.
- `git-filter-repo` **no está instalado** en esta máquina
  (`pip install git-filter-repo`).
- Hay al menos un worktree adicional activo (`fix/kobathe-tau-octahedral`);
  filter-repo debe correr en un **clon espejo fresco**, nunca en el árbol
  de trabajo con la tesis y las campañas en vuelo.

### Comandos preparados (ejecutar SOLO tras aprobación y al final de la limpieza)

```powershell
# 0) Prerrequisito (una vez)
pip install git-filter-repo

# 1) Paso previo en el árbol de trabajo: retirar del HEAD y citar en BibTeX
git rm --cached doc/ref/Ko_Bathe-2026.pdf doc/ref/Ko_Bathe-2026.docx doc/ref/Ko_Bathe-2026.extracted.txt
Add-Content .gitignore "`ndoc/ref/"
git commit -m "doc: retira la obra de terceros del indice; queda la cita en references.bib"

# 2) Clon espejo fresco (la purga corre aquí, no en c:\MyLibs\fall_n)
git clone --mirror https://github.com/sechavarriam/fall_n.git C:\tmp\fall_n_purge.git
cd C:\tmp\fall_n_purge.git

# 3) Purga (todas las refs)
git filter-repo --invert-paths `
  --path doc/ref/Ko_Bathe-2026.pdf `
  --path doc/ref/Ko_Bathe-2026.docx `
  --path doc/ref/Ko_Bathe-2026.extracted.txt `
  --path data/input/box.geo.opt

# 4) Verificación en el espejo purgado
git count-objects -vH
git log --all --oneline -- doc/ref/   # debe salir vacío

# 5) Publicación (DESTRUCTIVO: reescribe el remoto)
git push --force --all
git push --force --tags
```

### Impacto (leer antes de aprobar)

- **Cambian los SHA de todos los commits** desde `e2c8e17` en adelante, en
  todas las ramas que contienen esos archivos (incluye `master`,
  `entrega-final`, esta rama y las históricas).
- Todo clon existente (otras máquinas, CI, el worktree paralelo) queda
  divergente y debe **re-clonarse**; no hacer `pull` sobre clones viejos.
- Los hashes citados en bitácoras/tesis que referencien commits antiguos
  dejarán de resolver en el remoto.
- El submódulo `PhD_Thesis` no se ve afectado (repo aparte); el gitlink
  conserva su SHA.
- Recomendación operativa: ejecutar cuando no haya campañas en vuelo,
  re-clonar el working tree después, y re-aplicar los worktrees.
