# Ejecución de la auditoría de publicación — estado por ítem

Actualizado: 2026-07-21. Ejecuta el plan de `AUDITORIA_PUBLICACION.md`.
Rama de trabajo `feature/algorithms-metaheuristics`.

**Estado de la suite: 145/145 en verde** (`ctest --timeout 3000`, última
corrida completa sobre el HEAD `9aff9c4` con todo el barrido P1/P2; ~100 min).
Build completo (275 targets, `-Werror`) en verde tras cada lote. Sin push:
todos los commits son locales.

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
| Namespaces núcleo → `fall_n` | **HECHO** | `e1f0937`: NLAnalysis, LinearAnalysis, DynamicAnalysis, IncrementalControl y Benchmark envueltos en `fall_n`, ~200 sitios calificados en 50 archivos. Suite completa 145/145 en verde sobre ese árbol. De paso `48e2051` corrige una rotura preexistente del tuner CA (rename `genome`→`traits` de `bdae433`). |
| Dependencia invertida reconstruction→validation | **HECHO** | `89234c7`: `LocalModelVTKSnapshot.hh` nuevo en reconstruction; el adapter de validation conserva el nombre histórico como alias; el evolver ya no incluye validation. |
| ~17 archivos muertos | **HECHO** | `49db8c6` elimina **20** archivos verificados (los 17 del plan + 3 extra hallados) y limpia el umbrella legacy `header_files.hh` que incluía 6 de ellos. Aprobado por el autor ("purga de lo inútil"). Detalle de verificación abajo. |
| Mutación global de opciones PETSc | **HECHO** | `89234c7`. En `SubModelSolver.hh` (2 sitios) las opciones globales eran letra muerta para el solve local (el perfil tipado de `solve_incremental` las pisaba en cada intento) — se eliminaron sin cambio de comportamiento. En `NonlinearSubModelEvolver.hh` se reemplazaron por configuración directa del SNES propio tras `SNESSetFromOptions`, preservando la precedencia previa. Quedan fuera de alcance y documentadas: las opciones prefijadas de sub-solver en `NonlinearSolvePolicy.hh` (namespaced por prefijo del KSP, activas solo bajo perfil explícito) y `utils/SolverConfig.hh` (utilidad opt-in usada solo por `test_benchmark`). |
| Plan de corte del monolito (5.4k líneas) | pendiente | solo plan, sin ejecutar |

### Archivos muertos — verificación (2026-07-19)

Los 17 candidatos de la tabla original tienen **cero referencias** en
`src/`, `tests/`, los `main_*.cpp`, `CMakeLists.txt` y `scripts/`. Los dos
falsos positivos del grep se descartaron a mano (`ModelBuilder` casaba con
`StructuralModelBuilder.hh`, que es otro archivo y sí vive; `Section.hh`
colisiona con el módulo `section/`, por eso se buscó la ruta
`elements/Section.hh` completa). Los tres includes vestigiales que aún los
mencionaban se retiraron en `d5d1520` (suite de materiales 6/6 verde).

Grupo A — basura evidente (cascarón comentado, vacío, o con banner
DEPRECATED/`Deprecated*` propio):

```
src/graph/AdjacencyMatrix.hh                                  (0 líneas de código)
src/elements/element_geometry/BeamColumn_Euler.hh             (clase 100% comentada)
src/post-processing/VTK/VTKdataContainer.hh                   (banner DEPRECATED)
src/post-processing/VTK/VTKwriter.hh                          (vacío)
src/post-processing/VTK/VTKheaders.hh                         (solo includes; único usuario: VTKdataContainer)
src/numerics/Tensor.hh                                        (clase comentada, helpers sin uso)
src/numerics/linear_algebra/Vector.hh                         (clases Deprecated*, "to be removed")
src/numerics/linear_algebra/Matrix.hh                         (ídem)
src/numerics/linear_algebra/LinalgOperations.hh               (opera solo sobre los Deprecated*)
src/materials/update_strategy/non-lineal/FullImplicitBackwardEuler.hh  (vacío)
```

Grupo B — huérfanos con código real (cero referencias, pero contenido no
trivial que el autor podría querer conservar):

```
src/graph/AdjacencyList.hh
src/elements/NodalSection.hh
src/elements/Section.hh
src/geometry/Simplex.hh
src/geometry/geometry.hh
src/model/DoF.hh
src/model/ModelBuilder.hh
src/reconstruction/FiberToTrussMapper.hh
src/utils/GeneralConcepts.hh
src/utils/unique_void_ptr.hh
```

Comando preparado (un solo commit dedicado, tras aprobación):

```powershell
git rm src/graph/AdjacencyMatrix.hh src/graph/AdjacencyList.hh `
  src/elements/element_geometry/BeamColumn_Euler.hh `
  src/elements/NodalSection.hh src/elements/Section.hh `
  src/geometry/Simplex.hh src/geometry/geometry.hh `
  src/model/DoF.hh src/model/ModelBuilder.hh `
  "src/post-processing/VTK/VTKdataContainer.hh" `
  "src/post-processing/VTK/VTKwriter.hh" `
  "src/post-processing/VTK/VTKheaders.hh" `
  src/numerics/Tensor.hh `
  src/numerics/linear_algebra/Vector.hh `
  src/numerics/linear_algebra/Matrix.hh `
  src/numerics/linear_algebra/LinalgOperations.hh `
  "src/materials/update_strategy/non-lineal/FullImplicitBackwardEuler.hh"
git commit -m "src: elimina 17 archivos muertos verificados (cero referencias)"
```

Nota: `doc/ch82_cyclic_validation.tex.old` (backup trackeado) también es
candidato, pero vive en `doc/` — intocado por regla de esta sesión.

### Dependencia invertida reconstruction→validation — análisis

El único punto de contacto es `NonlinearSubModelEvolver.hh:88`, que incluye
`validation/ReducedRCManagedXfemLocalModelAdapter.hh` (2 274 líneas) solo
para nombrar el struct de valor `ReducedRCManagedXfemLocalVTKSnapshot`
(retorno del contrato duck-typed `write_vtk_snapshot`, compartido por los
cuatro modelos locales: el adapter XFEM, `ManagedXfemSubscaleEvolver`,
`SeismicFE2LocalModelVariant` y el evolver de reconstruction). El evolver
no usa ningún otro símbolo del header — sus demás dependencias ya son
propias.

Propuesta (mínima, sin mover los modelos locales de validation):

1. Nuevo header `src/reconstruction/LocalModelVTKSnapshot.hh` con el struct
   renombrado `LocalModelVTKSnapshot` (mismos campos), en `fall_n`.
2. El adapter de validation lo incluye y conserva
   `using ReducedRCManagedXfemLocalVTKSnapshot = LocalModelVTKSnapshot;`
   para no tocar los ~10 sitios de uso existentes.
3. El evolver cambia el include de validation por el header nuevo y usa el
   nombre neutro. Con eso `reconstruction` deja de depender de
   `validation` por completo (dirección correcta: validation→reconstruction).

### Plan de corte del monolito `ReducedRCColumnContinuumBaseline.cpp` (5 720 líneas) — SOLO PLAN

Estructura actual: helpers en namespace anónimo (57–2892: trazas de
control, sumarios de refuerzo, precarga axial, sondas cinemáticas,
snapshots VTK, escritores CSV, fábricas de materiales, extractores de
registros), `describe_*` públicos (2895–2979), el gigante
`run_reduced_rc_column_continuum_case_result_impl` (2980–5670, con las
ramas env-gated `KOBATHE_DYNAMIC` 3911–4360, `KOBATHE_ARCLEN` 4370–4590,
`KOBATHE_LMNEWTON`+TAO+CA 4593–5430 y el camino cuasi-estático por defecto)
y wrappers públicos (5672–5720).

Orden de extracción propuesto (cada paso: compilar
`fall_n_reduced_rc_column_continuum_baseline_test` + benchmark, correr el
test, y un smoke corto de la rama tocada):

1. **IO** → `ReducedRCColumnContinuumIO.{hh,cpp}`: escritores CSV
   (1886–2173), snapshots VTK (1462–1885) y extractores de registros
   (2610–2891). Sumideros puros; riesgo bajo. Requiere mover los structs de
   registro a un header de detalle compartido.
2. **Materiales** → `ReducedRCColumnContinuumMaterials.{hh,cpp}`: perfiles
   de concreto, fábricas Ko-Bathe/rebar, descriptores de confinamiento
   (605–621, 2174–2609) + los `describe_*` públicos.
3. **Cinemática** → header propio para `AffineTopCapDofTie` y los ties de
   tapa (candidato a `analysis/` como utilidad reusable de amarres DOF).
4. **Un TU por variante de solver**, extrayendo cada rama env-gated con un
   `ContinuumSolveContext` (struct de referencias: nl, model, spec, cfg,
   couplings, ties, recorders, out_dir):
   `...DynamicRelaxation.cpp`, `...ArcLength.cpp`, `...LMContinuation.cpp`
   (incluye el híbrido TAO), `...CATuner.cpp` (replay CA; depende del TU de
   LM). El orquestador queda delgado: parsing del spec, construcción del
   modelo, acoples, scheme de control, camino cuasi-estático y postproceso.

Riesgos y por qué NO se ejecuta ahora: las ramas capturan ~30 locales por
referencia (el `ContinuumSolveContext` es mecánico pero ancho), la rama CA
reenvía el estado interno del solver LM (checkpoint/restore), y el archivo
sustenta campañas de tesis ACTIVAS. Condición de entrada sugerida: harness
de golden-run (correr el smoke de 50 mm antes/después y diff byte a byte de
los CSV) y ventana sin campañas en vuelo.

## Bloque 4 — Barrido P1/P2 (2026-07-20/21)

| Ítem | Estado | Commit |
|---|---|---|
| `-Werror` incondicional → `FALL_N_WERROR` (ON por defecto) | **HECHO** | `98b6e54` |
| Fugas de doc interna ("Plan v2 §Fase", "Cap. 79 H1", "bitácora", jerga "ARREGLO C"/"genes") | **HECHO** | `e69b42e` (7 headers + KoBathe) |
| Español en DOC de API pública → inglés (`proximal_frac`, `AitkenRelaxation`, adapter bordered) | **HECHO** | `e69b42e` |
| Cita autorreferencial en `KnowledgeSources.hh` neutralizada | **HECHO** | `e69b42e` |
| Typos: `Funtor`→`Functor`, guard `CUADRATURE`→`QUADRATURE`, "efiiente", backslash suelto | **HECHO** | `e69b42e`, auto-contención |
| `index.hh`: `throw bool`/`catch(bool)` + stdout → `if` + stderr | **HECHO** | `b968ddc` |
| `PythonPlotter`: `std::system` endurecido (rechaza comillas/newline) | **HECHO** | `b968ddc` |
| Headers no auto-contenidos | **HECHO** | `Quadrature.hh` (+Eigen), `ShellKinematicPolicy.hh` (+ElementGeometry, +MITCShellPolicy), `StructuralFieldReconstruction.hh` (+MITCShellElement). `Model.hh`, `Domain.hh` y `NonlinearSubModelEvolver.hh` ya eran auto-contenidos (verificado con `-fsyntax-only` en aislamiento). |
| Banner de `MultiscaleAnalysis.hh` (clase insignia sin banner) + `ReadGmsh.hh` | **HECHO** | + declaración del subconjunto Gmsh soportado (MSH 4.1 ASCII) y borrado del bloque de ejemplo muerto tras el `#endif` |

### Ítems P1/P2 NO ejecutados (con razón)

- **Encapsular estado público crudo** (`Lagrange/Serendipity/SimplexElement`,
  `Domain::mesh`, `Mesh::dm`, `Node::sieve_id`, `ArcLengthControl::delta_u`):
  refactor de acceso que toca todos los llamadores; riesgo alto sin revisión
  humana. Pendiente.
- **Banners en ~30 headers**: cosmético, alta rotación en headers muy
  incluidos; se hizo la clase insignia como muestra. El resto pendiente.
- **`std::cout` en headers de librería → `std::ostream&` inyectable**:
  alcance real medido = 118 ocurrencias en 22 headers (materials `describe`,
  parsers Gmsh `print_raw`, timers, etc.). Es un refactor amplio de firmas con
  matiz semántico (distinguir salida de diagnóstico legítima de errores); no
  es un cambio mecánico acotado. Pendiente de decisión de diseño del autor.
- **Política única de idioma para el resto de comentarios ES**: barrido
  amplio; el audit lo marca como P2 diferible. Solo se tradujo la doc de API
  pública.
- **Parser Gmsh (8 stubs)**: se declaró el subconjunto soportado en el
  banner; implementar los stubs es trabajo de feature, no de higiene.

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
