# Auditoría de publicación — fall_n

Fecha: 2026-07-16 · Alcance: 18 módulos de `src/` (~117k líneas), raíz del
repositorio, scripts, documentación y estado de git. Método: 9 auditorías
paralelas módulo a módulo con rúbrica común (idioma/nomenclatura, código
muerto, conformidad concepts-sobre-CRTP, encapsulamiento/auto-contención,
riesgos legales/privacidad, cobertura de tests, banners).

## Veredicto global

**REQUIERE TRABAJO, acotado y mecánico.** La arquitectura es publicable: la
convención concepts-sobre-CRTP se cumple en TODOS los módulos sin excepción,
no hay CRTP fuera de mixins, el type-erasure es idiomático y la cobertura de
tests es amplia (146 tests). Los problemas son de higiene, no de diseño. Hay
UN bloqueante legal y un puñado de bloqueantes de primera impresión.

## P0 — Bloqueantes (resolver antes de cualquier publicación)

1. **Copyright de terceros trackeado en git**: `doc/ref/Ko_Bathe-2026.pdf`
   (7.4 MB), `.docx` (8.9 MB) y `.extracted.txt` (92 KB). Publicar el repo
   redistribuye la obra. Requiere `git rm` **y purga del historial**
   (git-filter-repo/BFG). Dejar solo la cita en `references.bib` y añadir
   `doc/ref/` al `.gitignore`.
2. **Submódulo `PhD_Thesis`** apunta a `github.com/sechavarriam/PhD_Thesis`.
   Verificar visibilidad/licencia; si es privado o embargado, retirar el
   submódulo (`git rm PhD_Thesis` + `.gitmodules`) antes de publicar.
3. **Ruta personal con username**: `src/post-processing/VTK/VTKdataContainer.hh:176`
   (`/home/sechavarriam/...` en bloque comentado). El archivo entero está
   DEPRECATED — eliminarlo resuelve esto y dos hallazgos más.
4. **`data/input/box.geo.opt`**: preferencias de Gmsh con rutas personales
   absolutas trackeadas. `git rm` + ignorar `*.geo.opt`.
5. **README.md**: 4 295 líneas de changelog interno con mojibake (29 casos de
   `Ã—`/`â€œ`). Reescribir a README de publicación (~200 líneas: qué es,
   dependencias, build msys2/ucrt64, ejemplos, tests, licencia) y mover el
   changelog a `CHANGELOG.md`.

## P1 — Mayores (dañan la primera impresión o la mantenibilidad)

### Documentación de variables de entorno (transversal, el mayor tema técnico)
~96 variables `KOBATHE_*` gobiernan física y solver sin documentación
centralizada:
- 6 en `materials/KoBatheConcrete3D.hh` (`:80,:874,:1117` + 3 macros):
  `ANCHOR_CAP_RATIO`, `CLOSURE_STIFF_FRAC`, `VISCOUS_RATCHET_BETA`,
  `MULTI_CRACK`, `CRACK_ANGLE_DEG`, `ANCHOR_FT` — **alteran resultados
  físicos silenciosamente**.
- ~90 en `validation/ReducedRCColumnContinuumBaseline.cpp:3912-5064`
  (familias `DYNAMIC/DYN_*`, `ARCLEN*`, `LMNEWTON/LM_*`, `TAO*`, `CA*`).
Acción: tabla única (doc/ o header dedicado) con nombre, default, rango y
efecto; idealmente migrar las de materials a configuración tipada.

### Namespace e includes
- Clases núcleo en el **namespace global**: `NonlinearAnalysis`,
  `LinearAnalysis`, `DynamicAnalysis`, `LoadControl`, `DisplacementControl`,
  `ArcLengthControl`, `CustomControl` (analysis) y todo `Benchmark.hh`
  (utils). Nombres genéricos → riesgo de colisión. Mover a `fall_n::`.
- Estilo de include inconsistente (raíz `src/...` vs relativo `../...`):
  `RegularizedNewtonContinuation.hh:40`, `TaoEnergyContinuation.hh:28`,
  `VTKModelExporter.hh:93`, `EnrichmentActivationPolicy.hh:25`,
  `algorithms/*`. Unificar (decisión única para todo el repo).
- Headers no auto-contenidos: `Model.hh` (6 includes estándar faltantes),
  `Quadrature.hh` (Eigen), `NonlinearSubModelEvolver.hh` (`<print>`),
  `ShellKinematicPolicy.hh` (MITC), `Domain.hh` (petsc.h),
  `StructuralFieldReconstruction.hh`.

### Capas y estado global
- **Dependencia invertida**: `reconstruction/NonlinearSubModelEvolver.hh:88`
  incluye `../validation/ReducedRCManagedXfemLocalModelAdapter.hh` (y
  devuelve un tipo de validation). El núcleo no debe depender de validation.
- **Mutación de la DB global de PETSc** desde librería:
  `SubModelSolver.hh:345,520` y `NonlinearSubModelEvolver.hh:2072`
  (`PetscOptionsSetValue(nullptr,...)`). Configurar SNES/KSP/PC localmente.
- `std::system` con concatenación sin sanitizar: `utils/PythonPlotter.hh:27`.

### Fugas de documentación interna
Comentarios con trazabilidad de bitácora/tesis que no significan nada para un
lector externo: `EnrichmentActivationPolicy.hh:4,19,21` ("Plan v2 §Fase 3.5",
"Cap. 79 hipótesis H1"), `LocalModelKind.hh:4,23,50`,
`NonlinearSubModelEvolver.hh:191,681,1073`; jerga coloquial en
`KoBatheConcrete3D.hh` ("ARREGLO C", "genes"). Los catálogos
`Computational*Catalog` (analysis) son metadatos de tesis → marcar como capa
de validación, no API núcleo.

### Archivos muertos a eliminar (consolidado, ~17)
| Archivo | Motivo |
|---|---|
| `src/graph/AdjacencyMatrix.hh` + `AdjacencyList.hh` | módulo entero muerto |
| `src/elements/element_geometry/BeamColumn_Euler.hh` | 100% comentado, include inexistente |
| `src/elements/NodalSection.hh`, `src/elements/Section.hh` | huérfanos, colisionan con section/ |
| `src/geometry/Simplex.hh`, `src/geometry/geometry.hh` | huérfanos |
| `src/model/DoF.hh`, `src/model/ModelBuilder.hh` | huérfano / cáscara comentada |
| `src/post-processing/VTK/VTKdataContainer.hh`, `VTKwriter.hh`, `VTKheaders.hh` | deprecados (uno con ruta personal) |
| `src/numerics/Tensor.hh` | clase entera comentada (aún lo incluye `Material.hh`) |
| `src/numerics/linear_algebra/{Vector,Matrix,LinalgOperations}.hh` | wrappers PETSc "to be removed" |
| `src/materials/update_strategy/non-lineal/FullImplicitBackwardEuler.hh` | vacío, huérfano, dir mal nombrado |
| `src/reconstruction/FiberToTrussMapper.hh` | huérfano |
| `src/utils/GeneralConcepts.hh`, `unique_void_ptr.hh` | huérfanos |
| `doc/ch82_cyclic_validation.tex.old` | backup trackeado |

### Otros mayores
- `validation/ReducedRCColumnContinuumBaseline.cpp` (5 444 líneas): partir por
  costuras — IO (CSV/VTK), cinemática (`AffineTopCapDofTie`), y un TU por
  variante de solver (dyn/arclen/LM/TAO/CA), dejando el orquestador delgado.
- ~4k líneas duplicadas entre 5 estudios de sensibilidad de validation →
  `StudyCommon.hh`; 3 targets CMake clonados del continuum benchmark →
  colapsar en 1.
- Bundles de referencia OpenSees referenciados y ausentes del repo
  (`ReducedRCColumnEvidenceClosureCatalog.hh:166`) → incluir o documentar
  obtención (reproducibilidad).
- Parser Gmsh: 8 stubs vacíos + bloques comentados (`ReadGmsh.hh:186-243`) →
  implementar o declarar subconjunto soportado ("v4.1 ASCII, secciones X").
- Estado público crudo: `Lagrange/Serendipity/SimplexElement` (tag_, points_,
  nodes_...), `Domain::mesh`, `Mesh::dm`, `Node::sieve_id`,
  `ArcLengthControl::delta_u` (sin RAII) → encapsular.
- Test faltante del modelo insignia: `KoBatheConcrete3D` (1 934 líneas) sin
  suite unitaria dedicada (envolvente, ratchet, cierre, efecto de las 6
  env-vars). Ídem `VTKModelExporter` (2 361 líneas).
- Contratos duck-typed sin concept nombrado donde el header lo promete:
  `ElementGeometryLike` (elements), `ModelLike` (ContinuumElement),
  `NodeSelector` definido y nunca aplicado (`Model.hh:443`), políticas
  Shell/MITC/Assembly sin `requires`, contratos de export VTK.
- `-Werror` incondicional en CMake → condicionar a `FALL_N_WERROR` (off en
  release) para compiladores de terceros.
- `index.hh` (utils): manejo de errores con `throw bool` + `catch(bool)` y
  print a stdout → reemplazar.

## P2 — Menores (pulido)

- **Idioma**: identificadores 100% inglés en todo el repo (bien). Español en
  comentarios concentrado en pocos archivos; en 3 casos está en DOC de API
  pública (`RegularizedNewtonConfig::proximal_frac`, `AitkenRelaxation`,
  adapter bordered) → traducir esas primero; política única para el resto.
- **Banners**: faltantes en ~30 headers (núcleo de materials, model,
  post-processing grandes, numerics, xfem parcial, `MultiscaleAnalysis.hh` —
  la clase insignia de 2 319 líneas sin banner).
- `std::cout` en headers de librería (materials describe/print, mesh
  warnings) → `std::ostream&` inyectable.
- Typos en identificadores/guards: "Voigth", "Cuadrature", "Interfase",
  "Funtor", "swich", "efiiente", "Beeng".
- Scripts: 168 archivos, mayoría experimentos de un solo uso → separar
  `scripts/experiments/`; 6 scripts con `/c/msys64/ucrt64/bin` hardcodeado →
  parametrizar; salidas de `TableCyclicValidationAPI` escriben en el árbol de
  fuentes → redirigir a build.
- LICENSE GPLv3 compatible con PETSc/Eigen/VTK ✓; falta línea de copyright
  del autor. Confirmar que copyleft fuerte es la intención.
- 3 JSON generados force-added bajo `data/output/validation_reboot/` →
  decidir si son evidencia canónica o retirarlos.
- `main.cpp` stub trivial; los 26 `main_*.cpp` compilan todos → agrupar bajo
  `examples/` distinguiendo ejemplo didáctico vs driver interno.
- Duplicación `Owned*` en petsc/ → `OwnedHandle<T,Destroy>` (opcional).
- Cita autorreferencial "the author's paper" en
  `algorithms/cultural/KnowledgeSources.hh:12` → dejar cita neutra.

## Veredictos por módulo

| Módulo | Veredicto | Nota dominante |
|---|---|---|
| algorithms | LISTO CON MENORES | mejor módulo del lote |
| analysis | LISTO CON MENORES | namespace global, includes, ES en API doc |
| continuum | LISTO CON MENORES | adaptadores acoplados a materials |
| domain | LISTO CON MENORES | bloque muerto, `mesh` público |
| elements | REQUIERE TRABAJO | huérfanos, estado público, concept faltante |
| fracture | LISTO CON MENORES | — |
| geometry | LISTO CON MENORES | 2 huérfanos |
| graph | REQUIERE TRABAJO | eliminar módulo |
| materials | REQUIERE TRABAJO | env-vars sin doc, test 3D faltante |
| mesh | REQUIERE TRABAJO | parser incompleto (8 stubs) |
| model | REQUIERE TRABAJO | huérfanos, no auto-contenido, concept sin usar |
| numerics | REQUIERE TRABAJO | linear_algebra deprecado, Tensor.hh muerto |
| petsc | LISTO CON MENORES | API homogénea pendiente |
| post-processing | REQUIERE TRABAJO | ruta personal, deprecados, exporter sin test |
| reconstruction | REQUIERE TRABAJO | capa invertida, fugas doc interna, PETSc global |
| utils | REQUIERE TRABAJO | namespace global, std::system, huérfanos |
| validation | LISTO CON MENORES* | *como apéndice de campaña; monolito 5.4k |
| xfem | LISTO CON MENORES | — |
| raíz/higiene | REQUIERE TRABAJO | P0.1-P0.5 |

## Orden de ataque sugerido

1. P0 completo (legal + privacidad + README). La purga de historial se hace
   una sola vez y al final de la limpieza, para purgar todo junto.
2. Barrido mecánico: borrar los ~17 muertos, mover namespaces, unificar
   includes, auto-contención (una tarde de trabajo + suite completa).
3. Tabla única de env-vars `KOBATHE_*` (+ decidir cuáles se promueven a
   config tipada).
4. Partición del monolito de validation y separación campañas/vs/librería
   (`campaigns/` o `examples/`).
5. Tests faltantes de las dos piezas insignia (KoBatheConcrete3D,
   VTKModelExporter) y de la ley cohesiva XFEM (tangente vs FD).
6. Pulido P2 (banners, idioma, typos, scripts).
