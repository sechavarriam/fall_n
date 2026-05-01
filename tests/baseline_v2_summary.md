# Baseline `multiscale-v2` — Snapshot inicial

Rama: `multiscale-v2` (creada desde `Validacion_ModelosLocales`)
Generado tras los primeros edits Fase 1 (scratch pre-alloc en `NLAnalysis` /
`DynamicAnalysis`). Sirve como baseline contra el cual se medirán todas las
intervenciones del plan `multiscale-v2`.

## Estado del build

- **Total ninja targets**: 346
- **Targets construidos**: 264 / 265 (excluyendo el target roto preexistente)
- **Falla preexistente (no relacionada con esta serie de edits)**:
  - `main_rc_beam_validation.cpp:111` — `RebarBar` se refactorizó a
    `{double ly, double lz, double area, double diameter, std::string group}`
    pero el helper `make_rebar_layout` aún emite `{int, int, double, "Rebar"}`.
  - Acción: documentar como deuda; **no** se aborda en Fase 1 (no afecta a
    `ctest`).

## Resultados `ctest`

- **116 tests configurados**
- **108 PASS / 8 FAIL** (93%)
- Tiempo total real: 385.62 s

### Fallas observadas

| #  | Test                                                            | Tipo     | Notas                                                                 |
|----|-----------------------------------------------------------------|----------|----------------------------------------------------------------------|
| 39 | `case5_frontier_probe`                                          | Crash    | `0xc0000409` (`__fastfail`) — investigar (posible stack corruption)  |
| 76 | `reduced_rc_column_moment_curvature_closure`                    | Failed   | **Bloqueante Fase 0**: cierre del closure η_M                        |
| 77 | `reduced_rc_column_moment_curvature_closure_matrix`             | Timeout  | Timeout 180 s; estudio largo                                         |
| 78 | `reduced_rc_column_node_refinement_study`                       | Timeout  | Estudio de sensibilidad N                                            |
| 79 | `reduced_rc_column_cyclic_node_refinement_study`                | Timeout  | Estudio cíclico N                                                    |
| 80 | `reduced_rc_column_cyclic_continuation_sensitivity_study`       | Timeout  | Sensibilidad continuación cíclica                                    |
| 81 | `reduced_rc_column_quadrature_sensitivity_study`                | Timeout  | Estudio quadrature {GL, Lobatto, GR-L, GR-R}                         |
| 82 | `reduced_rc_column_cyclic_quadrature_sensitivity_study`         | Timeout  | Estudio quadrature cíclico                                           |

Los 6 timeouts (#77–#82) son estudios long-running. Plan: re-ejecutar con
`--timeout 1800` (30 min) en una corrida dirigida; no son fallas verdaderas.

#76 (`...closure`) y #39 (`case5_frontier_probe`) son fallas reales que
deben diagnosticarse antes de avanzar a Fase 4.

## Etiquetas CTest disponibles

- `unit` (89 tests, 40.8 s·proc)
- `integration` (20 tests, 1324 s·proc)
- `slow` (7 tests, 143 s·proc)

**Falta crear** las etiquetas `phase0` … `phase6`, `validation_reboot`,
`phase3_structural_matrix` que el plan `multiscale-v2` requiere para
gobernar los gates por fase.

## Edits Fase 1 ya en branch

- `src/analysis/NLAnalysis.hh`: scratch buffers persistentes
  (`elem_dofs_scratch`, `elem_f_scratch`, `elem_K_scratch`) en `Context`.
  `FormResidual` y `FormJacobian` reutilizan capacidad entre iteraciones SNES.
- `src/analysis/DynamicAnalysis.hh`: scratch buffers persistentes
  (`elem_dofs_scratch`, `elem_f_scratch`, `elem_K_scratch`) en `Context`.
  `I2Function` y `I2Jacobian` reutilizan capacidad entre pasos TS.

Cambio semántico: ninguno (sólo allocator). Esperado: reducción de heap-thrash
por iteración Newton/TS; ganancia escala con `num_elements × #iter`.
