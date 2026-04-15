# fall_n

`fall_n` is a C++23 structural-analysis and multiscale research library focused on:

- finite elements for beams, shells, trusses, and 3D continua,
- constitutive models and local nonlinear integration,
- PETSc-backed nonlinear and dynamic analysis,
- beam-to-continuum multiscale coupling and FE2-style workflows,
- validation drivers and rich LaTeX-first technical documentation.

The repository is no longer just a collection of drivers. It already contains a real library architecture, a growing public multiscale surface, regression tests, and a large technical manuscript that documents both the theory and the implementation.

## Current Status

The most mature publication-path subsystem today is the multiscale module:

- `OneWayDownscaling`, `LaggedFeedbackCoupling`, and `IteratedTwoWayFE2` are now distinct and named honestly.
- The local 3D section response is injected through a single `SectionHomogenizedResponse` carrying `forces + tangent + strain_ref`.
- The sparse condensed tangent is validated against adaptive finite differences with configurable norm families.
- The FE2 outer loop now uses the same family of physically weighted norms for force/tangent residuals that the condensed-tangent validator uses.
- The persistent local facade has been split further internally through `LocalBoundaryConditionApplicator`, `PersistentLocalStateOps`, `BoundaryReactionHomogenizer`, `LocalCrackDiagnostics`, and `LocalVTKOutputWriter`.
- The local continuum-material contracts now live in `src/materials/SubmodelMaterialFactory.hh`, while the current Ko-Bathe/Menegotto reference defaults live in `src/materials/SubmodelMaterialFactoryDefaults.hh`; the multiscale core no longer owns those constitutive choices.
- A first generic subscale-model concept layer now exists in `src/analysis/SubscaleModelConcepts.hh`; the current `LocalModelAdapter` should now be read as a section-specialized FE2 contract rather than as the final abstraction level for every future local model.
- That abstraction step is no longer only conceptual: `src/reconstruction/AffineSectionSubscaleModel.hh` now provides a second, operator-driven local-model realization that satisfies both the generic subscale concepts and the current section-specialized multiscale path without requiring a local continuum solve.
- A first explicit continuum-semantics layer now exists in `src/continuum/ContinuumSemantics.hh`, together with `KinematicFormulationTraits<Policy>` in `src/continuum/KinematicPolicy.hh`; this encodes, at compile time, the description kind, assembly configuration, canonical conjugate pair, and current scientific maturity of each continuum formulation.
- That semantic lift is no longer limited to strain/stress labels: `BodyPoint`, `ReferencePoint`, `CurrentPoint`, `PlacementMap`, and `MotionSnapshot` now provide a first low-cost computational vocabulary for the Chapter 4 notions of body, configuration, placement, and motion, without introducing a runtime-heavy continuum object model.
- The same layer now also records formulation-level virtual-work semantics explicitly: `VolumeMeasureKind`, `VirtualWorkCompatibilityKind`, and `VirtualWorkSemantics` distinguish exact work-conjugate formulations (TL, UL), linearized-equivalent ones (small strain), and unaudited placeholders (continuum corotational).
- That semantic lift now also carries an explicit audit layer: `AuditEvidenceLevel` and `FormulationAuditScope` distinguish mathematical admissibility from implementation evidence, so the code can state at compile time whether a formulation is merely declared, regression-tested, reference-grade, or still in need of a finite-kinematics disclaimer.
- A second audit layer now separates formulation semantics from family-level deployment: `src/continuum/FormulationScopeAudit.hh` records, at compile time, whether a formulation is actually available for `continuum_solid_3d`, `beam_1d`, or `shell_2d`, whether it is only partial, and whether it is the current reference path for geometric nonlinearity in that family.
- That family-level audit is now queryable as structured data as well as traits: `canonical_family_formulation_audit_table()`, `canonical_family_formulation_audit_row(...)`, and the `find_family_*_reference_path(...)` helpers expose the same matrix that the thesis now uses as its normative reading of `family × formulation`.
- That family-level audit is no longer only documentary: `FamilyKinematicPolicyAuditTraits` and the `FamilyNormativelySupportedKinematicPolicy` / `FamilyReferenceGeometricNonlinearityKinematicPolicy` concepts now let element templates reject unsupported family/formulation combinations at compile time instead of relying on comments or downstream test failures.
- A third audit layer now separates "formulation available" from "solver route actually validated": `src/analysis/AnalysisRouteAudit.hh` records the current scope of `linear_static`, `nonlinear_incremental_newton`, `implicit_second_order_dynamics`, and `arc_length_continuation` for each audited `family × formulation` combination.
- That route-level layer is also exposed as structured compile-time data through `canonical_family_formulation_analysis_route_audit_table()`, `canonical_family_formulation_analysis_route_row(...)`, and route-count helpers, so the thesis can distinguish, truthfully, between a formulation that exists in an element family and a solver path that is actually reference-grade for that same scope.
- The representative `family × formulation × route` matrix now also has a canonical catalog in `src/analysis/AnalysisRouteCatalog.hh`. It is intentionally kept outside the main analysis umbrella so the thesis, the README, and the regression surface can reuse one audited route table without pushing extra metadata into the common hot-path include stack.
- The solver classes themselves now publish audited route tags: `LinearAnalysis`, `NonlinearAnalysis`, `DynamicAnalysis`, and `ArcLengthSolver` expose `analysis_route_kind` and `analysis_route_audit_scope`, which lets tests and documentation talk about solver maturity without introducing runtime polymorphism.
- A fourth audit layer now composes those claims over concrete C++ types: `src/analysis/ComputationalScopeAudit.hh` lets the code and the thesis talk about audited pairs such as `ContinuumElement<TotalLagrangian> + NonlinearAnalysis` or `BeamElement<Corotational> + NonlinearAnalysis`, instead of relying only on family-level statements.
- `ContinuumElement`, `BeamElement`, and `MITCShellElement` now publish `element_family_kind`, `formulation_kind`, and `family_formulation_audit_scope`, so the previous audit tables are no longer disconnected from the actual element types used in the library.
- A fifth audit layer now raises the same idea to real computational slices: `src/analysis/ComputationalModelSliceAudit.hh` composes `Model<...>` and solver types, so the code can distinguish between a merely admissible `element + route` combination and an actually instantiated solver/model slice such as `Model<TL> + NonlinearAnalysis<TL>`.
- That fifth layer now also classifies representative slices through `ComputationalModelSliceSupportLevel`: for example, `Model<small strain> + LinearAnalysis` and the current audited beam/shell small-rotation linear slices resolve to `reference_linear`, `Model<TL> + NonlinearAnalysis<TL>` resolves to `reference_geometric_nonlinearity`, `Model<UL> + NonlinearAnalysis<UL>` remains only `normative`, and slices such as `Model<TL> + DynamicAnalysis<TL>`, `Model<TL> + ArcLengthSolver<TL>`, beam corotational + nonlinear Newton, and shell corotational + nonlinear Newton are kept under `unsupported_or_disclaimed` until solver-level evidence catches up.
- That slice audit is now available as structured compile-time data as well: `make_model_solver_slice_audit_row(...)`, `count_model_solver_slice_support_level(...)`, and `count_model_solver_slices_requiring_scope_disclaimer(...)` let the regression surface and the thesis talk about the same representative `model + solver` table instead of maintaining that classification only as scattered `static_assert`s or prose.
- The representative `model + solver` table itself now lives in `src/analysis/ComputationalModelSliceCatalog.hh`, not only inside a regression test. That catalog is intentionally kept outside the main analysis umbrella so the thesis/documentation surface can reuse a canonical slice matrix without forcing extra audit metadata into the common hot-path include stack.
- The hard representative matrix `family x formulation x analysis route x typed computational slice` now also has a canonical source in `src/analysis/ComputationalSliceMatrixCatalog.hh`. This closes the gap between the route table and the slice table: the thesis, the README, and the regression surface can now talk about one combined audited matrix instead of manually reconciling two separate tables.
- A sixth semantic layer now lifts the virtual-work statements of Chapters 4 and 5 into discrete FEM carriers through `src/continuum/DiscreteVariationalSemantics.hh`. It records, at compile time, which kinematic/stress carriers each `family x formulation` pair integrates, over which discrete domain it assembles the internal work, where constitutive history is expected to live, and whether the tangent is canonically pointwise, sectional, or augmented by geometric terms.
- A seventh audit layer now lifts those discrete carriers up to the full typed computational slice through `src/analysis/ComputationalVariationalSliceAudit.hh` and `src/analysis/ComputationalVariationalSliceCatalog.hh`. For a concrete `Model + Solver` slice, the code can now state at compile time which global residual is being linearized, which tangent topology is assembled, how incremental state/history is managed, and whether structural effective-operator injection is even admissible for that slice.
- That variational-slice layer is no longer stabilized only indirectly through broader solver tests. A dedicated regression target, `fall_n_computational_variational_slice_catalog_test`, now freezes the coherence between the route catalog, the model-slice catalog, the hard combined slice matrix, and the discrete variational semantics. This matters scientifically because it prevents the thesis and the code from drifting apart on what residual/tangent/history object a representative slice actually denotes.
- A further traceability layer now ties representative scientific claims directly to typed slices, residual/tangent/history commitments, and evidence channels through `src/analysis/ComputationalClaimTraceCatalog.hh`. This makes the pre-validation status explicit in a scientifically stricter sense: we now freeze not only which slice supports a claim, but also which nonlinear problem that claim is allowed to denote, how its tangent is interpreted, and how internal history is committed before any physical-validation campaign is claimed.
- `Model<>`, `LinearAnalysis`, `NonlinearAnalysis`, `DynamicAnalysis`, and `ArcLengthSolver` now expose the minimal type aliases needed for that audit (`element_type`, `model_type`, audited family/formulation tags), keeping the information in compile time metadata instead of rebuilding it in runtime registries.
- One useful consequence of that stricter audit is that some structural claims are now intentionally more conservative than the formulation matrix alone: beam and shell corotational kinematics remain real and valuable, but the corresponding global solver routes are not all promoted to the same baseline status until solver-level evidence catches up.
- That semantic lift also freezes a more honest finite-kinematics status: continuum `TotalLagrangian` is the current reference path, continuum `UpdatedLagrangian` is real but still classified as partial, and continuum `Corotational` remains a placeholder distinct from the implemented corotational beam/shell paths.
- Beam and shell kinematic policies now expose parallel family-audit traits (`BeamKinematicFormulationTraits`, `ShellKinematicFormulationTraits`) so the library can state, without runtime cost, that beam corotational is a reference nonlinear path while shell corotational is available but still not at the same audit level.
- `ContinuumElement`, `BeamElement`, and `MITCShellElement` now consume those concepts directly, so the family-level formulation matrix begins to act as an architectural guardrail in the hot path rather than as a post-hoc explanatory table.
- That guardrail has now been rechecked on real structural targets, not only on concept-level harnesses: `fall_n_beam_test.exe` (`21/21`), `fall_n_mitc_shell_test.exe` (`12/12`), and `fall_n_seismic_infra_test.exe` (`33/33`) all still pass with the new family-aware compile-time constraints in place.
- A local stabilization gate exists in `scripts/ci_multiscale_stabilization.ps1`, and the intended CI surface is frozen in `.github/workflows/multiscale-stability.yml`.
- The GitHub Windows/MSYS2 gate now builds under the UCRT64 MSYS shell but runs regression executables from PowerShell with a UCRT-only runtime path, explicit MS-MPI discovery, and staged critical PETSc/OpenMP runtime DLLs next to the regression executables. This is meant to reduce `STATUS_DLL_NOT_FOUND` / `exit -1073741515` failures that originate in runner runtime resolution rather than in the binaries themselves.
- That runtime staging now reflects the PETSc-heavy regression surface more honestly: besides the curated core DLL list, the workflow also stages the broader PETSc dependency stack (`libpetsc*`, `libmetis*`, `libparmetis*`, `libhwloc*`, `libltdl*`, `libopenblas*`, compiler runtimes, and `msmpi.dll`) because `fall_n_evolver_advanced_test.exe` exercises a deeper runtime chain than the lighter API harness.
- The Windows/MSYS2 gate now also stages the full `ucrt64/bin/*.dll` runtime surface next to the regression executables. This is intentionally broader than the curated dependency list because the deepest PETSc/VTK tests can pick up extra transitive DLLs on GitHub runners even when lighter harnesses and local runs succeed.
- The regression step now also executes from inside `build/` with that directory prepended to `PATH`, and emits an `objdump` trace of `libpetsc-dmo.dll` if a runner still hits `STATUS_DLL_NOT_FOUND`. This keeps the CI diagnosis focused on the real PETSc/MS-MPI dependency chain instead of treating the symptom as a code regression.
- The same Windows gate now resolves staged runtime DLLs through explicit leaf-file checks before copying them into `build/`, so the workflow no longer assumes that every discovered MS-MPI runtime directory actually contains `msmpi.dll`. This closes a real GitHub runner failure mode where `System32` contained the runtime DLL but `C:\\Program Files\\Microsoft MPI\\Bin` did not.
- The same gate now also carries the exact `MSMPI_RUNTIME_DLL` path across steps, so runtime staging and diagnostics are anchored to the actual file discovered during setup instead of reconstructing that path later from partially populated candidate directories.
- A reproducible predefinitive physical-validation harness exists in `scripts/run_predefinitive_physical_validation.ps1`; it records both the current Case 4 short-run milestone and the current Case 5 frontier honestly.
- The structural beam path now exposes a compile-time beam-axis quadrature family in `src/numerics/numerical_integration/BeamAxisQuadrature.hh`. `TimoshenkoBeamN` is no longer nominally tied to Gauss-Legendre stations: Gauss-Legendre, Gauss-Lobatto, and Gauss-Radau rules can now be exercised through the same element formulation, and the reduced shear basis is rebuilt from the actual station coordinates supplied by the bound geometry rule.
- That beam-axis layer is frozen by `tests/test_beam_axis_quadrature.cpp`, which checks monomial exactness of the rule families, left/right Radau consistency, and reuse of the same `TimoshenkoBeamN` formulation with Lobatto and Radau station sets.
- The reduced structural RC-column reboot now has its own canonical matrix in `src/validation/ReducedRCColumnStructuralMatrixCatalog.hh` and a clean runtime surface in `src/validation/ReducedRCColumnStructuralBaseline.hh/.cpp`. This matters because the library can now state, explicitly, that the current Phase-3 runtime baseline is `TimoshenkoBeamN<N> + small strain + {Gauss, Lobatto, Radau}` for `N=2..10`, while `TimoshenkoBeamN + corotational` remains a planned family extension and `TimoshenkoBeamN + TL/UL` remains unavailable rather than silently implied.
- That structural reboot layer is frozen by `tests/test_reduced_rc_column_structural_matrix.cpp`, which constrains the case counts, the honest support split (`36` runtime baseline cases, `36` corotational extension rows, `72` finite-kinematics-unavailable rows), and a runtime smoke pass of the new reduced-column baseline under three beam-axis quadrature families with optional axial compression.
- The Ko-Bathe 3D concrete path now exposes explicit crack-stabilization profiles so the paper-reference parameters and the stabilized FE2-production defaults are no longer conflated.
- The Ko-Bathe 3D audit also closed a real constitutive-state bug: crack opening/closure is now refreshed from the final elastic strain after plastic correction, instead of silently inheriting a pre-return trial state.
- The Ko-Bathe 3D path now classifies `compressive flow` versus `no-flow` explicitly from the trial octahedral invariants, so tensile states and compressive unloading no longer accumulate effective plastic flow spuriously.
- The Ko-Bathe 3D no-flow branch now carries an explicit Eq. (26b)-style tensorial coupling update and switches back to the article-consistent constitutive law in tension/unloading instead of reusing the compressive fracture/plasticity tangent there.
- The FE2 cyclic driver now exposes submodel material-tangent mode explicitly; the numerical consistent tangent is available for audit, but it is not the default because the first-cracked Case 5 benchmark still regresses with it.
- The FE2 cyclic setup now keeps the owning `MultiscaleCoordinator` alive inside the returned case context, so local evolvers no longer hold dangling `MultiscaleSubModel*` references after the setup TU split.
- The FE2 cyclic driver was split one step further through `TableCyclicValidationFE2StepPostprocess`, moving crack-summary aggregation and recorder-row assembly out of the main FE2 runtime translation unit.
- The FE2 cyclic driver now also isolates turning-point restart retuning in `TableCyclicValidationFE2Restart`, reducing coupling between nonlinear step control and restart-budget policy.
- The FE2 cyclic driver now also isolates recorder/bootstrap wiring in `TableCyclicValidationFE2Recorders`, so schema/CSV changes no longer require reopening the main runtime loop translation unit.
- The persistent local solver now honors the configured first-ramp bisection budget and the `arc_length_from_start` flag during the very first nonlinear ramp, which closed a real semantic gap between the FE2 validation drivers and the local solver contract.
- The persistent local solver now also supports explicit late-tail continuation on the remaining segment of a partially converged ramp, with full diagnostics (`adaptive_tail_rescue_attempts`, trigger fraction, and propagated minimum step size) instead of silently treating every exhausted budget as the same failure.
- The iterated FE2 orquestator now damps the very first micro feedback predictor against a zero baseline, so the fixed-point relaxation policy can act before the second macro solve instead of waiting one full failed iteration.
- The FE2 relaxation path now blends full affine section laws consistently, not just raw `forces` and `tangent` arrays, and the iterated macro solve now has an explicit operator-space backtracking path for `MacroSolveFailed` recoveries.
- The FE2 coupling report now exposes macro SNES diagnostics (`reason`, iteration count, residual norm) and per-site section-operator diagnostics (minimum symmetric eigenvalue, maximum symmetric eigenvalue, trace, and nonpositive diagonal count) so a `MacroSolveFailed` event is no longer a black box.
- The iterated FE2 path now also supports a configurable predictor-admissibility filter: before a cracked local operator is sent into the next macro re-solve, the affine section law can be blended toward a baseline law until the symmetric-part eigenvalue floor is satisfied. This is explicit, reported, and intended as a validation aid rather than a hidden constitutive modification.
- The iterated FE2 path now also supports a configurable macro step-cutting continuation when the macro solver exposes runtime increment control. The cutback preserves the target control point by retrying the same FE2 macro re-solve with a smaller solver increment and `step_to(...)`, and it reports both the nominal and accepted increment sizes explicitly.

This does not mean the whole repository is “finished”. It means the codebase now has one path that is close to publication quality, and the rest of the library can be reviewed and strengthened around that standard.

## High-Level Architecture

The repository is organized into recognizable modules under `src/`:

- `geometry`: cells, topologies, and geometric primitives.
- `numerics`: interpolation, quadrature, tensors, condensation, sparse Schur utilities, and linear algebra helpers.
- `materials`: constitutive relations, material state, local nonlinear problems, plasticity, cyclic uniaxial laws, fiber sections, and section builders.
- `elements`: element contracts and concrete beam, shell, truss, and continuum elements.
- `model`: DoFs, checkpoints, model state, builders, ground-motion utilities, and domain assembly.
- `analysis`: nonlinear analysis, dynamics, arc-length, multiscale orchestration, executors, and coupling strategies.
- `reconstruction`: field transfer, submodel solving/evolution, homogenization, and local-model adapters.
- `mesh`: Gmsh readers/builders and mesh support.
- `post-processing`: VTK export, state queries, and structural output.
- `validation`: reusable driver APIs and heavy validation implementations.
- `utils`, `petsc`, `continuum`, `domain`, `graph`: support layers and legacy/bridge components.

The multiscale publication path currently looks like this:

1. `BeamMacroBridge` extracts local macro kinematics and section states.
2. `MultiscaleModel` binds `CouplingSite` objects to local models.
3. `MultiscaleAnalysis` owns the protocol, rollback, convergence, relaxation, and execution policy.
4. `NonlinearSubModelEvolver` acts as the persistent local-model facade.
5. `BoundaryReactionHomogenizer` computes section forces and the condensed tangent, with explicit fallback and validation against adaptive finite differences.

For the iterated FE2 path, the active section law is treated as
\[
s(e)=s_\mu + D_\mu (e-e_\mu).
\]
That matters operationally: both fixed-point relaxation and macro-failure
backtracking are now applied to the affine law itself, not to unrelated
arrays. In practice, the code first translates both laws to a common strain
reference, blends the translated force intercepts and tangents, and only then
reinjects the result into the macro beam section.

The current continuum local-model implementation is still reinforced-concrete-oriented, but the dependency boundary is now explicit: multiscale/reconstruction depends on abstract local-material factories, and the Ko-Bathe/Menegotto reference pair is just one materials-module realization. This is the seam intended for future local-model variants, including enriched/XFEM-like local solvers or discontinuous Petrov-Galerkin / DG local formulations.

At the utility-solver level, the embedded rebar penalty coupling is also no longer a hidden constant: `SubModelSolver` now exposes an `EmbeddedLinePenaltyCouplingConfig`, so the reinforced-concrete heuristic remains available as a default but is no longer hard-coded as if it were the only physically meaningful choice.

The local architectural split is already underway:

- `LocalBoundaryConditionApplicator`
- `PersistentLocalStateOps`
- `BoundaryReactionHomogenizer`
- `LocalCrackDiagnostics`
- `LocalVTKOutputWriter`
- `SparseSchurComplementWorkspace`

That split is important because it keeps the hot path static and efficient while creating real seams for future scientific upgrades.

## Public Surface

The explicit public multiscale umbrella is:

- `src/analysis/MultiscaleAPI.hh`
- CMake target: `fall_n::multiscale_api`

The local continuum-material factory boundary supporting that surface is:

- `src/materials/SubmodelMaterialFactory.hh`
- `src/materials/SubmodelMaterialFactoryDefaults.hh`

This is the recommended include/link surface for the multiscale subsystem. The older `header_files.hh` umbrella still exists and is still heavily used internally, but it should be treated as a transitional convenience rather than the long-term public API.

## Main Dependencies

The current build expects:

- CMake >= 3.21
- a C++23 compiler
- MPI
- PETSc
- Eigen3
- VTK
- OpenMP (optional but used when available)
- TeX Live / `pdflatex` or `latexmk` for the manuscript

On Windows/MSYS2, the workflow in `.github/workflows/multiscale-stability.yml` is the best reference for the dependency set currently exercised by the project.

## Build

Typical local configuration:

```powershell
cmake -S . -B build -G Ninja
```

Useful focused builds:

```powershell
ninja -C build fall_n_multiscale_api_test
ninja -C build fall_n_evolver_advanced_test
ninja -C build fall_n_tangent_validation_benchmark_test
ninja -C build fall_n_coupling_residual_benchmark_test
ninja -C build fall_n_beam_axis_quadrature_test
ninja -C build fall_n_reduced_rc_column_structural_matrix_test
ninja -C build fall_n_reduced_rc_column_validation_claim_catalog_test
ninja -C build fall_n_table_cyclic_validation
```

The repository also contains many executable validation drivers, including:

- `fall_n_lshaped_multiscale`
- `fall_n_lshaped_multiscale_16`
- `fall_n_table_multiscale`
- `fall_n_table_cyclic_validation`
- `fall_n_rc_beam_validation`

## Validation Reboot Program

The repository is now carrying an explicit validation reboot plan instead of
continuing to grow by inheriting authority from older drivers.

The canonical compile-time contract for that reboot lives in:

- `src/validation/ValidationCampaignCatalog.hh`
- `tests/test_validation_campaign_catalog.cpp`
- `src/validation/ReducedRCColumnStructuralMatrixCatalog.hh`
- `src/validation/ReducedRCColumnStructuralBaseline.hh`
- `src/validation/ReducedRCColumnValidationClaimCatalog.hh`
- `src/validation/ReducedRCColumnBenchmarkTraceCatalog.hh`
- `src/validation/ReducedRCColumnEvidenceClosureCatalog.hh`
- `tests/test_reduced_rc_column_structural_matrix.cpp`
- `tests/test_reduced_rc_column_validation_claim_catalog.cpp`
- `tests/test_reduced_rc_column_benchmark_trace_catalog.cpp`
- `tests/test_reduced_rc_column_evidence_closure_catalog.cpp`

The governing rule is simple and deliberately strict:

1. No existing test, driver, or manuscript chapter is treated as validated
   truth until it is re-audited against the actual computational claim it is
   supposed to support.
2. The first normative physical-validation target is a single reinforced-
   concrete rectangular column under progressively amplified cyclic lateral
   displacement, not a full structure.
3. Reduced-order structural models must be closed first, then the equivalent
   continuum column, then the reduced-versus-continuum equivalence gate, and
   only after that may larger structural or FE2-heavy campaigns become
   normative again.

The current structural reboot frontier is therefore intentionally explicit:

- Baseline runtime-ready slice: `TimoshenkoBeamN<N> + continuum::SmallStrain + BeamAxisQuadratureFamily::{GaussLegendre, GaussLobatto, GaussRadauLeft, GaussRadauRight}` for `N=2..10`.
- Planned but not yet runtime-ready family extension: `TimoshenkoBeamN<N> + beam::Corotational`.
- Unavailable in the current beam family: `TimoshenkoBeamN<N> + continuum::TotalLagrangian` and `TimoshenkoBeamN<N> + continuum::UpdatedLagrangian`.

That distinction is not cosmetic; it is the reason the validation reboot can stay scientifically honest while still remaining modular and extensible.

The same reboot now also distinguishes between "the runtime surface exists"
and "the scientific claim is already ready for a structural benchmark". The
current honest read for the reduced RC column is:

- Frozen enabling contract: the beam-axis quadrature family is now an explicit model axis and no longer a hidden implementation detail.
- Runtime-baseline ready: the small-strain `TimoshenkoBeamN<N>` column path exists for `N=2..10`, the optional axial-compression load path exists, the base-shear-vs-drift hysteresis CSV contract exists, and a normative base-side moment-curvature observable is now exported through `moment_curvature_base.csv`.
- Benchmark pending: node-refinement convergence and quadrature-sensitivity claims are still open and must be closed by explicit benchmark suites, not inferred from smoke runs.
- Prebenchmark caveat: the moment-curvature observable is defined at the active section station closest to the fixed end, so it is now a clean runtime observable, but it still requires benchmark closure and quadrature-sensitivity interpretation before it can anchor physical validation.
- The benchmark layer is now frozen explicitly as `claim -> benchmark -> reference class -> acceptance gate`, so the reduced-column campaign no longer jumps directly from runtime output contracts to physical-validation rhetoric.
- The evidence-closure layer is now frozen explicitly as `claim -> current artifact -> missing experiment -> closure artifact`, so every open reduced-column claim carries a concrete numerical obligation instead of only a generic “future benchmark” label.

The representative reboot workstreams are currently classified as:

- `mandatory_blocker`: must close before the reference column campaign or full
  structural escalation can be claimed.
- `conditional_enabler`: valuable extensions such as `TrussElement<Nnodes>` or
  alternative concrete models, but only promoted if the audited campaign shows
  they are truly needed.
- `deferred_growth_path`: important future work such as force-based structural
  elements, intentionally kept out of the first validation baseline.

Legacy validation surfaces are therefore preserved as audited input, not
destroyed immediately. They should be quarantined to `.old` only after the
replacement campaign exists and covers the same scientific front more honestly.

## Tests and Stability Gate

Focused multiscale regression executables include:

- `fall_n_multiscale_api_test.exe`
- `fall_n_micro_solve_executor_test.exe`
- `fall_n_evolver_advanced_test.exe`
- `fall_n_tangent_validation_benchmark_test.exe`
- `fall_n_coupling_residual_benchmark_test.exe`
- `fall_n_beam_test.exe`
- `fall_n_steppable_solver_test.exe`
- `fall_n_steppable_dynamic_test.exe`

The current local stabilization gate is:

```powershell
scripts/ci_multiscale_stabilization.ps1
```

It builds the focused regression surface, the heavy multiscale examples, and the LaTeX document.

The current predefinitive physical-validation harness is:

```powershell
scripts/run_predefinitive_physical_validation.ps1
```

It reruns the focused multiscale regression executables, launches a one-step one-way FE2 short run, launches a one-step iterated two-way FE2 short run, stores logs under `data/output/cyclic_validation/predefinitive_validation/logs`, and writes a machine-readable summary CSV.

The current full Case 5 launcher for VTK-oriented review is:

```powershell
scripts/run_case5_full_vtk.ps1
```

It starts the full `Case 5` FE2 campaign in the background with the current
`extended50 + crack50` defaults, preserves the run command in
`data/output/cyclic_validation/case5/full_run/manifest.txt`, redirects stdout
and stderr to log files in the same folder, writes the active PID to
`data/output/cyclic_validation/case5/full_run/pid.txt`, and records completion
status and exit code in `data/output/cyclic_validation/case5/full_run/completion.txt`.

## Thesis Submodule

The repository now includes the companion thesis as a Git submodule:

- path: `PhD_Thesis`
- remote: `https://github.com/sechavarriam/PhD_Thesis.git`

This keeps the library and the thesis versioned separately while still letting
us work in a unified local workspace.

The workspace now also version-controls the thesis recipe used in day-to-day
editing with LaTeX Workshop:

- recipe name: `kaobook`
- sequence: `xelatex -> makeindex -> makeindex_nomencl -> biber -> makeglossaries -> xelatex -> xelatex`

That sequence is intentionally explicit because the thesis uses `kaobook`,
`polyglossia`, bibliography, glossary, index, and nomenclature; a single
`pdflatex` pass is not representative of the real manuscript lifecycle.
The build should also be treated as strictly sequential: interrupted or
parallel `xelatex` runs can leave stale `main*.mw` intermediates from
`morewrites`, and those files can corrupt later passes until they are cleaned.

Typical usage after cloning `fall_n`:

```powershell
git submodule update --init --recursive
```

To pull the latest thesis state intentionally:

```powershell
git submodule update --remote PhD_Thesis
```

To work on the thesis itself:

```powershell
git -C PhD_Thesis status
git -C PhD_Thesis add ...
git -C PhD_Thesis commit
```

After advancing the thesis commit, record the new submodule pointer in
`fall_n` with a normal commit in the superproject.

Important convention:

- build artefacts and local edits inside `PhD_Thesis` belong to the thesis
  repository, not to `fall_n`;
- the superproject should only track the submodule pointer and `.gitmodules`,
  not the internal thesis file history;
- do not “clean” the thesis from the `fall_n` root unless you intend to clean
  the thesis repository itself.

## Documentation

The main technical document is:

- `doc/main.tex`

Compile it with:

```powershell
cd doc
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

or, if needed:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Important recent chapters for the multiscale/publication path are:

- `doc/ch83_multiscale_publication_audit.tex`
- `doc/ch84_multiscale_stabilization_execution.tex`
- `doc/ch82_cyclic_validation.tex`

The document compiles, but it still contains pre-existing warnings and historical chapters that should be read as provenance, not always as the latest normative behavior. Chapters 83 and 84 are the current authoritative multiscale reference.

## What Is Strong Today

- The library has a real concept/policy architecture in several core areas, especially materials and multiscale orchestration.
- The multiscale subsystem now has explicit semantics, rollback, diagnostics, and tests.
- The local condensed tangent path is much more honest than before: explicit sparse condensation, explicit fallback, explicit validation.
- The repository has unusually rich internal technical documentation for an active research codebase.
- The test culture is strong enough to support serious refactors.

## Known Limitations

- `NonlinearSubModelEvolver` is still too large, even though the most critical concerns have already been split out.
- The current weighted norms for condensed-tangent validation and FE2 residuals are physically better than pure Frobenius, but they are still proxies; `DualEnergyScaled` improves the generalized-work proxy, yet it is not a full proof of energetic equivalence.
- The numerical consistent tangent of `KoBatheConcrete3D` is still experimental in FE2 submodels: it is exposed and benchmarked, but the current audited Case 5 frontier is no longer best explained as a tangent-only problem. Under strict diagnostic budgets the limiter is still the late, heavily cracked tail of the first micro ramp; under the default `fe2_crack50` profile the frontier has already moved one level up to the macro FE2 re-solve after all four micro columns converge.
- The new predictor-admissibility filter improves observability and gives the macro solve a controllable prefilter for strongly indefinite cracked operators, but it is not yet evidence of full physical validation. It is an explicit numerical safeguard for the validation campaign, not a proof that the final accepted cracked section laws are globally admissible for arbitrary macro paths.
- The 3D Ko-Bathe path now enforces the paper's top-level `flow / no-flow` classification and now carries an explicit Eq. (26b) no-flow tensor update, but the full crack-status `m`-loop from Table 1 is still approximated rather than transcribed literally.
- The distributed MPI micro-solve engine is not implemented yet beyond contracts and communicator ownership.
- The root `CMakeLists.txt` is still monolithic.
- `header_files.hh` still causes avoidable build coupling and PCH invalidation.
- The dedicated `fall_n_case5_frontier_probe_test` target now builds again on the audited Windows/MSYS2 path after the probe-specific link cleanup, but it remains a runtime-limited artifact in the local harness and therefore is not yet a stable CI-speed oracle for the full Case 5 frontier.
- The repository mixes naming styles such as `non_lineal` / `nonlinear` and `Homogenisation` / `Homogenized`.
- The top-level documentation still contains historical drift and LaTeX warning debt.
- Some heavy validation drivers remain expensive to compile and run; on the latest audited cyclic-validation pass, the isolated `TableCyclicValidationFE2.cpp` object rebuilt in about `177.2 s`, while the extracted `TableCyclicValidationFE2Recorders.cpp` slice rebuilt in about `38.6 s`, so the heavy runtime TU frontier is still open even though the IO/recorder slice is now materially cheaper to edit.

## Quick Wins

The fastest improvements with a good effort/impact ratio are:

1. Keep shrinking the use of `header_files.hh` and move toward module-local umbrellas.
2. Split `CMakeLists.txt` into smaller module-oriented fragments.
3. Continue breaking up the FE2-heavy cyclic-validation translation units, because the setup split and the new FE2 step-postprocess split improved cost-of-change and ownership clarity but did not yet solve the heavy compile frontier.
4. Normalize naming and spelling across directories and public types.
5. Clean generated artefacts and keep runtime output out of the repository root.
6. Expand installable public targets beyond `fall_n::multiscale_api`.
7. Clean the remaining LaTeX warnings and align old chapters with the current code.

## Pre-Validation Readiness

The repository now distinguishes three different questions that used to be too
easy to blur together:

1. What formulation/solver slice exists?
2. What residual, tangent, and history commitments that slice actually carries?
3. Is the resulting scientific claim already ready to enter a physical-validation campaign?

The third question is now frozen in
`src/analysis/ComputationalValidationReadinessCatalog.hh`.

The current honest read is:

- `continuum_total_lagrangian_nonlinear` is the only representative slice that is presently classified as ready for a targeted physical-validation campaign.
- `continuum_updated_lagrangian_nonlinear`, `continuum_total_lagrangian_dynamic`, `beam_corotational_nonlinear`, and `shell_corotational_nonlinear` are real and useful, but still classified as scope-closure pending.
- `continuum_total_lagrangian_arc_length` is semantically audited, but still classified as runtime-regression pending.
- the linear continuum/beam/shell baselines are frozen references, not validation claims.

This matters because the validation chapters should compare numerical evidence
against the right object: not just a formulation name, but an audited slice
with an explicit evidence floor and an explicit remaining gate.

## Cyclic Validation Direction

The cyclic validation campaign now has:

- an `extended50` protocol,
- a `fe2_crack50` runtime profile,
- streamed hysteresis output,
- summary-only plotting support,
- a predefinitive physical-validation script with machine-readable summaries,
- and a clearer split between structural reference runs and FE2 runs.

The structural case already reaches the full `50 mm` envelope. The FE2 case has progressed to a reproducible partial frontier with crack activity and hysteresis data preserved, but it still needs more runtime work to reach the full envelope efficiently.

The current honest frontier is:

- Case 4 one-way FE2 reaches the first cracked point reproducibly in the short-run matrix and remains scientifically observable through `+5 mm` in the longer exploratory runs.
- Case 5 iterated two-way FE2 now survives the setup phase and enters an actually adaptive first local ramp under the tuned `fe2_crack50` short-run profile; the first-ramp bisection budget is no longer silently ignored by the local solver.
- In the strict one-step Case 5 audit (`4` initial increments, `2` first-ramp bisections, `20` local SNES iterations), all four submodels now pass the former `k = 1/4` wall and reach between `18.75%` and `31.25%` of the local target before the minimum fraction floor is hit.
- In the wider one-step Case 5 audit (`4` initial increments, `2` first-ramp bisections, `60` local SNES iterations, adaptive budget `12/4`), one submodel reaches the full target, two reach `81.25%`, and one reaches `93.75%`; the failed sites now report `AdaptiveMinFractionReached` rather than a featureless early Newton abort.
- The same audited run shows `72–82` active cracked material points, up to `3` cracks at a point, one-step no-flow stabilization, and no no-flow destabilization. That is strong evidence that the remaining frontier is the late, heavily cracked tail of the first micro ramp, not the top-level `flow / no-flow` split itself.
- In the current default `fe2_crack50` one-step audit, all four submodels now reach `100%` of the first `+2.5 mm` target with `4–7` accepted ramp substeps, `0–3` bisections, `80–82` active cracked points, and zero tail-rescue activations. The step still aborts, but now with `MacroSolveFailed` and `failed_submodels = 0`: the scientific frontier has moved from the local micro ramp to the first macro FE2 re-solve with injected cracked-section operators.
- A synthetic API regression now proves that the new macro backtracking path can recover a macro solve that would otherwise fail under the same affine section-law predictor. The full structural `Case 5` frontier remains macro-dominated, but the cheap probe used during stabilization is still runtime-limited; the new backtracking path should therefore be read as a validated algorithmic capability, not yet as a closed end-to-end physical-validation result.
- A second execution profile, `fe2_frontier_audit`, now exists to make the first cracked Case 5 point reproducible and machine-readable at lower runtime cost. In the current audited run `build/fall_n_table_cyclic_validation.exe --case 5 --protocol extended50 --fe2-profile frontier --max-steps 1 --global-output-interval 0 --submodel-output-interval 0`, the uncoupled macro step converges, but the FE2 step aborts with `MicroSolveFailed`, `failed_submodels = 2`, and failed coupling sites `eid=0/gp=0/xi=-0.57735` and `eid=2/gp=0/xi=-0.57735`.
- Failed FE2 steps now leave explicit `accepted=0` rows in `global_history.csv` and `crack_evolution.csv`. Structural quantities that are no longer physically defined after rollback, such as accepted base shear and accepted structural peak damage, remain `NaN` instead of being silently back-filled from the restored state.
- The current `fe2_frontier_audit` row is intentionally subtle: it records `total_cracked_gps = 0`, `total_cracks = 0`, and `max_opening = 0`, but also `total_active_crack_history_points = 306` and `max_num_cracks_at_point = 3`. That means the material points already carry crack history, while no open-crack observable survives the current recorder criterion at that failed FE2 point. The validation methodology therefore needs both observables; either one alone would give a misleading diagnosis.
- A second synthetic API regression now proves that the FE2 loop can also recover a macro failure by cutting back the macro increment while still reaching the same target control point. This is a continuation aid for the validation campaign, not yet a claim that the full structural `Case 5` path is closed.
- The next concrete validation task is now split in two levels instead of one. Under `fe2_frontier_audit`, the cheap reproducible frontier is micro-limited and localized at two coupling sites. Under the production-like `fe2_crack50` budget, the frontier is already macro-dominated. That split is scientifically useful: it tells us that further validation needs both a local late-tail continuation study and a macro post-peak continuation study, rather than a single undifferentiated "Case 5 still fails" verdict.
- The optional consistent-tangent override remains an audit path, not a promotion path. The earlier negative result is preserved as historical evidence, but after the FE2 setup-lifetime correction it should be treated as a benchmark that needs re-audit before any final scientific claim is made.

## Bottom Line

`fall_n` is best understood today as an actively hardened research library: ambitious, already technically rich, and increasingly explicit about what is mature, what is experimental, and what still needs scientific or architectural closure. The multiscale subsystem is currently the clearest example of that direction.
