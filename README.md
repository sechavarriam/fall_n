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
- The same layer now also records formulation-level virtual-work semantics explicitly: `VolumeMeasureKind`, `VirtualWorkCompatibilityKind`, and `VirtualWorkSemantics` distinguish exact work-conjugate formulations (TL, UL), linearized-equivalent ones (small strain), and partial/disclaimed formulations whose virtual-work tangent is not yet fully audited (continuum corotational).
- That semantic lift now also carries an explicit audit layer: `AuditEvidenceLevel` and `FormulationAuditScope` distinguish mathematical admissibility from implementation evidence, so the code can state at compile time whether a formulation is merely declared, regression-tested, reference-grade, or still in need of a finite-kinematics disclaimer.
- A second audit layer now separates formulation semantics from family-level deployment: `src/continuum/FormulationScopeAudit.hh` records, at compile time, whether a formulation is actually available for `continuum_solid_3d`, `beam_1d`, or `shell_2d`, whether it is only partial, and whether it is the current reference path for geometric nonlinearity in that family.
- That family-level audit is now queryable as structured data as well as traits: `canonical_family_formulation_audit_table()`, `canonical_family_formulation_audit_row(...)`, and the `find_family_*_reference_path(...)` helpers expose the same matrix that the thesis now uses as its normative reading of `family Ã— formulation`.
- That family-level audit is no longer only documentary: `FamilyKinematicPolicyAuditTraits` and the `FamilyNormativelySupportedKinematicPolicy` / `FamilyReferenceGeometricNonlinearityKinematicPolicy` concepts now let element templates reject unsupported family/formulation combinations at compile time instead of relying on comments or downstream test failures.
- A third audit layer now separates "formulation available" from "solver route actually validated": `src/analysis/AnalysisRouteAudit.hh` records the current scope of `linear_static`, `nonlinear_incremental_newton`, `implicit_second_order_dynamics`, and `arc_length_continuation` for each audited `family Ã— formulation` combination.
- That route-level layer is also exposed as structured compile-time data through `canonical_family_formulation_analysis_route_audit_table()`, `canonical_family_formulation_analysis_route_row(...)`, and route-count helpers, so the thesis can distinguish, truthfully, between a formulation that exists in an element family and a solver path that is actually reference-grade for that same scope.
- The representative `family Ã— formulation Ã— route` matrix now also has a canonical catalog in `src/analysis/AnalysisRouteCatalog.hh`. It is intentionally kept outside the main analysis umbrella so the thesis, the README, and the regression surface can reuse one audited route table without pushing extra metadata into the common hot-path include stack.
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
- That semantic lift also freezes a more honest finite-kinematics status: continuum `TotalLagrangian` is the current reference path, continuum `UpdatedLagrangian` is real but still classified as partial, and continuum `Corotational` now has a partial runtime path with a frozen-rotation tangent, distinct from the more mature corotational beam/shell paths.
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

This does not mean the whole repository is â€œfinishedâ€. It means the codebase now has one path that is close to publication quality, and the rest of the library can be reviewed and strengthened around that standard.

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

This is the recommended include/link surface for the multiscale subsystem. The older `header_files.hh` umbrella still exists and is still heavily used internally, but it is now explicitly transitional: the shared PCH no longer goes through it, and new code should prefer direct includes or narrower module-local umbrellas.

The current compile-time hygiene baseline is:

- shared PCH surface: `src/fall_n_common_pch.hh`
- legacy umbrella kept only for compatibility: `header_files.hh`
- first narrow migration umbrella for the reduced-column beam slice: `src/validation/BeamValidationSupport.hh`
- optional migration audit switch: `-DFALL_N_ENABLE_HEADER_FILES_DIAGNOSTICS=ON`

## Wrapper-Ready Benchmark Surfaces

The reduced RC benchmark family now has a more explicit user-facing contract:

- the runtime manifests of the structural, continuum, section, and material
  reduced-column benchmarks now carry the same versioned manifest contract,
- each benchmark declares an `input_surface` block, and
- each resolved run declares a `local_model_taxonomy` block.

The relevant seams are:

- `src/validation/ReducedRCColumnBenchmarkSurface.hh`
- `src/analysis/LocalModelTaxonomy.hh`

This is intentionally a small step, but it matters. We are not forcing a
premature JSON parser or a Python/Julia wrapper today; instead, we are making
the current CLI surfaces auditable and stable enough that a wrapper can target
them intentionally later, rather than reverse-engineering ad hoc flags and
one-off manifest fields.

That same stabilization now reaches all four benchmark layers of the reduced
RC family:

- material benchmark: uniaxial constitutive-point audit,
- section benchmark: moment-curvature control problem over the audited fiber
  section,
- structural benchmark: Timoshenko/fiber surrogate, now with explicit
  `--section-fiber-profile coarse|canonical|fine|ultra` refinement controls
  that preserve steel area and bar coordinates while changing only the
  concrete patch subdivision,
- continuum benchmark: 3D host plus explicit reinforcement branch metadata,
- structural-continuum bridge scripts now make the top-rotation comparator
  explicit. The default structural comparator is free top rotation because the
  current continuum benchmark prescribes only lateral top-face translation;
  `--structural-top-rotation-mode clamped` remains available as the stronger
  guided-cap control.

The elastic boundary-condition guardrail is reproducible with:

```powershell
py -3.11 scripts/run_reduced_rc_elastic_axis_stiffness_control.py `
  --output-dir data/output/cyclic_validation/reboot_elastic_axis_stiffness_control/reproducible
```

That guardrail currently reports the continuum lateral-top-face stiffness close
to the free-rotation structural beam and far from the clamped/guided beam. It
also runs an explicit `--top-cap-mode uniform-axial-penalty-cap` branch: that
branch ties the top-face axial displacement field to one free axial motion, but
is kept as a non-promoted audit because it over-stiffens this elastic control
relative to the structural clamped beam. This prevents future wrappers or
notebooks from silently comparing different boundary kinematics.

For the current reduced-column family, that taxonomy now distinguishes
explicitly between:

- the structural Timoshenko/fiber surrogate benchmark,
- section-level control problems over the same RC fiber ingredients,
- uniaxial constitutive-point ingredient audits,
- the promoted local continuum baseline (`Hex20` host + smeared fixed-crack
  concrete + interior embedded bars),
- explicit control branches such as `plain` continuum or `boundary bars`, and
- future extension routes such as XFEM- or DG-like local fracture solvers.

This helps in two directions at once: it makes the current validation inputs
clearer for users, and it keeps future wrapper work from hard-coding
benchmark-specific assumptions into external orchestration scripts.

## Local Fracture Model Strategy

The current promoted local continuum remains a smeared fixed-crack RC solid
with embedded reinforcement. That is a deliberate choice, not inertia.

Right now, that model is the one that:

- already carries the most audited preload / continuation / predictor chain,
- preserves a clean host-bar transfer story,
- can be compared honestly against the structural benchmark, and
- is the only local family whose physical and computational behaviour we have
  pushed far enough to promote toward future multiscale use.

XFEM or DG may still become the right next local family, especially if the
dominant cost really comes from resolving narrow localized crack paths with a
host mesh that is finer than the rest of the local physics requires. But they
would only help if we accept their real algorithmic cost as well:

- XFEM needs enriched displacement support, crack geometry tracking, cut-cell
  quadrature, and robust closure/contact handling.
- DG/HDG-style local fracture paths replace enriched nodal fields with
  interface/skeleton traces, which can be attractive for discontinuities but
  changes the effective-operator extraction and checkpoint semantics.

For that reason, the architectural move in this stabilization break is not to
ship a rushed XFEM element. The correct move is to:

1. keep the smeared-crack continuum as the promoted validated baseline,
2. make local-model families explicit in code, manifests, and subscale-model
   contracts, and
3. prepare clean seams so a future XFEM/DG local model can plug into the same
   validation and multiscale story honestly.

Concretely, this means a future enriched or DG local solver should arrive as a
new local-model family that can answer the same high-level questions as the
current promoted baseline:

- what driving state does it accept?
- what effective operator does it return?
- what checkpoint/restart semantics does it support?
- and what taxonomy does it declare for validation and wrapper tooling?

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
- `src/validation/ReducedRCColumnSectionBaseline.hh`
- `src/validation/ReducedRCColumnMomentCurvatureClosure.hh`
- `src/validation/ReducedRCColumnMomentCurvatureClosureMatrix.hh`
- `src/validation/ReducedRCColumnNodeRefinementStudy.hh`
- `src/validation/ReducedRCColumnQuadratureSensitivityStudy.hh`
- `src/validation/ReducedRCColumnCyclicQuadratureSensitivityStudy.hh`
- `tests/test_reduced_rc_column_structural_matrix.cpp`
- `tests/test_reduced_rc_column_validation_claim_catalog.cpp`
- `tests/test_reduced_rc_column_benchmark_trace_catalog.cpp`
- `tests/test_reduced_rc_column_evidence_closure_catalog.cpp`
- `tests/test_reduced_rc_column_section_baseline.cpp`
- `tests/test_reduced_rc_column_quadrature_sensitivity_study.cpp`
- `tests/test_reduced_rc_column_cyclic_quadrature_sensitivity_study.cpp`
- `tests/test_reduced_rc_column_moment_curvature_closure.cpp`
- `tests/test_reduced_rc_column_moment_curvature_closure_matrix.cpp`
- `tests/test_reduced_rc_column_node_refinement_study.cpp`

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
- The reboot now also has an independent section-level artifact in `src/validation/ReducedRCColumnSectionBaseline.hh/.cpp`, driven by axial-force equilibrium closure at each curvature step. That matters because the future `M`-`\kappa` comparison can now separate section constitutive behavior from beam discretization effects instead of collapsing both sources of error into one plot.
- A third artifact, `src/validation/ReducedRCColumnMomentCurvatureClosure.hh/.cpp`, now compares the representative structural base-side branch against that independent section baseline on the physically relevant control variable, curvature. The structural baseline now uses an explicit equilibrated axial-preload stage and then holds that load during the lateral branch. On the current representative slice (`N = 4`, Gauss-Lobatto, monotonic `2.5 mm` tip displacement, `0.02 MN` axial compression), the closure is now fully tight: the maximum relative errors are about `6.59e-7` in moment/secant, `1.50e-9` in tangent, and `1.56e-8` in axial force.
- That representative comparison is no longer the whole story. `src/validation/ReducedRCColumnMomentCurvatureClosureMatrix.hh/.cpp` now sweeps all `36` runtime-ready small-strain slices (`N = 2..10` crossed with `Gauss`, `Lobatto`, `Radau-left`, `Radau-right`) and writes case-level plus aggregate closure tables. The current preload-consistent matrix result is strong: `36/36` cases complete, `36/36` satisfy the representative closure tolerances including axial force, the worst moment/secant mismatch is about `5.095e-7` (at `n05_gauss_radau_left_small_strain`), the worst tangent mismatch is about `2.178e-9` (at `n08_gauss_legendre_small_strain`), and the worst axial-force mismatch is about `2.677e-7` (also at `n08_gauss_legendre_small_strain`).
- A fourth artifact, `src/validation/ReducedRCColumnNodeRefinementStudy.hh/.cpp`, now turns the reduced family into a full internal monotonic refinement study instead of leaving refinement as a purely verbal expectation. The study now spans the entire runtime-ready matrix `N=2..10 Ã— {GaussLegendre, GaussLobatto, GaussRadauLeft, GaussRadauRight}` and compares each base-side branch against the highest-`N` reference inside the same quadrature family. On the full monotonic bundle, all `36/36` cases complete and all `36/36` pass the representative internal refinement gate. The worst terminal-moment drift is about `1.41e-4`, the worst pointwise moment drift about `2.14e-4`, the worst tangent drift about `5.03e-7`, and the worst secant drift about `5.04e-5`. The drift decays strongly with `N` and is exactly zero at `N=10` by construction, so the remaining open gate is no longer the monotonic matrix itself but the cyclic refinement closure.
- A sixth artifact, `src/validation/ReducedRCColumnCyclicNodeRefinementStudy.hh/.cpp`, now opens the same question under cyclic evolution instead of assuming that monotonic internal closure automatically survives unloading and reloading. The first representative pilot spans `{N=2,4,10} Ã— {GaussLegendre, GaussLobatto}` with two amplitudes `{1.25, 2.50} mm`, `steps_per_segment = 2`, and `0.02 MN` axial compression. All `6/6` pilot cases complete, but only `3/6` pass the representative cyclic internal gate. The worst terminal return-moment drift is about `2.80e-1`, the worst history-moment drift about `6.33e-1`, the worst turning-point drift about `6.33e-1`, and the worst controlling-station shift is `1.0`; by contrast, the worst tangent-history drift is only about `4.53e-3` and the worst axial-force-history drift only about `1.74e-5`. This is a useful scientific result: the current cyclic frontier is dominated by reversal-sensitive base-side moment history, not by axial-force mismatch and not by tangent collapse.
- The same cyclic pilot also forced a methodological correction: secant drift over the full cyclic history is now treated as diagnostic only, not as the representative cyclic gate, because reversal and near-zero-curvature crossings make that metric ill-conditioned even when the underlying response remains interpretable. The representative cyclic gate is now based on terminal return moment, full-history moment drift, turning-point drift, tangent drift, and axial-force drift.
- That pilot is no longer the end of the story. The full cyclic matrix now also exists over the entire runtime-ready family `N=2..10 Ãƒâ€” {GaussLegendre, GaussLobatto, GaussRadauLeft, GaussRadauRight}`. On that full matrix, all `36/36` cases complete, but only `8/36` satisfy the representative cyclic internal gate. The worst terminal return-moment drift is about `8.40e-1`, the worst history-moment drift about `8.40e-1`, the worst turning-point drift about `8.40e-1`, and the worst controlling-station shift is `1.0`; by contrast, the worst tangent-history drift is only about `4.53e-3` and the worst axial-force-history drift only about `1.74e-5`. That is the current honest read: the cyclic frontier is dominated by reversal-sensitive base-side moment history and station migration, not by axial-force mismatch and not by tangent collapse.
- A seventh artifact, `src/validation/ReducedRCColumnCyclicContinuationSensitivityStudy.hh/.cpp`, now compares monolithic displacement control against a reversal-guarded displacement-control schedule on the same full cyclic matrix. The guarded policy keeps `36/36` completed cases and the same `8/36` representative passes, so it does not close the frontier by itself; however, it improves terminal return drift in `16/36` cases, history-moment drift in `19/36`, turning-point drift in `19/36`, secant drift in `15/36`, and axial-force drift in `32/36`, while leaving the controlling-station shift unchanged in every case. This is exactly the evidence we needed before talking seriously about stronger continuation such as arc-length.
- That continuation study has now also been compacted structurally: the baseline/candidate/delta bookkeeping and the node-wise/global summaries are expressed through small local accumulators and lambdas instead of repeated field-by-field update blocks. This does not touch the numerical hot path, but it does reduce maintenance drift in the postprocessing layer that supports the validation claims.
- In other words, the old statement that the full cyclic node-refinement matrix was still missing should now be read as historical only. The open gate is narrower: closure under a declared continuation policy, plus an explicit decision about whether any later escalation toward arc-length is actually justified by the continuation-sensitivity bundle.
- A fifth artifact, `src/validation/ReducedRCColumnQuadratureSensitivityStudy.hh/.cpp`, now turns quadrature-family sensitivity into an explicit controlled study instead of leaving it as a warning in the structural matrix. The study now spans the full runtime-ready matrix `N=2..10 Ã— {GaussLegendre, GaussLobatto, GaussRadauLeft, GaussRadauRight}` and compares every family against a Gauss-Legendre reference at the same `N`, while also recording the shift of the controlling base-side section station. On that full monotonic bundle, all `36/36` cases complete and all `36/36` pass the representative internal sensitivity gate. The worst terminal-moment spread is about `1.17e-4`, the worst pointwise moment spread about `2.41e-4`, the worst tangent spread about `3.73e-4`, the worst secant spread about `1.74e-4`, and the largest controlling-station shift about `4.23e-1` in parent coordinates. The `N=2` slices collapse to zero spread by symmetry, and both response spread and station shift decay as `N` grows. What remains open is the cyclic family-spread closure, not the monotonic matrix.
- Benchmark pending: node-refinement convergence and quadrature-sensitivity claims are still open and must be closed by explicit benchmark suites, not inferred from smoke runs. For node refinement, the monotonic matrix is already closed and a representative cyclic pilot now exists; what remains open is the full cyclic matrix closure over the entire `N=2..10 Ã— {Gauss, Lobatto, Radau-left, Radau-right}` family.
- Prebenchmark caveat: the moment-curvature observable is still defined at the active section station closest to the fixed end rather than at the exact boundary in a continuous sense. The internal preload-consistent closure gate is therefore now closed, but external benchmark closure still depends on node-refinement tables, quadrature sensitivity, and column-level hysteretic reference data.
- Practical correction to the previous benchmark note: the remaining open issue is no longer the existence of the full cyclic node-refinement matrix itself. That matrix already exists; what remains open is closure under a declared continuation policy together with the cyclic quadrature family-spread interpretation.
- The benchmark layer is now frozen explicitly as `claim -> benchmark -> reference class -> acceptance gate`, so the reduced-column campaign no longer jumps directly from runtime output contracts to physical-validation rhetoric.
- Validation correction, April 2026: the cyclic quadrature-family spread bundle now also exists in `src/validation/ReducedRCColumnCyclicQuadratureSensitivityStudy.hh/.cpp`. Under the declared `reversal_guarded_incremental_displacement_control` policy with segment-substep factor `2`, all `36/36` runtime-ready cases complete and `14/36` pass the representative cyclic family-spread gate.
- The current cyclic quadrature frontier is more specific than the earlier text suggested. The worst terminal/history/turning-point moment spread is about `9.42e-1`, the worst tangent-history spread about `1.32e-1`, the worst axial-force-history spread only about `9.25e-7`, the worst secant spread about `8.18e+0` and therefore remains diagnostic-only near reversal crossings, and the worst controlling-station shift is about `4.23e-1` in parent coordinates.
- The family breakdown is now explicit: `GaussLegendre` passes `9/9`, `GaussLobatto` passes `2/9`, `GaussRadauLeft` passes `1/9`, and `GaussRadauRight` passes `2/9`. That is the honest reason arc-length is still not the first move here: we first need to freeze the structural reference family and separate continuation effects from deliberate section-station placement effects.
- The active validation studies are also being compacted structurally: repeated min/max/average reductions are now expressed through small local accumulators and lambda-based group builders instead of hand-written field-by-field copies. That change lives outside the numerical hot path, but it matters scientifically because it reduces maintenance drift between case, node, family, and summary tables.
- The cyclic node-refinement study now follows that same rule too: turning-point detection is expressed through a dedicated helper, worst-case/global reductions through local accumulators, and the representative tolerance projection through short local lambdas. The numerical contract is unchanged, but the postprocessing layer is more compact, easier to audit, and less likely to drift as the validation matrix grows.
- The cyclic quadrature-family study now follows that same rule all the way to its global summary: worst-case reductions are expressed through a dedicated local accumulator, and the candidate/reference spread checks use small local lambdas for max-update and tolerance projection instead of repeated field-by-field clauses. The numerical contract is unchanged, but the postprocessing layer is narrower and easier to audit.
- The monotonic node-refinement and monotonic quadrature-sensitivity bundles now follow that same rule as well: group summaries are projected through compact accumulators, worst-case tracking goes through short local lambdas, and reference-row counts use narrow declarative predicates instead of bespoke loops. Again, this does not alter any benchmark metric; it just makes the validation evidence harder to drift as the matrix grows.
- The reduced-column structural baseline now also routes the low-level Newton/bisection trace through `NonlinearAnalysis::set_incremental_logging(bool)`. When a validation study sets `print_progress = false`, the runtime still evaluates the same slice but stops flooding the harness with incremental solver chatter, so the remaining output is the declared benchmark-level evidence rather than incidental step-by-step noise.
- Practical correction: the remaining open issue is no longer the existence of the cyclic node-refinement or cyclic quadrature-sensitivity matrices. Those bundles already exist. What remains open is closure of the declared base-side observable under the declared continuation policy together with a declared structural reference family and an explicit rationale for any later escalation toward arc-length.
- The evidence-closure layer is now frozen explicitly as `claim -> current artifact -> missing experiment -> closure artifact`, so every open reduced-column claim carries a concrete numerical obligation instead of only a generic â€œfuture benchmarkâ€ label.

The reboot now also has a staged **external computational bridge** for that
same reduced RC column:

- Driver: `scripts/opensees_reduced_rc_column_reference.py`
- Role: independent OpenSeesPy reference for the same single-column geometry,
  axial preload, beam-integration family, and lateral displacement protocol
  used by the `fall_n` reduced structural slice
- Output contract: `reference_manifest.json`, `hysteresis.csv`,
  `section_response.csv`, `moment_curvature_base.csv`, `control_state.csv`,
  and `preload_state.json`
- Scientific status: external computational reference only; it is a bridge
  before later experimental/literature closure, not a substitute for physical
  validation

That bridge is intentionally honest about constitutive equivalence. It no
longer hides one fixed OpenSees mapping; instead it exposes an explicit policy
surface:

- `monotonic-reference`: no-tension `Concrete02` plus baseline `Steel02`
- `cyclic-diagnostic`: reduced-tension `Concrete02` plus tuned `Steel02`
  imported from the uniaxial constitutive audit
- `elasticized-parity`: same fiber layout with elastic materials only

The meaningful comparison is therefore
\[
  \mathcal{O}_{\text{fall\_n}}
  \leftrightarrow
  \mathcal{O}_{\text{OpenSees}}
\]
over shared observables such as base shear versus drift and base-side
moment-curvature, not blind parameter identity.

The current operational policy is also deliberate:

- CI runs the script in `--dry-run` mode only, so the manifest/CSV contract is
  kept under regression without turning OpenSeesPy into a hard dependency of
  the core build.
- A real local smoke path is already verified on Windows with `Python 3.12`
  and `OpenSeesPy 3.8.x`.

Representative local invocation:

```powershell
py -3.12 scripts/opensees_reduced_rc_column_reference.py `
  --analysis cyclic `
  --mapping-policy cyclic-diagnostic `
  --beam-element-family disp `
  --beam-integration legendre `
  --integration-points 3 `
  --geom-transf linear `
  --axial-compression-mn 0.02 `
  --amplitudes-mm 1.25 `
  --steps-per-segment 1 `
  --reversal-substep-factor 2 `
  --max-bisections 4 `
  --output-dir data/output/cyclic_validation/reboot_opensees_reference_cyclic_smoke
```

There is now also a canonical comparative runner:

```powershell
python scripts/run_reduced_rc_column_external_benchmark.py `
  --analysis cyclic `
  --mapping-policy cyclic-diagnostic `
  --beam-nodes 4 `
  --beam-integration legendre `
  --beam-element-family disp `
  --geom-transf linear `
  --axial-compression-mn 0.02 `
  --amplitudes-mm 1.25 `
  --steps-per-segment 1 `
  --reversal-substep-factor 2 `
  --max-bisections 4 `
  --output-dir data/output/cyclic_validation/reboot_external_benchmark_cyclic_smoke
```

That benchmark bundle now writes `fall_n/runtime_manifest.json`,
`fall_n/control_state.csv`, `fall_n/preload_state.json`,
`opensees/reference_manifest.json`, `opensees/control_state.csv`,
`opensees/preload_state.json`, `opensees/comparison_summary.json`,
`timing_summary.csv`, and `benchmark_summary.json`.

The first short computational benchmark is already useful and honest:

- Nothing conceptual is missing anymore to start seeing `fall_n vs OpenSees` benchmark results on the reduced reference slice. The missing piece was the unified runner plus explicit timing surfaces, and that now exists.
- A real bug was found and corrected in the external bridge: geometry and loads were already in `m` and `N`, but concrete/steel parameters were still being fed to OpenSees in `MPa` without conversion to `Pa`.
- A second real bug was then found and corrected in the structural bridge: the exported OpenSees base-shear observable was using the opposite reaction sign convention from the one already used by `fall_n`.
- A third benchmark mismatch was more subtle but just as important: the external structural slice had been anchored on `forceBeamColumn` with `PDelta`, while the current internal reduced-column slice is a small-strain, displacement-based beam benchmark. The parity anchor is now `dispBeamColumn + Linear` transformation.
- The benchmark now also carries a direct spatial-parity audit. On the current audited slice, both codes use the same `92` fiber centroids, the same tributary areas, the same zone/material labels, and the same three Gauss-Legendre section stations within numerical roundoff (`~1e-10` in position, `~1e-12` in area).
- The structural bundle now also carries a station-by-station section-path audit over those same three Gauss-Legendre stations, so the remaining gap is no longer hidden behind a single base observable.
- On the current short cyclic reference slice (`N=4`, Gauss-Legendre, `0.02 MN` axial compression, `1.25 mm` amplitude), both routes complete, the compared history now spans the full `13` sampled points of the guarded displacement path, and both routes report compute time.
- The timing read is auxiliary but worth recording: under the declared
  `cyclic-diagnostic` policy and the displacement-based structural parity
  anchor, `fall_n` now reports about `1.99e-1 s` total wall time, OpenSeesPy
  about `2.16e-1 s`, while the reported-total ratio is about `9.17e-1`.
- The mechanical read is still the real frontier. The declared
  `cyclic-diagnostic` policy now produces a structurally aligned short bundle:
  the base-shear sign is consistent, the fiber cloud and station cloud are
  spatially matched, and the remaining mismatch is no longer attributable to
  section placement. On active steps,
  `max_rel_base_shear_error ~ 2.18e-1`,
  `rms_rel_base_shear_error ~ 1.90e-1`,
  `max_rel_moment_error ~ 2.33e-1`,
  `rms_rel_moment_error ~ 1.96e-1`.
- The new boundary-condition audit makes one point much clearer: the remaining
  structural gap is not caused by a mismatch in the imposed lateral path.
  Across the full sampled history, `actual_tip_drift` closes essentially at
  machine precision (`max_rel_tip_drift_error ~ 6.94e-16`), and the maintained
  base axial reaction also closes essentially exactly
  (`max_rel_base_axial_reaction_error ~ 2.50e-8`).
- What does not close is the preload deformation state. At the preload
  equilibrium point (`step = 0`), `fall_n` gives about `-2.92e-5 m` top axial
  shortening, while OpenSees gives about `-4.35e-5 m`; the relative drift in
  top axial displacement is about `4.94e-1`. The same mismatch is already
  visible in the preload section state: mean axial strain is about
  `-9.12e-6` in `fall_n` versus `-1.36e-5` in OpenSees, and the mean section
  flexural tangent is about `11.74` versus `9.62`. That is a strong hint that
  the benchmark is now being limited by nonlinear section/preload response, not
  by how the displacement Dirichlet path is applied.
- The new station-path audit sharpens that diagnosis. Over the same structural
  slice, the distributed section history now reads
  `max_rel_moment_error ~ 3.40e-1`,
  `rms_rel_moment_error ~ 2.09e-1`,
  `max_rel_curvature_error ~ 1.96e-1`,
  `rms_rel_curvature_error ~ 6.15e-2`,
  `max_rel_tangent_error ~ 1.47e+0`,
  `rms_rel_tangent_error ~ 3.80e-1`,
  `max_rel_axial_force_error ~ 7.47e-2`.
- That localization matters physically: the largest moment/curvature drift sits
  at the loaded-end station (`gp=2`), while the strongest tangent drift sits at
  the base station (`gp=0`). So the remaining mismatch is no longer a vague
  â€œglobal loop problemâ€; it is a distributed nonlinear response mismatch with a
  clear spatial signature. The tangent metric is stricter than in earlier
  notes because the benchmark now compares the same condensed tangent notion on
  both sides, `dM/dkappa|N=const`, instead of mixing that quantity with a raw
  diagonal section-stiffness entry.
- The elasticized structural control on the exact same geometry, fiber cloud,
  station cloud, beam family, and protocol now closes tightly both globally and
  station by station. On the displacement-based comparator,
  `max_rel_base_shear_error ~ 5.17e-3`,
  `max_rel_moment_error ~ 5.17e-3`,
  `max_rel_section_tangent_error ~ 1.65e-9`,
  `max_rel_section_axial_force_error ~ 2.43e-15`. On the force-based
  sensitivity comparator, the global error drops further to
  `~ 1.32e-4`, while the section tangent still stays at `~ 1.65e-9` and the
  section axial-force error at `~ 2.06e-11`.
- That is the strongest current validation read of the bridge: spatial parity,
  beam-family parity, and structural observable extraction are now no longer
  credible primary suspects. The frontier is genuinely nonlinear, and it sits
  in the constitutive/distributed-response layer.
- A deeper structural formulation audit is now frozen in
  `data/output/cyclic_validation/reboot_structural_formulation_audit_summary`.
  Its role is narrower and more useful than another generic benchmark run: it
  compares the four structurally relevant slices
  `{elasticized, nonlinear} x {dispBeamColumn, forceBeamColumn}` over the same
  cantilever, the same fiber cloud, the same three Gauss-Legendre stations, and
  the same guarded displacement protocol.
- That audit makes one architectural point explicit. The current `fall_n`
  reduced structural slice is `TimoshenkoBeamN` with mixed interpolation and
  `DirectAssembly`; no element-level static condensation is active in this
  benchmark. The OpenSees structural comparators are `dispBeamColumn` and
  `forceBeamColumn` with `SectionAggregator(Vy,Vz)`. They are useful structural
  comparators, but they are not literal clones of `fall_n`'s Timoshenko beam
  and they are not the explicit `elasticTimoshenkoBeam` family from OpenSees.
- That distinction no longer needs to be guessed from the code. In the
  elasticized audit, both OpenSees comparators close tightly against `fall_n`:
  `disp` gives `max_rel_base_shear_error ~ 5.17e-3`, while `force` gives
  `~ 1.32e-4`; both keep `section axial-force error` at `~ 1e-11` or better,
  `section tangent error` at `~ 1e-9`, `tip drift error` at machine precision,
  and `tip axial displacement error ~ 2.33e-10`.
- That is the strongest current falsification result on the structural side:
  once constitutive nonlinearity is removed, the benchmark closes. So the
  current mismatch is not primarily a beam-axis, fiber-cloud, station-layout,
  or axial-boundary-condition bug.
- The nonlinear audit sharpens the diagnosis further. Switching OpenSees from
  `dispBeamColumn` to `forceBeamColumn` improves the station-wise axial-force
  consistency by about five orders of magnitude
  (`7.47e-2 -> 7.56e-7`) and also reduces the section-tangent mismatch
  (`1.47 -> 8.02e-1`), but it does not close the global response. In fact, the
  global base-shear mismatch worsens from `~ 2.18e-1` to `~ 2.81e-1`.
- That same comparison also changes how the structural station bundle should
  be read. For the declared nonlinear parity anchor
  `dispBeamColumn + LinearCrdTransf`, station-wise section force/tangent traces
  are now treated as **family-aware diagnostics**, not as unconditional
  acceptance gates, because `dispBeamColumn` is weak-equilibrium at station
  level. Those traces remain useful to localize distributed nonlinear drift,
  but strong pointwise spatial equivalence must be judged with that
  formulation contract in view.
- The honest structural read is therefore now much cleaner: axial-load
  application is already aligned at the support-resultant level, static
  condensation is not active in this slice, and a stronger OpenSees
  beam-column formulation does not magically close the loop. The frontier still
  points back to nonlinear preload plus reversal response, which is consistent
  with the problematic extremal `cover_top` concrete fiber tracked at `step=8`.
- The OpenSees side also deserves to be described fairly. OpenSees does expose
  an `ElasticTimoshenkoBeam` element for elastic shear-flexible controls, and
  we now keep it as a fifth slice in the structural formulation audit. When
  that control is calibrated against the exported elastic section stiffness of
  `fall_n`, it closes the structural elastic benchmark almost exactly:
  `max_rel_base_shear_error ~ 1.32e-4`,
  `max_rel_moment_error ~ 1.32e-4`,
  `max_rel_tangent_error = 0`,
  `max_rel_raw_k00_error = 0`, and
  `max_rel_tip_drift_error = 0`. That is valuable because it shows the
  Timoshenko beam theory itself is not the blocking issue on the elastic
  slice. However, the current nonlinear reduced-column bridge still has to go
  through `dispBeamColumn` and `forceBeamColumn`, because those are the
  documented nonlinear beam-column routes tied to fiber sections, beam
  integration, and the `Linear`/`Corotational` transformation layer. In other
  words, `ElasticTimoshenkoBeam` is a very good elastic control, but it is not
  yet a drop-in nonlinear parity anchor for the audited RC-column slice.
- That same audit now lets us say something positive about `fall_n` without
  overselling it. The current audited advantages are real but narrow:
  `fall_n` gives us a compile-time typed mixed-interpolation Timoshenko path, an
  explicit quadrature family as a model axis, raw and condensed tangent blocks
  (`K00`, `K0y`, `Kyy`) at section level, and direct protocol/reversal/fiber
  diagnostics without leaving the same code base. Runtime is already
  competitive rather than universally superior: on the elasticized
  displacement-based parity slice `fall_n` reports about `0.180 s` versus
  OpenSees `0.216 s`; on the stiffness-equivalent `ElasticTimoshenkoBeam`
  control it reports about `0.161 s` versus `0.184 s`; while on the nonlinear
  displacement-based slice OpenSees is slightly faster (`~0.228 s` versus
  `~0.269 s`). The honest claim is therefore not "always faster", but
  "already competitive while exposing a richer and more auditable formulation
  surface".
- The beam-family roadmap is now explicit too. Two growth paths are worth
  opening after the reference reduced column is closed: a force-based
  Timoshenko family for nonlinear distributed plasticity under shear-flexible
  kinematics, and a geometrically exact beam family of Simo-Reissner type for
  large-rotation, objective beam kinematics. Both are now treated as deferred
  research tracks rather than hidden assumptions inside the current benchmark.

The audit now also has canonical figures:

- `doc/figures/validation_reboot/reduced_rc_structural_formulation_audit_errors.png`
- `doc/figures/validation_reboot/reduced_rc_structural_formulation_audit_timing.png`

The same bundle can now be plotted reproducibly:

```powershell
py -3.11 scripts/plot_reduced_rc_external_benchmark.py `
  --bundle data/output/cyclic_validation/reboot_external_benchmark_cyclic_bc_audit `
  --figures-dir doc/figures/validation_reboot `
  --secondary-figures-dir PhD_Thesis/Figuras/validation_reboot
```

This emits overlay figures for structural hysteresis, base-side
moment-curvature, station-by-station section-path parity, the new
boundary-control/preload trace, and reported wall times. On the current local
audit, the plotting runtime is `Python 3.11` because that environment already
carries `matplotlib`, while the OpenSeesPy bridge remains audited on
`Python 3.12`.

To keep that gap interpretable, the reboot now also has a **section-only**
external computational bridge:

- internal runner: `fall_n_reduced_rc_column_section_reference_benchmark`
- orchestrator: `scripts/run_reduced_rc_column_section_external_benchmark.py`
- external bridge mode: `scripts/opensees_reduced_rc_column_reference.py --model-kind section`
- OpenSees side: `zeroLengthSection` carrying the same reduced RC fiber
  section, driven by constant axial compression plus monotonic curvature control

Representative local invocation:

```powershell
python scripts/run_reduced_rc_column_section_external_benchmark.py `
  --output-dir data/output/cyclic_validation/reboot_external_section_benchmark_policy_smoke `
  --mapping-policy monotonic-reference `
  --axial-compression-mn 0.02 `
  --max-curvature-y 0.03 `
  --section-steps 40
```

That section-only bundle is already very informative:

- The first implementation exposed a real bug in the OpenSees bridge: for
  `zeroLengthSection`, `eleResponse(..., "force")` returns nodal end forces,
  not the condensed section-force vector. Reading that response as if it were
  `[P, M_z, M_y, ...]` produced a false axial-force mismatch of `100%`.
- After correcting that mapping, axial-force closure became essentially exact
  (`max_rel_axial_force_error â‰ˆ 3.9e-7`), which is strong evidence that the
  axial preload and sign conventions are now aligned between the two codes.
- The moment-curvature gap is much smaller than in the structural benchmark,
  but still not closed: on the current smoke slice the section benchmark gives
  `max_rel_moment_error â‰ˆ 5.53e-1` and `rms_rel_moment_error â‰ˆ 1.04e-1`.
- The remaining frontier is therefore sharper than before:
  `fall_n vs OpenSees` is no longer dominated by gross axial inconsistency at
  section level; it is now dominated by early-branch flexural response and
  tangent mismatch (`max_rel_tangent_error ~ 6.62e-1`,
  `rms_rel_tangent_error ~ 1.44e-1` on the same slice) once the benchmark is
  forced to compare the same condensed tangent on both sides.
- The timing observable is already useful here too:
  `fall_n` reports about `6.93e-2 s` total wall time, OpenSeesPy about
  `1.67e-1 s`, and the process-level ratio is about `3.01e-1` on the current
  local audited runtime.

The same section bridge now also has a short **cyclic** smoke slice:

```powershell
python scripts/run_reduced_rc_column_section_external_benchmark.py `
  --analysis cyclic `
  --output-dir data/output/cyclic_validation/reboot_external_section_benchmark_cyclic_spatial_parity `
  --mapping-policy cyclic-diagnostic `
  --axial-compression-mn 0.02 `
  --max-curvature-y 0.03 `
  --section-amplitudes-curvature-y 0.01 `
  --steps-per-segment 4 `
  --max-bisections 4
```

That cyclic section bundle changes the interpretation in a useful way:

- The cyclic mismatch is already visible at section level, but it is now much
  more tightly localized once spatial parity is audited explicitly:
  `max_rel_moment_error ~ 5.91e-2`,
  `rms_rel_moment_error ~ 3.59e-2`,
  `max_rel_tangent_error ~ 7.31e-1`,
  `rms_rel_tangent_error ~ 1.85e-1`.
- The tangent plot now has a much sharper interpretation. Splitting the same
  bundle into nonzero-curvature branch points versus zero-curvature anchors
  shows:
  `branch-only max_rel_tangent_error ~ 6.72e-2`,
  `branch-only rms_rel_tangent_error ~ 3.12e-2`,
  while the large peak remains concentrated in the three zero-curvature anchor
  states.
- Axial-force closure remains essentially exact:
  `max_rel_axial_force_error ~ 1.10e-7`.
- The large tangent max is no longer a convention artifact. After aligning the
  observable to the same condensed tangent `dM/dkappa|N=const` in both codes,
  the remaining peak is concentrated at the preload state and, more sharply,
  at the exact return to zero curvature after reversal. Away from those two
  singular states, the nonzero-curvature branch tangents are much closer.
- The new tangent-diagnostic CSVs make that statement testable instead of
  rhetorical:
  `fall_n/section_tangent_diagnostics.csv` and
  `opensees/section_tangent_diagnostics.csv` export the raw axial-flexural
  block (`K00, K0y, Ky0, Kyy`) together with direct, left, right, and centered
  numerical tangents. On the reversal-return state, both left and right
  numerical slopes stay positive (`fall_n ~ 4.10/4.16`, OpenSeesPy
  `~ 4.03/3.93`), so the residual discrepancy no longer looks like a sign
  inversion. It is a localized condensation/state issue: OpenSees condenses
  `Kyy ~ 4.31` down to `~ 1.68` because the axial-flexural coupling term is
  stronger (`K0y ~ Ky0 ~ 50.6`, `K00 ~ 975`), while `fall_n` condenses
  `Kyy ~ 7.45` down to `~ 6.27` with a milder coupling
  (`K0y ~ Ky0 ~ 44.8`, `K00 ~ 1699`).
- A deeper fiber-state audit then exposed an additional convention issue that
  the section-level `M-kappa` plot could not reveal by itself: under the raw
  OpenSees fiber readout, positive reported `kappa_y` was compressing the
  opposite cover face from the one used by `fall_n`'s plane-section convention.
  This is now handled explicitly in the comparison layer by reading OpenSees
  fiber states under the exported section observable, but projecting them into
  the `fall_n` curvature/cover convention before matching fibers spatially.
- That convention fix collapsed the apparent fiber-level gap dramatically on
  the same cyclic slice. The mapped fiber comparison now gives
  `max_rel_fiber_stress_error ~ 2.59e-1`,
  `rms_rel_fiber_stress_error ~ 9.72e-2`,
  `max_rel_fiber_tangent_error ~ 1.00`,
  `rms_rel_fiber_tangent_error ~ 1.00e-1`,
  and, on the zero-curvature anchors only,
  `max_rel_fiber_stress_error ~ 6.25e-1`,
  `rms_rel_fiber_stress_error ~ 5.28e-1`,
  `max_rel_fiber_tangent_error ~ 5.12e-1`,
  `rms_rel_fiber_tangent_error ~ 9.52e-2`.
- The remaining anchor discrepancy is therefore much sharper than before. It is
  no longer a diffuse sign mismatch over the whole section; it is localized at
  the reversal-return anchor and is dominated by the cover-concrete residual
  state, especially the `cover_top` contribution in the `fall_n` convention,
  while the compressive branch-side cover response aligns much more closely
  once the convention is fixed.
- The cyclic section timing is now also part of the benchmark bundle:
  `fall_n ~ 2.04e-2 s`,
  `OpenSeesPy ~ 1.77e-1 s`,
  reported-total ratio `~ 1.15e-1`.
- The benchmark now also carries an explicit section control trace on both
  sides. That closes an important methodological question: the compared
  pseudo-time schedule and the actually achieved curvature schedule now match
  essentially to machine precision,
  `max_rel_target_curvature_error ~ 2.31e-16`,
  `max_rel_actual_curvature_error ~ 2.31e-16`,
  `max_rel_delta_actual_curvature_error ~ 8.67e-16`,
  and `max_rel_pseudo_time_error = 0`.
- The guarded increment policy is also no longer a hidden suspect on the
  audited short cyclic section slice. Both routes accept the same physical
  substep schedule:
  `max_abs_accepted_substep_error = 0`,
  `max_abs_bisection_level_error = 0`,
  with `accepted_substep_count = 1` and `max_bisection_level = 0` on every
  non-initial step.
- The iterative work is *not* identical even under that matched control
  schedule. The Newton iteration histories differ on `15/17` shared section
  states, with `max_abs_newton_iteration_error = 4` and
  `rms_abs_newton_iteration_error ~ 2.03`. That is scientifically useful:
  the remaining tangent/anchor gap is occurring under the same physical
  benchmark path, but the two codes still traverse that path with different
  local nonlinear work.
- OpenSees `domainTime` is now treated as a diagnostic only, not as a
  physically comparable benchmark time. Under static `DisplacementControl`,
  OpenSees advances an internal pseudo-time/load-factor state tied to the
  active load patterns and the reference-load system, so its reported
  `domainTime` on the short cyclic section slice becomes large and
  non-monotone even while the benchmark pseudo-time remains exactly aligned.
  The current audited bundle reports
  `domain_time_monotone = false`,
  `final_domain_time ~ 3.52e+1`,
  `max_abs_domain_vs_pseudo_time ~ 3.85e+4`,
  and `max_abs_increment_vs_pseudo_increment ~ 1.08e+4`.
- That also gives a fairly clear architectural recommendation for `fall_n`.
  Copying OpenSees-style non-monotone `domainTime` semantics would be
  high-cost and low-return here: it would blur the physical meaning of the
  current checkpointable control parameter without addressing the actual
  benchmark gap. The part worth borrowing is narrower and more useful:
  richer protocol metadata, explicit reversal/branch traces, stronger
  continuation diagnostics, and adaptive increment policies, while keeping the
  benchmark pseudo-time itself monotone and physically declared.
- The control trace now exports exactly that narrower seam on both sides:
  `target_increment_direction`, `actual_increment_direction`,
  `protocol_branch_id`, `reversal_index`, `branch_step_index`, and
  `newton_iterations_per_substep`. On the audited short cyclic section slice,
  those new protocol semantics close exactly between `fall_n` and OpenSees.
  That matters because it tells us the remaining solver-effort gap is *not*
  caused by one code traversing a different reversal schedule.
- The new anchor read is especially useful at the problematic step
  `step = 8`: both routes are on the same `protocol_branch_id = 2`,
  `reversal_index = 1`, with the same negative increment direction and the
  same single accepted substep. Even there, OpenSees uses `4` Newton
  iterations while `fall_n` uses `5`. So there is something genuinely worth
  learning from OpenSees in continuation diagnostics and local solver
  efficiency, but not a good reason to import its non-monotone `domainTime`
  as the benchmark time variable.
- The structural cyclic gap therefore cannot now be attributed only to beam
  continuation or station placement. The section bridge is now spatially
  aligned and mechanically much closer than the structural bundle, so the
  frontier is more precise: the cyclic constitutive/section bridge remains open
  under reversal, and only on top of that sits the beam/distributed-response
  layer.

To make that diagnosis more rigorous, the same section benchmark now also
supports an `elasticized` material mode on both sides:

```powershell
python scripts/run_reduced_rc_column_section_external_benchmark.py `
  --output-dir data/output/cyclic_validation/reboot_external_section_benchmark_elastic_policy_smoke `
  --material-mode elasticized `
  --mapping-policy elasticized-parity `
  --axial-compression-mn 0.02 `
  --max-curvature-y 0.03 `
  --section-steps 40
```

That control slice is extremely valuable because it separates geometry and
observable parity from constitutive parity:

- In the elasticized bundle, the same fiber layout closes essentially at
  machine precision:
  `max_rel_moment_error â‰ˆ 4.01e-9`,
  `max_rel_tangent_error â‰ˆ 1.65e-9`,
  `max_rel_axial_force_error â‰ˆ 5.20e-14`.
- The structural/section benchmark machinery is therefore no longer the main
  suspect. The remaining nonlinear section gap is now localized much more
  honestly in the external constitutive mapping (`KentParkConcrete` vs
  `Concrete02`, `MenegottoPintoSteel` vs `Steel02`) and in how that mapping
  shapes the early flexural branch and tangent.
- Timing also stays explicit in the control slice:
  `fall_n` reports about `1.93e-2 s` total wall time and OpenSeesPy about
  `8.98e-2 s` on the current audited runtime.

The next constitutive step is now also frozen as a dedicated **uniaxial
material bridge**:

- internal runner: `fall_n_reduced_rc_material_reference_benchmark`
- orchestrator: `scripts/run_reduced_rc_material_external_benchmark.py`
- external bridge: `scripts/opensees_reduced_rc_material_reference.py`
- OpenSees side: `testUniaxialMaterial` over `Steel02` and `Concrete02`, so
  the comparison stays constitutive and does not reintroduce section or beam
  effects

Representative local invocation:

```powershell
python scripts/run_reduced_rc_material_external_benchmark.py `
  --output-dir data/output/cyclic_validation/reboot_external_material_benchmark_smoke `
  --python-launcher "py -3.12" `
  --steps-per-branch 24
```

This bundle is already giving a very sharp read:

- `steel_monotonic` closes essentially exactly:
  `max_rel_stress_error â‰ˆ 1.54e-6`,
  `max_rel_tangent_error â‰ˆ 3.24e-5`,
  `max_rel_energy_error â‰ˆ 4.39e-7`.
- `steel_cyclic` does **not** close even under the guarded two-level protocol:
  `max_rel_stress_error â‰ˆ 1.29e+1`,
  `max_rel_tangent_error â‰ˆ 9.90e-1`,
  `max_rel_energy_error â‰ˆ 9.93e+1`.
- `concrete_monotonic` remains materially different:
  `max_rel_stress_error â‰ˆ 1.19`,
  `max_rel_tangent_error â‰ˆ 3.28e-1`.
- `concrete_cyclic` also remains open:
  `max_rel_stress_error â‰ˆ 3.26`,
  `max_rel_tangent_error â‰ˆ 1.10e+1`,
  `max_rel_energy_error â‰ˆ 3.04e-1`.

That split is scientifically useful. The external bridge now says, much more
precisely than the section benchmark alone:

- the monotonic steel mapping is already essentially aligned;
- the remaining nonlinear disagreement is dominated by cyclic steel history and
  by the concrete mapping, not by section geometry or observable extraction;
- the material tester in the current OpenSeesPy runtime becomes non-finite for
  more aggressive multi-level steel cycles, so the canonical steel-cyclic
  benchmark is intentionally frozen at a guarded two-level protocol instead of
  pretending that a numerically unstable path is a valid scientific reference.

The timing surface is also now explicit at constitutive level:

- `steel_monotonic`: `fall_n â‰ˆ 8.51e-4 s`, OpenSeesPy `â‰ˆ 4.03e-3 s`
- `steel_cyclic`: `fall_n â‰ˆ 1.94e-3 s`, OpenSeesPy `â‰ˆ 1.50e-2 s`
- `concrete_monotonic`: `fall_n â‰ˆ 6.46e-4 s`, OpenSeesPy `â‰ˆ 3.28e-3 s`
- `concrete_cyclic`: `fall_n â‰ˆ 2.86e-3 s`, OpenSeesPy `â‰ˆ 1.82e-2 s`

So the benchmark frontier is no longer vague. Before reinterpreting the
remaining `fall_n vs OpenSees` section gap, we now have to close or at least
honestly bound the constitutive bridge itself.

That constitutive frontier is now also audited explicitly through:

- `scripts/run_reduced_rc_material_mapping_audit.py`

Representative local invocation:

```powershell
python scripts/run_reduced_rc_material_mapping_audit.py `
  --output-dir data/output/cyclic_validation/reboot_material_mapping_audit_smoke `
  --python-launcher "py -3.12" `
  --steps-per-branch 24
```

This audit does not reopen section or beam effects. It keeps the comparison at
uniaxial level and asks a stricter question: *is the remaining gap mainly a
parameter-mapping issue inside an admissible OpenSees family, or a genuine
family-level constitutive mismatch?*

The current audit already sharpens that answer:

- For cyclic steel, a small `Steel02` grid search improves the bridge but does
  not close it. The best audited profile (`R0=30`, `cR1=8`, `cR2=0.3`) lowers
  the RMS errors to about `1.67` in stress, `3.65e-1` in tangent, and `4.75`
  in energy, but that is still far from a credible closure reference.
- For monotonic concrete, the current `Concrete02` reference profile remains
  the best external bridge among the audited candidates.
- For cyclic concrete, a low-tension `Concrete02` profile is currently the
  best audited diagnostic bridge: it improves the tangent RMS materially
  relative to the default `Concrete02` path, even though stress and energy do
  not close perfectly.

The practical consequence is important. The next section-level and column-level
external benchmarks should not treat `Concrete02/Steel02` as a fixed dogma:

- keep a no-tension `Concrete02` plus baseline `Steel02` as the
  `monotonic-reference` bridge,
- keep a reduced-tension `Concrete02` plus tuned `Steel02` as the current
  `cyclic-diagnostic` bridge,
- and do **not** treat tuned `Steel02` as sufficient closure for cyclic steel
  yet.

That means the next structural comparison can be more honest: it should use
the constitutive audit to declare the external mapping policy explicitly before
asking the section or the reduced column to â€œmatch OpenSeesâ€.

The next localization layer is now also explicit:

- `scripts/run_reduced_rc_problematic_fiber_replay_audit.py`

This runner replays the exact strain history of the most problematic audited
section fiber through both uniaxial bridges. The current audited bundle
(`data/output/cyclic_validation/reboot_problematic_fiber_replay_audit`) says
something much sharper than the earlier global loop mismatch:

- the selected fiber sits at the reversal-return anchor (`step = 8`) in the
  `cover_top` unconfined-concrete zone;
- after projecting the OpenSees fiber cloud to the declared `fall_n`
  convention, the two section codes differ only mildly in that fiberâ€™s strain
  history (`max_abs_strain_difference ~ 5.02e-5`);
- but under that same near-zero compressive strain the uniaxial responses do
  **not** collapse to the same state:
  `fall_n` returns about `0 MPa` with `Et ~ 3.0e-2 MPa`, while OpenSees
  `Concrete02` returns about `+4.36e-1 MPa` with `Et ~ -6.0e4 MPa` on the
  exact `fall_n` replay history;
- the same pattern repeats on the OpenSees strain history: `fall_n` stays on
  the cut-off state while OpenSees retains a tensile/softening residual state
  near the zero-crossing anchor.

That is the most useful current read of the nonlinear gap:

- the active compressive branch is no longer the main mystery;
- the dominant disagreement is localized to the crack-closure /
  reversal-return regime of the unconfined concrete cover;
- the replay now also confirms that this constitutive disagreement happens
  under the same benchmark branch and reversal semantics, not because the two
  codes walked through different non-monotone step agendas;
- so the next meaningful improvement is not to mimic OpenSees `domainTime`,
  but to enrich the concrete cyclic law in `fall_n` or introduce a more
  OpenSees-like comparison model for that near-zero unloading/reloading
  regime.

That localization is now followed by an explicit **cyclic amplitude-escalation
audit**:

- `scripts/run_reduced_rc_amplitude_escalation_audit.py`

One infrastructure bug had to be removed before trusting this audit again. The
amplitude-escalation orchestrator was not propagating its declared
`--python-launcher` into the nested section and structural benchmark runners,
so a headless rerun could silently fall back to a broken ambient `py -3.12`
launcher. The same script also assumed an interactive matplotlib backend. Both
issues are now fixed in
[`scripts/run_reduced_rc_amplitude_escalation_audit.py`](/c:/MyLibs/fall_n/scripts/run_reduced_rc_amplitude_escalation_audit.py):
the nested runners inherit the explicit launcher, and figure generation now
defaults to a non-interactive `Agg` backend. That matters scientifically
because it separates a real OpenSees frontier from a wrapper failure.

Representative local invocation:

```powershell
py -3.11 scripts/run_reduced_rc_amplitude_escalation_audit.py `
  --output-dir data/output/cyclic_validation/reboot_external_amplitude_escalation_solver_policy_shared `
  --mode both `
  --print-progress
```

This audit is deliberately split into two coupled but distinct questions:

- **section-level external escalation**, where the current nonlinear frontier
  is already localized;
- **structural external escalation** on the declared reference family
  (`N=4`, Gauss-Legendre, displacement-based parity anchor, reversal-guarded
  displacement control).

The current audited read is now sharper and more asymmetric than the earlier
one:

- section-level nonlinear external bundles still complete through
  `kappa_y = 0.015 1/m`;
- the first section-level external failure still appears at
  `kappa_y = 0.020 1/m`, and it still fails on the OpenSees
  `zeroLengthSection` side;
- on the **combined structural external benchmark**, the declared reference
  slice is no longer limited by the old single-profile OpenSees solve; under
  the audited solver-policy cascade plus `max_bisections = 8`, the combined
  structural bundle now completes at `20, 35, 50, 100 mm`;
- the first combined failure still appears at `75 mm`, and it is still on the
  OpenSees side, not on the `fall_n` side;
- that makes the current external structural frontier **non-monotone** rather
  than simply â€œclosed through X mmâ€: `75 mm` remains a pathological OpenSees
  bridge case, `100 mm` is reopened, and `125/150 mm` fail again on the
  OpenSees side;
- the reopened larger-amplitude probes also show that the internal `fall_n`
  slice remains controlled beyond that OpenSees bridge frontier: the canonical
  `125 mm` and `150 mm` bundles both complete on the `fall_n` side in about
  `0.412 s` and `0.457 s` reported total wall time, while the OpenSees bridge
  fails before reaching protocol point `step=10`
  (`target_drift = -62.5 mm` and `-75 mm`, respectively);
- on the same audited structural amplitude sweeps, the internal `fall_n`
  structural slice still completes the broader `11/11` single-amplitude cases
  through `400 mm`, while the current robust external OpenSees bridge completes
  `4/7` cases on the focused `20..150 mm` sweep (`20, 35, 50, 100 mm`).

That split matters scientifically. The amplitude frontier is no longer one
vague "large-amplitude failure"; it is now three distinct gates:

- the **external section bridge** loses robustness first, at
  `0.020 1/m`, under the current `cyclic-diagnostic` constitutive policy;
- the **combined structural external benchmark** is currently limited by a
  non-monotone OpenSees structural bridge frontier whose first failure still
  occurs at `75 mm`, even though `100 mm` is already reachable under the
  declared solver-policy cascade;
- the **internal fall_n structural slice** is no longer the blocker on this
  front and now completes, on the declared reversal-guarded path, through the
  largest audited single-amplitude case `400 mm`.

The error/timing growth is also already visible in the audit figures:

- [section amplitude errors](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_section_amplitude_escalation_errors.png)
- [section amplitude timing](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_section_amplitude_escalation_timing.png)
- [structural amplitude errors](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_structural_amplitude_escalation_errors.png)
- [structural amplitude timing](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_structural_amplitude_escalation_timing.png)
- [structural family errors](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_structural_amplitude_family_errors.png)
- [structural family timing](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_structural_amplitude_family_timing.png)
- [structural family frontier](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_structural_amplitude_family_frontier.png)

A section-only rerun of the canonical amplitude audit, now with explicit
launcher propagation fixed, confirms the earlier mechanical read instead of
changing it:

- [section-only canonical rerun](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_external_section_amplitude_escalation_axial_admissibility/amplitude_escalation_summary.json)
- the combined section comparator still closes at `kappa_y = 0.010, 0.015 1/m`
  and still fails first at `0.020 1/m` on the OpenSees `zeroLengthSection`
  side;
- the completed section bundles also make the efficiency contrast explicit:
  at `0.010 1/m` `fall_n` reports about `0.056 s` total wall time versus
  `0.274 s` in OpenSees, and at `0.015 1/m` about `0.050 s` versus `0.261 s`;
- so the current section frontier is now simultaneously a mechanical result
  and a reproducible software result, not an artifact of launcher drift.

That read had to be tightened one layer further before it could be trusted.
The external OpenSees section driver had been treating `ops.analyze(1) == 0`
as an accepted displacement-control step, but a deeper audit showed that this
criterion was too weak near cyclic reversals: some quasi-Newton profiles could
return â€œsuccessâ€ while the controlled curvature moved in the wrong direction
or by an absurd amount. The driver now enforces a **directional and magnitude
admissibility contract** in
[`scripts/opensees_reduced_rc_column_reference.py`](/c:/MyLibs/fall_n/scripts/opensees_reduced_rc_column_reference.py):

- a converged step is accepted only if the achieved control-DOF increment has
  the same sign as the requested increment;
- and only if its magnitude remains consistent with the requested
  `DisplacementControl` step;
- the accepted replay path is preserved explicitly through recursive cutbacks;
- and the same admissibility diagnostics are exported into the section and
  structural control traces.

This was not cosmetic. Once that acceptance contract was enforced, the old
false frontier at the first reversal disappeared and the benchmark could be
tracked much deeper into the cycle. The later failure at `step = 23` remains
useful only as a historical debugging milestone because it proved the wrapper
itself was no longer lying about the control increment. It is no longer the
operative section frontier once the stronger mechanical admissibility rule
below is enforced. To keep that result reproducible on Windows, the benchmark
runners now also normalize launcher strings through
[`scripts/python_launcher_utils.py`](/c:/MyLibs/fall_n/scripts/python_launcher_utils.py)
instead of relying on brittle `shlex` parsing of quoted absolute paths.

Validation update, April 2026: that directional-and-magnitude fix closed only
the first half of the methodological gap. The section benchmark is declared as
an audit of `dMy/dkappa_y | N = const`, so a step cannot be considered valid
merely because it satisfies the curvature control. It must also remain on the
constant-axial-force slice. The OpenSees bridge now enforces that second
admissibility contract too: accepted section states must satisfy an explicit
axial-force residual tolerance in addition to the curvature-control
admissibility.

Under that stronger and mechanically correct contract, the interpretation of
the canonical `kappa_y = 0.020 1/m` failure changes again. The old later-cycle
failure at `step = 23` remains useful as a historical debugging milestone
because it proved the wrapper itself was no longer lying, but it is no longer
the operative section frontier. The operative frontier is now:

- [canonical axial-admissible failed bundle](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_external_section_kappa0p02_axial_admissibility_audit/benchmark_summary.json)
- `failure_step = 9`
- `failure_target_curvature_y = +1.75e-02 1/m`

This is the first reversal. In other words, once we require the OpenSees slice
to stay on the declared `N = const` path, the comparator does not complete the
first unloading increment of the `0.020 1/m` cycle.

That stronger criterion also sharpens the positive read of the accepted branch:

- on the shared accepted branch up to `kappa_y = 0.020 1/m`, axial-force
  closure is essentially exact (`max_rel_axial_force_error ~ 2.05e-7`);
- moment-curvature mismatch on that accepted branch is modest
  (`max_rel_moment_error ~ 3.00e-2`);
- fiber-stress mismatch remains moderate
  (`max_rel_fiber_stress_error ~ 1.68e-1`);
- and `fall_n` remains materially faster on those section bundles:
  about `0.101 s` vs `0.593 s` at `0.010 1/m`, and
  about `0.109 s` vs `0.628 s` at `0.015 1/m`.

So the current section frontier should now be read much more honestly: it is
not the largest curvature where OpenSees returns nominal solver success, but
the largest curvature amplitude for which the external section bridge still
follows the declared cyclic path and remains on the constant-axial-force
slice. On the current audited matrix, the combined section comparator closes
at `kappa_y = 0.010, 0.015 1/m` and fails first at `0.020 1/m` on the
OpenSees side.

That read was pushed one layer deeper through a dedicated reversal-frontier
profile audit:

- `scripts/run_reduced_rc_section_reversal_frontier_audit.py`
- [reversal frontier summary](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_section_reversal_frontier_audit/reversal_frontier_summary.json)

At the first failing reversal of the section comparator, no materially
plausible single-profile override rescues the OpenSees slice. The declared
`cyclic-diagnostic` bridge, `Concrete02` with zero tension, `Concrete02` with
`lambda = 0.5`, and a `Concrete01` Kent-Park-like control all fail at the same
first unloading step:

- `failure_step = 9`
- `failure_target_curvature_y = +1.75e-02 1/m`
- `failure_trial_actual_curvature_y = +2.00e-02 1/m`

This matters because it closes a tempting but wrong shortcut. The problematic
fiber replay did show that zero tension in the external concrete bridge can
match the extremal cover-fiber history much more closely, but the section
reversal audit shows that this alone does not rescue the zeroLengthSection
benchmark under the full `N = const` contract. So the current external
section frontier is no longer best read as a pure uniaxial material-mapping
issue; it is a section-level reversal frontier of the external comparator.

The structural family sweep itself is frozen in:

- `scripts/run_reduced_rc_structural_amplitude_family_audit.py`
- [structural amplitude family summary](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_structural_amplitude_family_audit/structural_amplitude_family_audit_summary.json)

On the current audited sweep:

- section moment-curvature error grows from about `5.06e-2` at `0.010 1/m`
  to about `1.09e-1` at `0.015 1/m`;
- section branch-only tangent error is of order `5.60e-1` at `0.010 1/m`
  and about `3.45e-1` at `0.015 1/m`;
- on the current structural external amplitude-escalation bundle,
  the combined comparator closes at `20, 35, 50 mm` and also at `100 mm`
  under the declared solver-policy cascade, with
  hysteresis/base-moment mismatch about
  `1.52e-1 / 1.39e-1`, `8.46e-2 / 8.45e-2`, and
  `6.63e-2 / 8.23e-2` at `20/35/50 mm`, while the reopened `100 mm` case
  already shows much larger mismatch (`2.21e+0 / 1.61e+0`);
- the same bundle now reports
  `highest_combined_completed_amplitude = 100 mm`,
  `highest_fall_n_completed_amplitude = 150 mm` on the focused external sweep,
  and `highest_opensees_completed_amplitude = 100 mm`;
- so the present larger-amplitude frontier is no longer best read as a
  `fall_n` continuation barrier. It is now primarily an
  **OpenSees-comparator frontier**, and a non-monotone one.

I also probed stronger internal continuation on the `15 mm` structural case:

That earlier `15 mm` frontier should now be read as a historical debugging
milestone, not as the current structural limit. A deeper root-cause audit
showed that the decisive instability was not only in the continuation logic
but in the concrete tension crack-onset law itself: the reduced RC reference
section was crossing from elastic tension to softening with a hard tangent
jump. The benchmark now regularizes that onset in
`src/materials/constitutive_models/non_lineal/KentParkConcrete.hh` through a
`C^1` transition from the elastic branch to the tensile-softening branch.

That is a solution of fondo rather than a local patch. Before accepting it,
the reduced structural slice was re-audited with:

- frozen-state section and element finite-difference tangents;
- internal uniaxial history export on the tracked cover-concrete fibers;
- repeated `100 mm` frontier probes on the same declared structural slice.

The new read is materially different:

- the old `100 mm` barrier is gone;
- the same declared internal structural slice now completes at
  `100, 125, 150, 200, 250, 300, 400 mm`;
- the solver-health trace remains controlled on those reopened amplitudes.

On the current internal high-amplitude probes:

- `100 mm`: `max_iter = 7`, `avg_iter â‰ˆ 4.85`, `max_bisection_level = 1`;
- `125 mm`: `max_iter = 8`, `avg_iter â‰ˆ 4.77`, `max_bisection_level = 0`;
- `150 mm`: `max_iter = 17`, `avg_iter â‰ˆ 5.08`, `max_bisection_level = 0`;
- `200 mm`: `max_iter = 10`, `avg_iter â‰ˆ 6.08`, `max_bisection_level = 0`;
- `250 mm`: `max_iter = 12`, `avg_iter â‰ˆ 5.92`, `max_bisection_level = 1`;
- `400 mm`: `max_iter = 9`, `avg_iter â‰ˆ 4.85`, `max_bisection_level = 1`.

So the reopened path is not being held together by runaway bisection or by a
fragile Newton trace. The present blocker is no longer the internal
displacement-driven `fall_n` algorithm on this slice.

I also rechecked PETSc/SNES line-search sensitivity at `400 mm`:

- `basic` keeps the lowest Newton work (`avg_iter â‰ˆ 4.85`, `max_iter = 9`);
- `bt` uses more Newton work (`avg_iter â‰ˆ 6.31`, `max_iter = 19`) but was
  faster in wall time on that single case;
- `l2` lands in between on Newton work (`avg_iter â‰ˆ 5.85`, `max_iter = 11`)
  and is the slowest in wall time of the three.

That old single-line-search read is now superseded by a more robust internal
PETSc policy in the reduced structural benchmark. The benchmark no longer
forces a global `-snes_linesearch_type basic`; instead it declares an
object-local SNES profile cascade inside `NonlinearAnalysis`:

- `newton_backtracking` = `newtonls + bt + preonly/lu`
- `newton_l2` = `newtonls + l2 + preonly/lu`
- `newton_trust_region` = `newtontr + preonly/lu`

This is a more fundamental fix than tweaking one PETSc flag in the validation
driver:

- it keeps the benchmark on the same monotone pseudo-time path;
- it avoids polluting the global PETSc option database with a fragile default;
- and it records, per accepted runtime step, which SNES profile actually
  carried the solve.

On the current internal high-amplitude probes, the declared PETSc profile
cascade keeps the structural slice alive through at least `400 mm`, and the
harder reversed steps now leave an auditable trail instead of silently relying
on one line-search choice. For example:

- `150 mm`: all accepted steps still converge, and the slice remains on
  `newton_backtracking`;
- `250 mm`: one reversed step escalates from `newton_backtracking` to
  `newton_l2` and still converges cleanly;
- `400 mm`: the cycle still completes with a controlled Newton trace and no
  loss of the imposed-vs-total-state displacement consistency.

So the present blocker is no longer an under-specified PETSc Newton policy on
the `fall_n` side. The main open frontier remains the external comparator and,
behind it, the constitutive/seccional mismatch already localized in the
cover-concrete reversal state.

What changed in a deeper sense is the solver architecture, not just one
benchmark flag. The imposed displacement path in `fall_n` was re-audited at
the model and PETSc levels and the critical point is now explicit: the
residual/Jacobian already operate on the total state
`u_total = u_global + u_imposed`, so the robust fix was not to "patch" the
Dirichlet update but to remove the benchmark's dependence on one fragile
global PETSc option. That is why the active policy now lives in a shared
header, `src/analysis/NonlinearSolvePolicy.hh`, and why the structural
benchmark declares its SNES profiles object-locally instead of relying on
ambient command-line defaults.

That policy layer is also no longer stringly-typed at its core. The active
profiles are now expressed through typed method and line-search families
(`newton_line_search`, `newton_trust_region`, `quasi_newton`,
`nonlinear_gmres`; `backtracking`, `l2`, `basic`, `none`) and only mapped to
PETSc names at the application boundary. That keeps the extension point honest:
future non-Newton routes can be introduced without scattering PETSc string
literals across validation code or analysis drivers.

That shared policy layer is now also wired into `DynamicAnalysis` through
`TSGetSNES`. The dynamic path does not yet retry a full profile cascade step by
step the way `NonlinearAnalysis` does, but it already exposes the same
extension point and the same profile type. This is deliberate: the thesis is
ultimately centered on dynamics, so the solver story must remain consistent
across quasi-static and transient paths. In practice this means the current
code base is ready for future quasi-Newton (`qn`), nonlinear-GMRES
(`ngmres`), stronger trust-region, or even Levenberg-Marquardt-style
least-squares strategies behind the same declared interface, without hard-coding
one family into the material, element, or validation layers.

That statement is no longer just architectural intent. The dynamic runtime path
is now rechecked directly in `fall_n_steppable_dynamic_test.exe`: the active
trust-region profile reaches the `SNES` embedded in PETSc `TS` and the time
step still converges cleanly. I am still being careful about the scope of that
claim, though. What is validated end-to-end today is the shared extension point
plus one real dynamic trust-region profile; quasi-Newton, nonlinear-GMRES, and
Levenberg-Marquardt-style routes remain explicitly open growth paths, not
validated dynamic production baselines yet.

That shared solve surface has now been exercised on a canonical internal cyclic
bundle all the way to `200 mm`, frozen by
`scripts/run_reduced_rc_internal_hysteresis_200mm.py` in
`data/output/cyclic_validation/reboot_internal_hysteresis_200mm`. The important
point is not just that the run completes, but how it completes: under the
declared `canonical-cascade` policy the slice reaches `u_max = 200 mm` with
`V_max â‰ˆ 25.05 kN`, `M_max â‰ˆ 71.12 kN m`, `max_newton_iterations = 33`, and
`max_bisection_level = 1`, while the accepted steps are still carried only by
`newton_backtracking` and `newton_l2`. That makes the current internal
hysteresis visible as a real benchmark artifact rather than a verbal claim.

Turning that same slice into a solver-policy benchmark exposed one more point
that was worth fixing at the architectural level. It was not enough to type the
nonlinear method family (`newton`, `trust-region`, `quasi-Newton`,
`nonlinear-GMRES`); the declared line-search family also had to be applied
honestly to non-Newton `SNES` types. Before that correction, `QN` could crash
hard and `NGMRES` was not being benchmarked under an explicitly declared
line-search contract. `NonlinearSolvePolicy.hh` now applies line-search types
through the `SNESLineSearch` object whenever the active solver exposes one, not
only for `newtonls`. In the current reduced-column benchmark this means:

- `quasi_newton` now runs with the declared `critical_point` line search;
- `nonlinear_gmres` now runs with the declared `l2`/secant line search;
- both methods now fail cleanly and reproducibly on the benchmark slice instead
  of dying behind an infrastructure ambiguity.

The resulting policy benchmark is now frozen more completely in
[`scripts/run_reduced_rc_nonlinear_solver_policy_benchmark.py`](/c:/MyLibs/fall_n/scripts/run_reduced_rc_nonlinear_solver_policy_benchmark.py)
and
[`solver_policy_benchmark_summary.json`](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_internal_solver_policy_benchmark_200mm_stage3/solver_policy_benchmark_summary.json).
Its current read is much sharper:

- `canonical-cascade`, `newton-backtracking-only`, `newton-l2-only`, and
  `newton-trust-region-dogleg-only`, `nonlinear-cg-only`, and
  `anderson-only` all complete the full `50,100,150,200 mm` protocol;
- `newton-l2-only` is the best current internal baseline on the canonical
  slice, with `process wall ~ 4.74 s` versus `5.70 s` for the cascade,
  `7.39 s` for backtracking-only, `10.63 s` for dogleg trust-region,
  `11.18 s` for `anderson-only`, and `109.85 s` for `nonlinear-cg-only`;
- `newton-l2-only` also keeps the worst Newton work much lower
  (`max_iter = 16`) than the cascade (`33`), backtracking-only (`51`), or
  dogleg trust-region (`136`), Anderson acceleration (`518`), or nonlinear
  conjugate gradient (`3257`);
- `newton-trust-region-only` still fails on the first `50 mm` reversal;
- `quasi-newton-only` and `nonlinear-gmres-only` now fail cleanly and
  reproducibly on the canonical slice;
- `nonlinear-richardson-only` fails cleanly and immediately on the canonical
  slice;
- `nonlinear-cg-only` now does close the canonical slice under its declared
  PETSc `ncglinear` contract, but only at prohibitive cost and therefore still
  does not become a credible promoted baseline.

This is the kind of comparison that matters for the thesis: not â€œwhich PETSc
method sounds modern,â€ but which declared solver policy actually gives a more
robust or cheaper path on the same physical slice. On the evidence available
today, the strongest internal baseline is no longer the full Newton cascade but
the simpler `newton-l2-only` policy. The cascade still has value as a conservative
fallback surface, but it is no longer the cheapest honest default for this
reduced structural benchmark.

Stage-3 evidence sharpens that statement further. `anderson-only` remains the
strongest additional non-Newton route that closes the full canonical `200 mm`
slice at moderate cost (`process wall ~ 11.18 s`, `max_iter = 518`).
`nonlinear-cg-only` also now closes that slice, but only at a prohibitive
computational price (`~109.85 s`, `max_iter = 3257`). At the same time,
`newton-trust-region-dogleg-only` closes the same protocol and therefore makes
the Newton-family surface richer without displacing the promoted baseline.
So the internal read is now cleaner: `newton-l2-only` is the promoted
baseline, the full cascade is a conservative fallback surface, dogleg is a
viable but slower Newton alternative, `anderson-only` is a validated but still
expensive non-Newton extension path, and `nonlinear-cg-only` is admissible but
far too costly to promote.

This also sharpens the fair comparison against OpenSees. On the structural
side the slices are not "the same code in two languages":

- `fall_n` uses a mixed-interpolation Timoshenko beam with explicit shear
  strains, direct global assembly, and an exported condensed section tangent.
- OpenSees is compared through three declared families:
  `dispBeamColumn`, `forceBeamColumn`, and `ElasticTimoshenkoBeam`.
- `ElasticTimoshenkoBeam` is now kept as the clean elastic shear-flexible
  control, so the kinematic part can be separated from the nonlinear
  beam-column machinery.

The timing story must be stated just as carefully. On the current elasticized
Timoshenko parity control, `fall_n` is slightly faster in total wall time
(`0.1608 s` vs `0.1841 s`) and clearly lower in process wall time
(`0.2137 s` vs `0.4557 s`). On the current nonlinear `100 mm` external cyclic
benchmark, however, `fall_n` is still slower
(`1.3137 s` vs `0.9714 s` total wall time; `1.5158 s` vs `1.3283 s` process
wall time). So the honest current claim is not blanket runtime dominance. The
present advantages of `fall_n` are architectural: explicit formulation
control, typed extension surfaces, finer solver diagnostics, and cleaner
auditability of the section/material state. Speed is already competitive on
some equivalent slices, but not yet on all of them.

That architectural surface is now exercised on a broader structural family
matrix, frozen in `scripts/run_reduced_rc_timoshenko_matrix_experiment.py`
and
[`timoshenko_matrix_experiment_summary.json`](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_timoshenko_matrix_experiment_stage2_hifi2d/timoshenko_matrix_experiment_summary.json).
The experiment covers `TimoshenkoBeamN` with `N = 2..10`, four beam-axis
integration families (`legendre`, `lobatto`, `radau-left`, `radau-right`),
the common internal cyclic protocol `50, 100, 150, 200 mm`, and a declared
external high-fidelity OpenSees reference over the **full** protocol.

The matrix already tells a useful and fairly nuanced story:

- `26/36` `fall_n` cases complete the full internal `200 mm` protocol.
- `N = 2, 3, 4` complete for all four integration families.
- for `N >= 5`, robustness depends strongly on beam-axis integration:
  `radau-left` is the strongest family (`5/6` completed), `radau-right`
  follows closely (`4/6`), `lobatto` remains viable (`4/6`), and
  `legendre` becomes fragile (`1/6`).
- among completed high-order cases, `radau-right` currently offers the best
  robustness/work balance: lower average worst Newton work than
  `radau-left` while keeping most of its completion rate.
- `legendre` stays the cheapest family when it works, so its weakness is not
  raw cost but loss of robustness as the interpolation order grows.

That is exactly the kind of result we needed before scaling the benchmark:
the structural element in `fall_n` is no longer judged by one favorite slice,
but by a declared family matrix where robustness, timing, and convergence all
move together. The corresponding figures are already frozen in:

- [timing and convergence](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_timoshenko_matrix_timing_convergence.png)
- [physical coherence](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_timoshenko_matrix_physical_coherence.png)
- [hysteresis overlays](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_timoshenko_matrix_hysteresis_overlays.png)

The external high-fidelity OpenSees comparator also becomes much more honest
under this framing. The old 3D hi-fi route stayed trapped in local
`forceBeamColumn` compatibility failures before becoming a useful
large-amplitude reference, so the campaign now freezes a **single-direction 2D
hi-fi structural comparator** that preserves the same RC fiber section, axial
preload, and cyclic drift history while removing the extra 3D frame noise that
was not helping this benchmark. The declared hi-fi reference now lives in
[`reference_manifest.json`](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_timoshenko_matrix_experiment_stage2_hifi2d/opensees_hifi_reference/reference_manifest.json):

- `dispBeamColumn`
- `2d`
- `20` structural elements
- `legendre`
- `5` section integration points per element
- `pure-secant-disp` as the declared OpenSees profile family
- a `Steel02` reversal surface reset to much more standard values
  (`R0 = 20`, `cR1 = 0.925`, `cR2 = 0.15`)
- the same reduced RC section physics
- a stable nonlinear comparison window over the **full `200 mm` protocol**

That last point matters. The external hi-fi comparator is no longer just a
short-window probe. It is now a reproducible nonlinear structural reference
through the full `200 mm` protocol, so the current hi-fi hysteresis overlays
are finally comparable on the whole reduced cyclic case rather than only on a
short prefix.

The reasoning that led here is worth preserving because it was not cosmetic.
Two things had to change together:

- the hi-fi runner now exposes the structural model dimension explicitly, so
  the same fiber-section policy can be exercised in `2d` and `3d` instead of
  hiding the benchmark behind one hard-wired OpenSees frame choice;
- the OpenSees convergence surface now exposes **pure profile families**
  (`pure-krylov-disp`, `pure-broyden-disp`, `pure-secant-disp`, etc.), which
  matters because the mixed-profile cascade does not have a trustworthy domain
  restore on this Windows/OpenSeesPy path once a candidate profile fails
  mid-increment.

That second point is methodological, not stylistic. The mixed OpenSees cascade
is still useful diagnostically, but the canonical hi-fi comparator is now
intentionally frozen on a single-profile family to avoid hidden cross-profile
state contamination inside one displacement increment.

The corresponding OpenSees hi-fi solver-profile benchmark is now frozen in
[`solver_profile_benchmark_summary.json`](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_opensees_hifi_solver_profile_benchmark_200mm/solver_profile_benchmark_summary.json)
and the figures:

- [OpenSees hi-fi profile timing](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_opensees_hifi_solver_profile_timing.png)
- [OpenSees hi-fi profile robustness](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_opensees_hifi_solver_profile_status.png)

The current read is already useful:

- `pure-krylov-disp`, `pure-broyden-disp`, `pure-secant-disp`, and
  `robust-large-amplitude` all complete the full hi-fi `200 mm` protocol;
- `pure-krylov-disp` is presently the fastest completed OpenSees family on the
  frozen `2d dispBeamColumn` hi-fi case (`~3.93 s`);
- `pure-secant-disp` remains the declared canonical family because it stays
  single-profile and completed robustly under the same full protocol;
- plain Newton and line-search-only fail early, and
  `pure-modified-initial-energy` is non-promotable on this slice because it is
  both late-failing and prohibitively expensive.

Under the new full-window comparator, the structural matrix read also becomes
much more meaningful than it was under the old `50 mm` external window. On the
current frozen bundle, the best completed `fall_n` cases against the hi-fi
comparator cluster around the higher-order `lobatto`/`radau-left` families,
with representative hysteresis RMS error of order `1.5e-1` to `1.7e-1` and
total-work error of order `1.0e-1` to `1.4e-1`. That is still not closure, but
it is a physically interpretable benchmark rather than a short-window
placeholder.

In parallel, the solver-comparison infrastructure has now been exercised on a
representative nonlinear case matrix and on a supplemental Lobatto-only
non-Newton matrix, frozen in
[`scripts/run_reduced_rc_solver_policy_case_matrix.py`](/c:/MyLibs/fall_n/scripts/run_reduced_rc_solver_policy_case_matrix.py),
[`solver_policy_case_matrix_summary.json`](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_solver_policy_case_matrix_stage1/solver_policy_case_matrix_summary.json),
and
[`solver_policy_case_matrix_summary.json`](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_solver_policy_case_matrix_stage3_extended/solver_policy_case_matrix_summary.json).
The pattern is now hard to ignore:

- no non-Newton family is yet a promotion candidate on this structural family;
- `newton-l2-only` is the strongest current baseline across the harder
  completed slices;
- `newton-trust-region-dogleg-only` is now a real Newton-family alternative:
  it closes `N=2`, `N=4`, and `N=10` on Lobatto, but remains clearly slower
  than `newton-l2-only`;
- `anderson-only` is the strongest new non-Newton route:
  it completes `N=2` and `N=4` on Lobatto, but loses robustness at `N=10`
  and remains much more expensive;
- `nonlinear-cg-only` can complete the mid-order `N=4` Lobatto slice, but at
  prohibitive cost (`~94.7 s` vs `~6.57 s` for `newton-l2-only`) and still
  fails on `N=2` and `N=10`;
- `nonlinear-richardson-only` remains non-promotable: it fails on all three
  representative Lobatto slices;
- `quasi-newton` and `nonlinear-gmres` remain useful extension paths, but not
  validated baselines.

So the current fair architectural claim is stronger than before and still
verifiable: `fall_n` is already ahead of OpenSees in formulation explicitness,
typed solver-policy declaration, and auditability of the numerical chain. It
is sometimes faster, sometimes not, depending on the slice. The speed story is
therefore still conditional, but the flexibility and traceability advantages
are now backed by a broader structural-family experiment instead of one-off
probes. The same typed `NonlinearSolvePolicy` surface is also the declared seam
for future nonlinear families beyond Newton and the currently benchmarked
PETSc-SNES routes; in particular, it is where TAO/Levenberg-Marquardt-style
damped least-squares paths should attach if they later prove scientifically
useful on dynamic or strongly degraded slices.

- `arc-length` did not rescue the current structural reference slice;
- a more segmented displacement path with a larger continuation substep factor
  did not rescue it either.
- the new family audit confirms the same result at `15 mm` for both
  `dispBeamColumn` and `forceBeamColumn` comparators.

A deeper structural-fiber audit then exposed one more comparison bug worth
fixing before drawing conclusions. The structural OpenSees bridge was querying
fiber states with the section `z` coordinate left in native OpenSees sign,
while the benchmark comparison contract already assumes the declared `fall_n`
section convention. Once that `z` projection was corrected in
`scripts/opensees_reduced_rc_column_reference.py`, the apparent structural
fiber gap dropped sharply on the short cyclic slice:

- `max_rel_fiber_stress_error` fell from about `7.96` to about `1.15`;
- `rms_rel_fiber_stress_error` fell from about `1.72` to about `3.70e-1`;
- the worst tangent mismatch remained localized mainly at the base station,
  not spread uniformly across the whole beam.

That matters because it changes the structural read from "the whole fiber
cloud is inconsistent" to a much narrower statement: after the `z` convention
fix, the dominant structural mismatch is localized in specific unconfined
cover fibers during cyclic reversal, especially around the base and the
intermediate station.

I then re-probed the `15 mm` structural frontier on the `fall_n` side with
much stronger guarded displacement continuation. The frontier moved
substantially:

- the default audited path was stopping near `11.25 mm`;
- a stronger guarded path reached about `13.59 mm`;
- a very aggressive path with many more preload/substeps/bisections reached
  about `13.9453125 mm`, but still did not complete `15 mm`.

The converged frontier hotspot on those stronger runs remains very specific:
the last committed state is dominated by base-station `cover_top`
unconfined-concrete fibers that are fully open in tension
(`stress ~ 0`, `tangent ~ 0.03 MPa`) while adjacent reinforcement remains
active. So the current structural amplitude frontier is not just a lazy
continuation default anymore; it is now a mixed numerical/constitutive
frontier concentrated in the tensile cover response near the base.

A new transition audit now freezes that statement more tightly through
`scripts/run_reduced_rc_structural_frontier_transition_audit.py`. It compares
the strongest successful structural slice (`12.5 mm`) against both the last
converged state and the exact failed recursive trial of the strongest `15 mm`
continuation bundle, on the exact same tracked fiber (`section_gp = 0`,
`cover_top`, unconfined concrete) and on the same base-section tangent block.
The result is even more precise than the earlier â€œlocalized hotspotâ€ reading:

- the tracked extremal cover fiber is already fully open at the successful
  `12.5 mm` peak (`stress = 0`, `tangent = 0.03 MPa`);
- that same fiber remains in the same open state at the last converged
  frontier point near `13.9453125 mm`;
- the exact failed recursive target sits only at about `13.9672852 mm`, i.e.
  just `~0.16%` beyond the last converged drift;
- at that exact failed trial, the same tracked fiber is still fully open
  (`stress = 0`, `tangent = 0.03 MPa`) and its strain increases by only
  `~0.15%` relative to the last converged point;
- between those two states, the fiber strain grows by about `13.6%`, the
  base-section curvature magnitude by about `11.9%`, and the base moment
  magnitude by about `9.6%`;
- over the same interval, the condensed base flexural tangent changes only
  mildly (`-1.6%`), `raw_k00` drops by about `7.1%`, `raw_k0y` by about
  `2.3%`, and `raw_kyy` stays nearly unchanged on the committed branch;
- between the last converged point and the exact failed trial, the base
  section barely changes: curvature by `~0.15%`, moment by `~0.14%`, and the
  condensed tangent by only `~-0.004%`;
- zone/material aggregates at the base also stay smooth between those two
  states; the compressive cover/core and the steel continue carrying the
  section without an abrupt redistribution;
- the new solver trace says the committed branch stays calm on both sides of
  the frontier: the successful `12.5 mm` run never needs bisection, and the
  strongest `15 mm` bundle still reaches its last converged point as a
  committed `2`-iteration step with `0` committed bisection depth;
- the actual break happens on one attempted runtime step near `13.9672852 mm`,
  where the solver accepts one substep, exhausts bisection to depth `4` on
  the remainder, and exits with `SNES reason = -6` and residual norm
  `~1.67e-3`.

So the current `15 mm` blockage is better read as a fine continuation
frontier acting on an already degraded section/fiber state, not as a new
abrupt local collapse that appears only beyond `12.5 mm`.

Artifacts:

- [frontier transition summary](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_structural_frontier_transition_failed_attempt_audit/frontier_transition_summary.json)
- [tracked fiber transition](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_structural_frontier_tracked_fiber_transition.png)
- [base section block transition](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_structural_frontier_section_block_transition.png)
- [base zone/material transition](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_structural_frontier_zone_transition.png)
- [solver trace transition](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_structural_frontier_solver_trace_transition.png)

To make sure that reading is not fighting the local physics, I also audited the
same extremal base fiber through the full successful `12.5 mm` structural
cycle in
`scripts/run_reduced_rc_structural_fiber_closure_audit.py`. That audit is
useful because it separates â€œwrong local event orderingâ€ from â€œsame ordering,
different timingâ€. The result is encouraging:

- both `fall_n` and OpenSees show the same broad sequence on that fiber:
  initial compression, tensile opening on the positive branch, closure near
  the return through zero drift, and re-entry into compression on the reverse
  branch;
- the main mismatch is not the existence or absence of closure/recompression,
  but its timing: OpenSees releases that fiber to zero tensile stress one
  structural step earlier on the positive branch;
- closure and compressive re-engagement then occur at the same structural step
  (`step 6`, zero drift crossing);
- the remaining difference is most visible at the final return to zero drift:
  `fall_n` has already released back to `stress = 0`, `tangent = 0.03 MPa`,
  while OpenSees still keeps a small compressive traction
  (`~-3.16e-2 MPa`) and a large tangent (`~1.46e4 MPa`).

So the local chronology now looks physically consistent on both sides. The
remaining gap is better interpreted as a constitutive timing/residual-state
difference around closure and final re-contact of the cover concrete, not as a
grossly wrong local mechanism.

Artifacts:

- [fiber closure summary](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_structural_fiber_closure_audit/fiber_closure_summary.json)
- [fiber closure trace](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_structural_fiber_closure_audit/fiber_closure_trace.csv)
- [fiber closure events](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_structural_fiber_closure_events.png)
- [fiber closure phase path](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_structural_fiber_closure_phase_path.png)

### Stage-1 Closure

The first reduced-column validation stage is now closed as an
**algorithmic/computational baseline**, not as final physical validation. The
canonical closure bundle is
[reboot_stage1_closure_audit](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_stage1_closure_audit),
generated by
[`scripts/run_reduced_rc_stage1_closure_audit.py`](/c:/MyLibs/fall_n/scripts/run_reduced_rc_stage1_closure_audit.py).

Its role is to freeze one honest statement of what is already solid and what is
still open:

- the declared internal `fall_n` structural slice now completes through the
  full internal `200 mm` protocol under the shared PETSc SNES profile cascade;
- the promoted internal solver baseline is now explicit:
  `newton-l2-only` is the cheapest completed route on the canonical slice,
  while `anderson-only` is already a validated but expensive non-Newton
  extension path;
- the elasticized external Timoshenko parity control closes tightly enough to
  remove gross geometry/observable mismatch from the active suspicion list
  (`max_rel_base_shear_error ~ 1.32e-4`, section tangent error `~ 0`);
- the external section comparator still closes only through
  `kappa_y = 0.015 1/m` and still fails first at `0.020 1/m`;
- no materially plausible single-profile external concrete override rescues
  that first section reversal;
- the external structural comparator is now declared honestly as
  non-monotone: it fails first at `75 mm`, reopens at `100 mm`, and fails
  again at `125/150 mm`;
- timing is now part of the closure, not an afterthought:
  `fall_n` is slightly faster on the elasticized Timoshenko parity control,
  `newton-l2-only` beats the internal cascade and Anderson route on the full
  `200 mm` protocol, while OpenSees remains faster on the nonlinear `100 mm`
  external benchmark.

So this stage is closed in a precise sense: the reduced-column slice is no
longer blocked by an internal algorithmic ambiguity, the internal nonlinear
solve surface has a promoted baseline plus a validated non-Newton extension,
and the remaining gap is localized. What is **not** closed yet is the next
stage: physically defendable large-amplitude equivalence to external RC-column
evidence.

Artifacts:

- [stage-1 closure summary](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_stage1_closure_audit/stage1_closure_summary.json)
- [stage-1 closure checkpoints](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_stage1_closure_audit/stage1_closure_checkpoints.csv)
- [stage-1 closure frontier](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_stage1_closure_frontier.png)
- [stage-1 closure timing](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_stage1_closure_timing.png)

The closure is also rechecked on the library side, not only in postprocessing:
the reduced-column benchmark-trace and evidence-closure catalog tests pass, and
the dynamic steppable regression confirms that the shared nonlinear-solve
profile contract reaches the `TS/SNES` path as intended.

So the next amplitude step should stay disciplined:

- first reduce the constitutive/section gap enough to push the external
  section bridge beyond `0.015 1/m`,
- then extend the OpenSees structural bridge beyond `50 mm` so the external
  comparator catches up with the now-stable `fall_n` slice,
- and only after that ask whether a new continuation benchmark with
  `arc-length` deserves to become part of the declared validation surface.

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
- do not â€œcleanâ€ the thesis from the `fall_n` root unless you intend to clean
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
- `header_files.hh` still causes avoidable build coupling, even though the shared PCH no longer depends on it and the beam-validation slice already has a narrower replacement umbrella.
- The dedicated `fall_n_case5_frontier_probe_test` target now builds again on the audited Windows/MSYS2 path after the probe-specific link cleanup, but it remains a runtime-limited artifact in the local harness and therefore is not yet a stable CI-speed oracle for the full Case 5 frontier.
- The repository mixes naming styles such as `non_lineal` / `nonlinear` and `Homogenisation` / `Homogenized`.
- The top-level documentation still contains historical drift and LaTeX warning debt.
- Some heavy validation drivers remain expensive to compile and run; on the latest audited cyclic-validation pass, the isolated `TableCyclicValidationFE2.cpp` object rebuilt in about `177.2 s`, while the extracted `TableCyclicValidationFE2Recorders.cpp` slice rebuilt in about `38.6 s`, so the heavy runtime TU frontier is still open even though the IO/recorder slice is now materially cheaper to edit.

## Quick Wins

The fastest improvements with a good effort/impact ratio are:

1. Keep shrinking the direct use of `header_files.hh` and move toward module-local umbrellas and explicit includes, following the new `src/validation/BeamValidationSupport.hh` migration pattern.
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
- The same audited run shows `72â€“82` active cracked material points, up to `3` cracks at a point, one-step no-flow stabilization, and no no-flow destabilization. That is strong evidence that the remaining frontier is the late, heavily cracked tail of the first micro ramp, not the top-level `flow / no-flow` split itself.
- In the current default `fe2_crack50` one-step audit, all four submodels now reach `100%` of the first `+2.5 mm` target with `4â€“7` accepted ramp substeps, `0â€“3` bisections, `80â€“82` active cracked points, and zero tail-rescue activations. The step still aborts, but now with `MacroSolveFailed` and `failed_submodels = 0`: the scientific frontier has moved from the local micro ramp to the first macro FE2 re-solve with injected cracked-section operators.
- A synthetic API regression now proves that the new macro backtracking path can recover a macro solve that would otherwise fail under the same affine section-law predictor. The full structural `Case 5` frontier remains macro-dominated, but the cheap probe used during stabilization is still runtime-limited; the new backtracking path should therefore be read as a validated algorithmic capability, not yet as a closed end-to-end physical-validation result.
- A second execution profile, `fe2_frontier_audit`, now exists to make the first cracked Case 5 point reproducible and machine-readable at lower runtime cost. In the current audited run `build/fall_n_table_cyclic_validation.exe --case 5 --protocol extended50 --fe2-profile frontier --max-steps 1 --global-output-interval 0 --submodel-output-interval 0`, the uncoupled macro step converges, but the FE2 step aborts with `MicroSolveFailed`, `failed_submodels = 2`, and failed coupling sites `eid=0/gp=0/xi=-0.57735` and `eid=2/gp=0/xi=-0.57735`.
- Failed FE2 steps now leave explicit `accepted=0` rows in `global_history.csv` and `crack_evolution.csv`. Structural quantities that are no longer physically defined after rollback, such as accepted base shear and accepted structural peak damage, remain `NaN` instead of being silently back-filled from the restored state.
- The current `fe2_frontier_audit` row is intentionally subtle: it records `total_cracked_gps = 0`, `total_cracks = 0`, and `max_opening = 0`, but also `total_active_crack_history_points = 306` and `max_num_cracks_at_point = 3`. That means the material points already carry crack history, while no open-crack observable survives the current recorder criterion at that failed FE2 point. The validation methodology therefore needs both observables; either one alone would give a misleading diagnosis.
- A second synthetic API regression now proves that the FE2 loop can also recover a macro failure by cutting back the macro increment while still reaching the same target control point. This is a continuation aid for the validation campaign, not yet a claim that the full structural `Case 5` path is closed.
- The next concrete validation task is now split in two levels instead of one. Under `fe2_frontier_audit`, the cheap reproducible frontier is micro-limited and localized at two coupling sites. Under the production-like `fe2_crack50` budget, the frontier is already macro-dominated. That split is scientifically useful: it tells us that further validation needs both a local late-tail continuation study and a macro post-peak continuation study, rather than a single undifferentiated "Case 5 still fails" verdict.
- The optional consistent-tangent override remains an audit path, not a promotion path. The earlier negative result is preserved as historical evidence, but after the FE2 setup-lifetime correction it should be treated as a benchmark that needs re-audit before any final scientific claim is made.

## Continuum Bring-Up

The reduced-column continuum phase is no longer blocked by the baseline driver
itself.

The new canonical artifact is:

- `scripts/run_reduced_rc_continuum_bringup_audit.py`
- [continuum bring-up summary](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_continuum_bringup_audit/continuum_bringup_summary.json)

This audit exists because the first apparent continuum failure was not yet a
useful scientific frontier. It mixed together three different questions:

1. whether the small-strain continuum with imposed top-face displacement was
   stable on its own;
2. whether the embedded-bar boundary treatment introduced a null mode or a
   spurious kinematic gap;
3. whether the so-called monotonic driver was actually monotonic, instead of a
   disguised cyclic path.

The current baseline resolves those questions explicitly:

- `ReducedRCColumnContinuumRunSpec` now exposes a real reinforcement policy:
  `continuum_only` versus `embedded_longitudinal_bars`.
- The embedded-bar boundary treatment is explicit instead of hidden:
  `dirichlet_rebar_endcap` versus `full_penalty_coupling`.
- The continuum benchmark now treats monotonic loading as its own protocol.
- The same typed PETSc/SNES solve-policy seam is reused here, so the
  continuum slice is not running on an ad hoc nonlinear solve path.

The present audited read is:

- `continuum_only_elastic_monotonic_hex8_2x2x4` completes in about `0.132 s`
  with peak `|V_base| â‰ˆ 2.026 kN`.
- `embedded_elastic_monotonic_hex8_2x2x4` completes in about `0.167 s` with
  peak `|V_base| â‰ˆ 2.129 kN`, i.e. about `5.1%` stiffer than the plain
  continuum control.
- `continuum_only_nonlinear_monotonic_hex8_2x2x8` completes in about `6.64 s`
  with peak `|V_base| â‰ˆ 4.634 kN`.
- `embedded_nonlinear_monotonic_hex8_2x2x8` completes in about `10.49 s` with
  peak `|V_base| â‰ˆ 5.163 kN`, i.e. about `11.4%` stiffer than the plain
  nonlinear continuum control.
- `embedded_nonlinear_cyclic_hex8_2x2x8` completes the short audited cyclic
  path (`1.25, 2.50 mm`, `24` runtime steps) in about `34.60 s`.
- The top rebar-face gap stays at machine precision (`~1e-19 m`) on the
  embedded slice, which is the correct early sign that the current
  endcap-plus-penalty treatment is kinematically coherent on this bring-up
  matrix.

So the next continuum-phase work is now the real validation work that was
planned from the start: Hex20/Hex27, richer material comparisons, larger
amplitudes, and then the structural-vs-continuum reduced-column comparison on
top of a stable, auditable baseline.

## Continuum Order Matrix

The next continuum gate is no longer a generic â€œdoes the 3D column run?â€ but a
more specific question: which hex family remains physically coherent once we
restore the **same axial-preload semantics** already present on the structural
reference slice.

The canonical artifact is now:

- `scripts/run_reduced_rc_continuum_order_matrix_audit.py`
- [continuum order matrix summary](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_continuum_order_matrix_audit_axial/continuum_order_matrix_summary.json)

This iteration corrected a real modeling gap. The structural reduced-column
reference already carried a `0.02 MN` axial preload with an equilibrated preload
stage. The continuum baseline did not. That is no longer true:

- `ReducedRCColumnContinuumRunSpec` now exposes `axial_compression_force_mn`,
  `use_equilibrated_axial_preload_stage`, and `axial_preload_steps`.
- The preload is applied through a **consistent top-face traction** on the host
  continuum only, not through an ad hoc nodal-force patch.
- `Domain::create_boundary_from_plane(...)` now has a range-restricted overload
  so embedded rebar elements do not pollute the top-face traction boundary.
- The continuum control trace now records whether a given accepted point is the
  preload-equilibrated anchor or part of the lateral branch.

That correction matters because it changes the physical reading of the
hex-order comparison.

With the same axial preload (`0.02 MN`) and the same elastic short monotonic
probe (`0.5 mm` tip drift, `2x2x4` mesh), the current read is:

- `Hex8` completes in `0.44 s` with `|V_base|max ~= 2.129 kN`.
- `Hex20` completes in `2.57 s` with `|V_base|max ~= 0.493 kN`.
- `Hex27` completes in `5.27 s` with `|V_base|max ~= 0.492 kN`.
- `Hex20` and `Hex27` differ by only about `0.21%`.
- The structural elastic reference
  [reboot_structural_reference_elastic_0p5mm_radau_right](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_structural_reference_elastic_0p5mm_radau_right)
  gives `|V_base|max ~= 0.536 kN`, so `Hex20/Hex27` sit only about `8%` low,
  while `Hex8` is about `4.3x` stiffer and roughly `297%` above the structural
  reference.

That is the first strong sign that the low-order `Hex8` slice is currently
locking-dominated or otherwise too stiff to promote as the physical continuum
reference, whereas `Hex20/Hex27` already cluster in a much more believable
elastic band.

The nonlinear/preloaded read is more subtle:

- `Hex8` matched-topology monotonic (`2.5 mm`, `2x2x4`) completes in `18.56 s`
  with `|V_base|max ~= 12.667 kN`.
- `Hex20` matched-topology monotonic completes in `247.25 s` with
  `|V_base|max ~= 2.647 kN`.
- `Hex27` matched-topology monotonic does not close within the current
  `480 s` budget.

This must be interpreted carefully. The structural nonlinear reference still
uses a fiber-section beam slice, while the continuum slice uses a 3D concrete
law plus embedded Menegotto bars. So this stage is not yet a formal
nonlinear-convergence claim across formulations; it is an **algorithmic and
physical viability frontier**. Even under that more modest reading, `Hex8`
still looks suspiciously stiff.

The short preloaded cyclic cost-controlled matrix is even sharper:

- `Hex8` (`2x2x4`) completes in `42.84 s`.
- `Hex20` (`2x2x4`) exceeds the current `240 s` audit budget.
- `Hex27` (`1x1x4`) also exceeds the current `240 s` audit budget.

The rebar-face gap remains at machine precision (`~1e-19 m`) on every completed
case, which means the active frontier is no longer a gross host-bar kinematic
mismatch. The frontier is now the combination of:

- physical coherence across hex families,
- runtime cost under axial preload,
- and the choice of which continuum family deserves promotion before the first
  strong structural-vs-continuum comparison.

The honest promotion path at this point is:

- keep `Hex8` as a low-order control, not as the physical continuum reference;
- promote `Hex20` as the first serious continuum reference for short monotonic
  comparisons;
- keep `Hex27` as a higher-order runtime frontier until the nonlinear cost is
  reduced or better justified.

## Continuum Ko-Bathe Crack Audit

The next step was to stop treating cracking as an implicit side effect and make
it a first-class observable of the reduced continuum pilot.

The canonical artifact is now:

- `scripts/run_reduced_rc_continuum_crack_audit.py`
- [continuum crack audit summary](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_continuum_ko_bathe_crack_audit/continuum_crack_audit_summary.json)

This pass also exposed a real formulation decision rather than a cosmetic one.
The new generalized `TrussElement<Dim, NNodes>` and the reinforced prismatic
builder can now mount either:

- `2-node` embedded bars, or
- `3-node` embedded bars on quadratic host meshes.

That extension mattered immediately. Under the same monotonic precompressed
column slice (`20 mm`, `0.02 MN`, `2x2x2`):

- `Hex20 + 3-node` embedded bars was not robust enough to serve as the default
  baseline.
- `Hex20 + 2-node` embedded bars completed cleanly and produced a cracking
  response very close to `Hex27 + 3-node`.
- `Hex27 + 3-node` completed cleanly and remains the higher-fidelity host/bar
  pairing.

That is why the validated default policy is now:

- `Hex20 automatic -> two_node_linear rebar`
- `Hex27 automatic -> three_node_quadratic rebar`

This is not a patch. It is a promoted baseline based on an explicit audit of
host interpolation, embedded-bar interpolation, and nonlinear solve closure.
The explicit `three-node` override remains available for continued research on
serendipity-host coupling, but it is no longer the canonical path for `Hex20`.

The first frozen Ko-Bathe cracking comparison is already quite encouraging:

- `Hex20 auto` completes in `43.97 s` with `|V_base|max ~= 20.999 kN`.
- `Hex27 auto` completes in `164.83 s` with `|V_base|max ~= 21.345 kN`.
- The peak base-shear ratio `Hex20/Hex27` is `~0.984`.
- Both first crack at the same drift, `~6.67 mm`.
- `Hex20` reaches `48` cracked Gauss points with `max crack opening ~= 7.21e-4`.
- `Hex27` reaches `42` cracked Gauss points with `max crack opening ~= 7.34e-4`.
- `Hex20` closes the slice in only `~26.7%` of the `Hex27` solve time.

That is exactly the kind of result we wanted before moving on to richer
continuum comparisons: `Hex20` and `Hex27` now look physically coherent in the
same cracked regime, while still exposing a meaningful cost hierarchy.

The audit also finally makes cracking visible as an artifact, not just as a
derived metric:

- each case writes `crack_state.csv`
- each case writes `vtk/continuum_mesh.pvd`
- each case writes `vtk/continuum_gauss.pvd`
- each case writes `vtk/continuum_cracks.pvd`

The `gauss` output carries the Ko-Bathe crack fields (`qp_num_cracks`,
`qp_crack_normal_*`, `qp_crack_strain_*`, `qp_crack_closed_*`), while the
`cracks` output writes explicit quad crack planes at the active Gauss points.
That means the reduced continuum reboot can now show the onset and spread of
cracking directly in ParaView, not just infer it from force-displacement data.

The next correction was more subtle, but just as important for the
structural-vs-continuum comparison: the first attempt at a local steel
hysteresis bridge was not actually comparing the same physical point. The
continuum slice had already been corrected to use the same eight-bar layout as
the structural/OpenSees reference, but the embedded `TrussElement` Gauss point
of the promoted `Hex20` pairing does not sit on the clamped boundary. With a
`2x2x2` host mesh and the validated automatic `Hex20 -> two_node_linear`
policy, the first steel Gauss point on the tracked top-right bar sits at
`z ~= 0.338 m`, not at `z = 0`. Comparing that directly against the structural
base section was therefore still physically misaligned.

That matched-height bridge still hid one more mistake. Even after fixing the
axial position, the local steel curves were still being compared on the wrong
structural face: the nominal structural steel fiber selected by `max(z,y)`
did not live on the same active bending side as the embedded bar. The bridge
now audits that explicitly too.

That is now fixed in
`scripts/run_reduced_rc_structural_continuum_steel_hysteresis_audit.py`.
The audit now does four things explicitly:

- picks the same longitudinal bar in both models (`y = z = 0.095 m`);
- tracks the embedded-truss Gauss point closest to the base in the continuum;
- interpolates the structural steel-fiber history axially between beam section
  stations to the same physical height before comparing the loops;
- enumerates the structurally compatible steel-fiber candidates and selects
  the orientation-parity candidate that best matches the embedded bar on the
  initial elastic branch.

This is a methodological correction, not a cosmetic post-processing tweak. The
old `2x2x2` bridge is now superseded by the promoted `4x4x2` local continuum
baseline:

- structural reference:
  `TimoshenkoBeamN<10> + Lobatto + newton-l2-only + clamped top rotation`;
- continuum reference:
  `Hex20 4x4x2 + cover_core_split + cover_aligned + structural_matched_eight_bar + production-stabilized + fracture-secant + fixed-end lb`;
- same axial preload:
  `0.02 MN`.

The promoted bridge is now frozen in three canonical bundles:

- monotonic `20 mm`:
  [`reboot_structural_continuum_promoted_monotonic_audit`](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_structural_continuum_promoted_monotonic_audit/structural_continuum_steel_hysteresis_summary.json)
- cyclic `5-10-15-20 mm`:
  [`reboot_structural_continuum_promoted_cyclic_20mm_audit`](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_structural_continuum_promoted_cyclic_20mm_audit/structural_continuum_steel_hysteresis_summary.json)
- cyclic `5-10-15-20-25 mm`:
  [`reboot_structural_continuum_promoted_cyclic_25mm_audit`](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_structural_continuum_promoted_cyclic_25mm_audit/structural_continuum_steel_hysteresis_summary.json)
- cyclic `5-10-15-20-25-30 mm`:
  [`reboot_structural_continuum_promoted_cyclic_30mm_audit`](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_structural_continuum_promoted_cyclic_30mm_audit/structural_continuum_steel_hysteresis_summary.json)

The read is now materially cleaner:

- the monotonic `20 mm` bridge closes with
  `max_rel_base_shear_error ~ 3.81e-1`, `rms ~ 2.74e-1`,
  and the local steel comparison drops to
  `max_rel_stress_vs_drift_error ~ 1.03e-1`, `rms ~ 7.07e-2`;
- extending that same promoted bridge to `20 mm` still converges on both sides,
  with `base shear rms ~ 4.12e-1` and `steel rms ~ 3.67e-1`;
- pushing one step further to `25 mm` keeps the bridge itself stable,
  with `base shear rms ~ 4.02e-1` and `steel rms ~ 3.71e-1`;
- pushing again to `30 mm` still keeps the promoted bridge operational,
  with `base shear rms ~ 3.90e-1` and `steel rms ~ 3.79e-1`;
- the orientation audit still promotes structural fiber `85`
  (`y = +0.095 m`, `z = -0.095 m`) instead of the nominal fiber `87`.

The new part of the story is even more useful than the remaining mismatch.
The continuum benchmark now exports the host-projected axial strain on the same
embedded bar path, so we can finally test whether the steel discrepancy is
coming from the host-to-truss transfer itself. On the promoted `Hex20 4x4x2`
cyclic `15 mm` slice, it is not:

- `max_abs_host_rebar_axial_strain_gap ~ 1.64e-10`
- `rms_abs_host_rebar_axial_strain_gap ~ 6.65e-11`
- `max_abs_projected_axial_gap_m ~ 1.38e-11`
- `max_abs_rebar_kinematic_consistency_gap = 0`

That means the embedded bar is following the projected host kinematics almost
to machine precision on the selected path. The open physical gap between the
structural steel fiber and the continuum embedded bar is therefore no longer
credibly explained by a loose penalty transfer, a bar-selection mistake, or a
height-parity mistake. It now points back to the remaining continuum physics:
host constitutive response, distributed cracking, and the way the continuum
shares curvature and tension stiffening with the embedded steel.

The corresponding figures are now also part of the canonical record:

- `doc/figures/validation_reboot/reduced_rc_structural_continuum_embedded_transfer_strain_20mm.png`
- `doc/figures/validation_reboot/reduced_rc_structural_continuum_embedded_transfer_gap_20mm.png`

Once that kinematic excuse was eliminated, the next question became more local:
what does the concrete host look like around the same embedded bar path? The
new locality audit in
`data/output/cyclic_validation/reboot_continuum_host_bar_locality_promoted_30mm_audit/continuum_host_bar_locality_summary.json`
shows that the nearest host Gauss point is **not** telling the same local story
as the embedded bar:

- `max_rel_bar_vs_host_axial_strain_error ~ 1.17e-1`
- `rms_rel_bar_vs_host_axial_strain_error ~ 8.14e-2`
- `max_rel_bar_vs_host_axial_stress_error ~ 8.98e1`
- `rms_rel_bar_vs_host_axial_stress_error ~ 2.55e1`
- `peak_host_crack_count = 3`
- `max_abs_host_crack_opening ~ 9.19e-4`
- the nearest host point first cracks around `drift ~ -10 mm`

So the open gap is now localized more honestly: the embedded steel is being
driven by a host neighborhood that is already heavily cracked and much softer
than the structural beam abstraction would suggest. That is a much better
scientific target than “maybe the tie is loose”.

The new locality figure is now part of the canonical record:

- `doc/figures/validation_reboot/reduced_rc_continuum_host_bar_locality_hex20_30mm.png`

With that locality read frozen, the next useful question was whether the three
cover/core-aware continuum families are actually competing on the same physics,
or only on the same geometry. The new family audit in
`data/output/cyclic_validation/reboot_structural_continuum_family_comparison_30mm/structural_continuum_family_comparison_summary.json`
compares all three against the same clamped structural reference over the
`5-10-15-20-25-30 mm` cyclic window:

- `embedded interior`: active RMS base-shear error `~3.16e-1`, wall time `~2998 s`
- `boundary bars`: active RMS base-shear error `~2.43e-1`, wall time `~2628 s`

This is a useful result, but it should be read carefully. The `boundary` branch
is now both cheaper and globally closer to the clamped beam hysteresis than the
interior branch. At the same time, it is still a different physical family:
its steel paths sit on the host boundary rather than on the structural matched
interior layout. So it remains an explicit comparison branch, not the promoted
local baseline for future microscale work. The promoted baseline is still the
interior embedded branch, because it preserves the intended steel geometry
while materially improving the global bridge over the `plain` host.

That conclusion is now also backed by a dedicated local branch audit in
`data/output/cyclic_validation/reboot_structural_continuum_branch_locality_comparison_30mm/branch_locality_comparison_summary.json`.
The boundary branch does improve the same bridge metrics
(`global active RMS ~ 2.43e-1`, `steel active RMS ~ 4.63e-1`) and it is
cheaper than the promoted interior branch
(`~2628 s` vs `~2998 s`), but it gets there by moving the selected bar
`30 mm` outward in each section axis relative to the structural target
(`|y|, |z| : 95 mm -> 125 mm`). That outward shift also changes the local
host story around the bar:

- peak nearest-host crack opening grows from `~0.768 mm` to `~0.874 mm`
- peak steel stress rises from `~133.9 MPa` to `~169.9 MPa`
- the selected bar sits in a more strongly cracked neighborhood even though the
  host-bar kinematic tie remains clean

So the current evidence is more precise than “boundary bars are not the same
geometry.” They are globally closer because they are also mechanically more
aggressive at the steel path. That makes them a valuable physical control
branch, but still not the promoted local baseline for future microscale work.

That `30 mm` read has now been pushed one stage further, and an important
methodological inconsistency had to be removed before trusting it. The
structural-vs-continuum bridge script was still running the promoted `Hex20
4x4x2` continuum with `longitudinal_bias_power = 3`, while the canonical
`plain / embedded interior / boundary` family audit at `4x4x2` was using the
uniform longitudinal layout. With only `nz = 2`, that changes the axial
position of the first bar Gauss point quite a lot, so the old `40 mm`
comparison was mixing two different longitudinal discretizations without
saying so. The promoted bridge now uses the same uniform `4x4x2` baseline as
the continuum family audit, and the current canonical bridge artifacts are:

- cyclic `5-10-20-40 mm`:
  [`reboot_structural_continuum_promoted_cyclic_40mm_audit`](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_structural_continuum_promoted_cyclic_40mm_audit/structural_continuum_steel_hysteresis_summary.json)
- cyclic `5-10-20-30-40-50 mm`:
  [`reboot_structural_continuum_promoted_cyclic_50mm_audit`](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_structural_continuum_promoted_cyclic_50mm_audit/structural_continuum_steel_hysteresis_summary.json)

The new read is strong enough to promote as the current front:

- the promoted interior bridge now closes through `40 mm`, with
  `base shear rms ~ 4.11e-1`, `steel stress rms ~ 4.22e-1`,
  and `max_abs_host_bar_axial_strain_gap ~ 8.84e-11`;
- it also closes through `50 mm`, with
  `base shear rms ~ 4.80e-1`, `steel stress rms ~ 4.36e-1`,
  and `max_abs_host_bar_axial_strain_gap ~ 1.13e-10`;
- the nearest-host locality around that same promoted bar remains heavily
  cracked and much softer than the embedded steel path:
  at `40 mm`, `max_abs_host_crack_opening ~ 1.03e-3 m`,
  and at `50 mm`, `~1.27e-3 m`;
- the steel/host transfer itself still closes almost exactly, so the open gap
  continues to point back to the continuum physics, not to the embedding tie.

The family audit now also closes all three `Hex20 4x4x2` branches at both
`40 mm` and `50 mm`:

- [`reboot_continuum_cover_core_cyclic_audit_40mm_branches`](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_continuum_cover_core_cyclic_audit_40mm_branches/continuum_cover_core_cyclic_summary.json)
- [`reboot_continuum_cover_core_cyclic_audit_50mm_branches`](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_continuum_cover_core_cyclic_audit_50mm_branches/continuum_cover_core_cyclic_summary.json)

At `50 mm`, the family read is already quite informative:

- `plain`: `~39.07 kN`, `~1802 s`;
- `embedded interior`: `~48.63 kN`, `~4011 s`;
- `boundary`: `~55.04 kN`, `~3113 s`.

So the promoted interior branch is no longer struggling just to survive at
moderate amplitudes. It stays slower than the boundary control, but it still
completes and keeps the intended steel geometry. The boundary branch remains a
useful control because it is cheaper and stronger, but it still gets there by
moving the bars `30 mm` outward in each section axis and by making the local
host story more aggressive. At `50 mm`, for example:

- peak host crack opening near the selected bar grows from `~1.20 mm`
  (`interior`) to `~1.48 mm` (`boundary`);
- peak steel stress grows from `~230.6 MPa` to `~289.3 MPa`.

That is why the promotion still does not change. For future microscale work,
the current honest baseline remains the `embedded interior` branch, now with a
cleanly aligned bridge and a validated operational window through `50 mm`.

The next amplitude step strengthens that read. The promoted bridge now also
closes the cyclic window through `60 mm` in
[`reboot_structural_continuum_promoted_cyclic_60mm_audit`](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_structural_continuum_promoted_cyclic_60mm_audit/structural_continuum_steel_hysteresis_summary.json),
with:

- `base shear rms ~ 5.69e-1`
- `steel stress rms ~ 4.98e-1`
- `max_abs_host_bar_axial_strain_gap ~ 1.38e-10`

So the branch is still operational and the embedded tie still closes almost to
machine precision, but the global and local physical gap both widen. The family
audit at the same window,
[`reboot_continuum_cover_core_cyclic_audit_60mm_branches`](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_continuum_cover_core_cyclic_audit_60mm_branches/continuum_cover_core_cyclic_summary.json),
is equally informative:

- `plain`: `~44.54 kN`, `~1820 s`
- `embedded interior`: `~56.33 kN`, `~3419 s`
- `boundary`: `~64.06 kN`, `~2878 s`

At `60 mm`, the dedicated locality comparison in
[`reboot_structural_continuum_branch_locality_comparison_60mm`](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_structural_continuum_branch_locality_comparison_60mm/branch_locality_comparison_summary.json)
actually becomes more favorable to the promoted interior branch again. The
boundary branch remains cheaper, but now it is also globally worse
(`global active RMS ~ 1.13` vs `~1.03`) and locally harsher
(`peak host crack opening ~ 1.87 mm` vs `~1.43 mm`, `peak bar stress ~ 351.8 MPa`
vs `~282.6 MPa`). That is a healthier outcome for the promotion logic: once the
window is large enough, preserving the intended steel geometry also buys a
better comparison, not only a more honest one.

Pushing the same promoted bridge one more step, to the `75 mm` cyclic window in
[`reboot_structural_continuum_promoted_cyclic_75mm_audit`](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_structural_continuum_promoted_cyclic_75mm_audit/structural_continuum_steel_hysteresis_summary.json),
shows where the next real frontier lies:

- the continuum still completes;
- the embedded transfer still closes almost to machine precision
  (`max_abs_host_bar_axial_strain_gap ~ 1.78e-10`);
- but the global hysteresis mismatch jumps sharply
  (`base shear rms ~ 2.30`);
- while the local steel mismatch rises more moderately
  (`steel stress rms ~ 5.50e-1`).

So the current promoted local model is no longer failing because of the
embedded tie or because it cannot survive larger amplitudes. It now survives
through `75 mm`, and the new frontier is more specific: the host continuum
physics around the steel path is drifting away from the structural reference
faster than the steel carrier itself.

The family audit at the same window,
[`reboot_continuum_cover_core_cyclic_audit_75mm_branches`](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_continuum_cover_core_cyclic_audit_75mm_branches/continuum_cover_core_cyclic_summary.json),
confirms that all three `Hex20 4x4x2` branches are still operational:

- `plain`: `~51.02 kN`, `~2002 s`
- `embedded interior`: `~66.50 kN`, `~4051 s`
- `boundary`: `~74.87 kN`, `~3458 s`

And the updated locality comparison in
[`reboot_structural_continuum_branch_locality_comparison_75mm`](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_structural_continuum_branch_locality_comparison_75mm/branch_locality_comparison_summary.json)
keeps the same promotion logic as `60 mm`, only more clearly:

- `embedded interior`: active global RMS `~1.10`, active steel RMS `~0.80`
- `boundary`: active global RMS `~1.22`, active steel RMS `~0.96`

So once the window is large enough, the `boundary` branch is not only a
different physical family; it is also a worse comparator for the intended steel
path, even though it remains somewhat cheaper. That is a much more robust place
to stop this pass than the earlier “boundary looks better” ambiguity.

To check whether that remaining gap was truly concentrated in the immediate
host neighborhood of the promoted steel path, I opened an explicit host-probe
surface in the continuum benchmark:

- `--host-probe label:x:y:z`
- `host_probe_history.csv`
- [host-probe family summary](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_continuum_host_probe_family_audit_75mm/continuum_host_probe_family_summary.json)

That new surface is a real architectural change, not a one-off script trick:
the same benchmark can now export the nearest actual host Gauss-point history
at any declared coordinate, with or without embedded steel. On the `75 mm`
window, using the promoted interior steel coordinate
`(x,y,z) = (0.095, 0.095, 0.338119785) m`, the read is surprisingly sharp:

- `plain`: host-probe active stress RMS vs promoted interior `~6.52e-2`
- `boundary`: host-probe active stress RMS vs promoted interior `~1.26e-2`
- `plain`: host-probe active strain RMS vs promoted interior `~1.87e-2`
- `boundary`: host-probe active strain RMS vs promoted interior `~1.21e-2`
- all three branches report essentially the same nearest-host distance
  `~1.5787e-1 m`, because the host mesh is the same

So the host point right at the intended steel coordinate does **not** separate
nearly as much as the global hysteresis does. That is an important narrowing of
the physics frontier. The promoted interior branch is still the right local
baseline, but the remaining `beam vs continuum` gap is now better read as a
more distributed host-field difference than as a purely pointwise failure of
the concrete immediately surrounding the steel.

The new host-probe surface also validated cleanly against the embedded-bar
trace on the same rerun:

- `max_abs_stress_gap_mpa ~ 5.0e-8`
- `rms_abs_stress_gap_mpa ~ 1.08e-8`
- `max_abs_strain_gap ~ 4.0e-12`

That is exactly the kind of check we wanted before using this probe machinery
as part of the promoted local-model validation ladder.

The new front figures already frozen for this stage are:

- [structural-continuum hysteresis 60 mm](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_structural_continuum_hysteresis_overlay_60mm.png)
- [structural-continuum steel hysteresis 60 mm](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_structural_continuum_steel_hysteresis_overlay_60mm.png)
- [structural-continuum branch locality 60 mm](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_structural_continuum_branch_locality_overlay_60mm.png)
- [continuum cover/core hysteresis 60 mm](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_continuum_cover_core_cyclic_hysteresis_60mm.png)
- [structural-continuum hysteresis 75 mm](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_structural_continuum_hysteresis_overlay_75mm.png)
- [structural-continuum steel hysteresis 75 mm](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_structural_continuum_steel_hysteresis_overlay_75mm.png)
- [continuum host-probe stress 75 mm](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_continuum_host_probe_stress_75mm.png)
- [continuum host-probe strain 75 mm](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_continuum_host_probe_strain_75mm.png)
- [continuum host-probe crack opening 75 mm](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_continuum_host_probe_crack_opening_75mm.png)

The corresponding family-comparison figures are now part of the canonical
record:

- `doc/figures/validation_reboot/reduced_rc_structural_continuum_family_hysteresis_20mm.png`
- `doc/figures/validation_reboot/reduced_rc_structural_continuum_family_error_timing_20mm.png`
- `doc/figures/validation_reboot/reduced_rc_structural_continuum_family_steel_20mm.png`
- `doc/figures/validation_reboot/reduced_rc_structural_continuum_branch_locality_overlay_20mm.png`
- `doc/figures/validation_reboot/reduced_rc_structural_continuum_branch_locality_overview_20mm.png`

At `25 mm`, the family comparison had exposed a first operational frontier
because the promoted `embedded interior` branch no longer fit inside the old
`2400 s` budget while `boundary` still did. The new `30 mm` sweep shows that
this was a budget frontier, not a physical impossibility: all three `Hex20
4x4x2` branches now complete when the campaign makes the runtime budget
explicit. That is a much better place to be for the local-model validation,
because the remaining gap is now again physical rather than merely operational.
structural-vs-continuum gap is no longer credibly explained by coarse `Hex20`
cross-section resolution alone. Refining `2x2x2 -> 3x3x2` buys almost no
physical closure while multiplying cost by about `3.7x`, and the axial
refinement `2x2x4` remains outside the practical budget of this stage.

The nonlinear solver-policy read is also now explicit on the same `Hex20 2x2x2`
slice:

- `newton-l2-only` remains the promoted continuum baseline:
  it closes in `~194.05 s`.
- `canonical-cascade` also closes, but slower, at `~346.60 s`.
- `newton-trust-region-dogleg-only` does close the full slice, but only in
  `~1844.27 s`; it is therefore a viable Newton-family fallback, not a
  promotable baseline.
- `anderson-only` and `nonlinear-gmres-only` fail before the first accepted
  preload step. Both diverge repeatedly at vanishingly small increments and
  never export any rebar history.
- `nonlinear-cg-only` still exceeds the current `1200 s` audit budget.
- `arc-length` was tested as a branch candidate and fails immediately for a
  principled reason: the continuum reduced-column baseline does not yet expose
  a dedicated arc-length wrapper for the embedded-solid slice, so it cannot
  honestly be promoted as a valid continuation surface here.

That last point matters. The audit now distinguishes between:

- routes that are physically valid but too expensive (`dogleg`);
- routes that fail to initialize the preload (`anderson`, `ngmres`);
- and routes that simply are not exposed yet as validated continuations
  (`arc-length` on this continuum baseline).

This is also why the runner infrastructure was hardened in this pass. The
continuum audit workers now kill the full Windows process tree on timeout
instead of leaving orphan PETSc benchmark processes alive in the background.
That is not a cosmetic cleanup: without it, timing and completion status of
the solver audit would be methodologically ambiguous.

The resulting figures are now frozen in:

- [mesh refinement hysteresis](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_structural_continuum_mesh_refinement_hysteresis.png)
- [mesh refinement steel error](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_structural_continuum_mesh_refinement_steel_error.png)
- [solver-policy timing](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_structural_continuum_solver_policy_timing.png)
- [solver-policy steel error](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_structural_continuum_solver_policy_steel_error.png)

The conclusion is therefore much narrower, but much stronger: for this first
embedded-bar continuum pilot, the next scientific step is not to keep refining
`Hex20` or to escalate blindly to arc-length. The next step is to keep
`newton-l2-only` as the promoted continuum baseline, treat `dogleg` as an
expensive fallback surface, and investigate the remaining physical mismatch at
the host/bar constitutive level.

## Continuum Monotonic Foundation Audit

Before reopening larger cyclic windows on the continuum, we also stepped back
to the monotonic physics of the embedded-solid slice. The canonical artifact is
now:

- `scripts/run_reduced_rc_continuum_physics_foundation_audit.py`
- [continuum physics foundation summary](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_continuum_physics_foundation_audit_v3/continuum_physics_foundation_summary.json)

This audit deliberately asked three narrower questions.

First, does longitudinal refinement or a base-side `z` bias materially close
the elastic bridge? The answer is no. Moving from `Hex20 2x2x2` to
`Hex20 2x2x10` reduces the peak embedded gap from `~2.70e-5 m` to
`~4.20e-6 m`, but the elastic base shear only moves from `0.497 kN` to
`0.483 kN`, while the structural beam reference stays at `0.536 kN`. A
base-clustered `bias=3` mesh does not buy closure either.

Second, is the penalty tie itself driving the mismatch? Again, no. Varying
`penalty_alpha_scale_over_ec` from `1e3` to `1e5` leaves both the elastic and
the nonlinear `Hex20 2x2x2` response essentially unchanged, including the local
embedded-gap metric. That is a strong negative result: the current mismatch is
not explained by a poorly tuned penalty spring.

Third, in the nonlinear monotonic pushover, is the excess strength coming
mainly from the embedded steel? The answer is also no. On the `20 mm`
precompressed slice:

- the structural beam baseline peaks at `~7.67 kN`
- the continuum host alone already peaks at `~17.09 kN`
- the embedded-bar continuum rises further to `~20.73 kN`

So the embedded reinforcement does add stiffness and strength, but the host
continuum itself already sits far above the reduced structural baseline. That
means the next scientific target is not another blind solver sweep. It is the
continuum constitutive/sectional physics itself.

The audit also exposed an important asymmetry that had been implicit before.
The continuum benchmark now distinguishes explicitly between two Ko-Bathe 3D
concrete profiles:

- `benchmark_reference`: requested `tp = 0.02`, clamped internally to the
  article floor `tp = 0.05`, with paper-reference crack retention
  `eta_N = 1e-4`, `eta_S = 0.1`
- `production_stabilized`: implicit `tp ~= 0.0965` for `f'c = 30 MPa`, with
  the FE2-oriented stabilized crack profile `eta_N = 0.2`, `eta_S = 0.5`

That difference is real and scientifically important. But the new monotonic
audit also shows something subtler: switching from the production-stabilized
profile back to the benchmark-reference profile changes the crack pattern
substantially, yet it barely moves the global base-shear response on the
`Hex20 2x2x2` push-over. For the host-only case at `20 mm`, the peak cracked
Gauss-point count rises from `42` to `48` and the maximum crack opening drops
from about `7.35e-4` to `4.65e-4`, while the peak base shear stays at
`~17.09 kN`. For the embedded case, the cracked-point count rises from `48` to
`66` and the maximum crack opening drops from about `7.21e-4` to `4.81e-4`,
while the peak base shear still stays at `~20.73 kN`.

That is a useful negative result: the validation path was indeed using a
material profile mismatch, and that needed to be corrected. But once corrected,
the dominant source of the global monotonic gap still does not appear to be the
tension-side crack profile alone. The next constitutive question is therefore
the compressive/effective host response of the Ko-Bathe 3D continuum under the
same reduced-column loading path, not another blind penalty or mesh sweep.

The audit figures are now frozen in:

- [continuum elastic overlay](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_continuum_foundation_elastic_overlay.png)
- [continuum penalty sensitivity](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_continuum_foundation_penalty_sensitivity.png)
- [continuum nonlinear host-vs-embedded overlay](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_continuum_foundation_nonlinear_overlay.png)

## Continuum Monotonic Equivalence Audit

With that constitutive baseline cleaned up, I also re-audited a deeper but
still monotonic question: are we even comparing the continuum against the
right beam kinematics? The canonical artifact is now:

- `scripts/run_reduced_rc_continuum_monotonic_equivalence_audit.py`
- [continuum monotonic equivalence summary](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_continuum_monotonic_equivalence_audit_v2/continuum_monotonic_equivalence_summary.json)

This audit uses two structural references:

- `free rotation`: the usual reduced-column beam slice with lateral tip drift
  and free tip bending rotation
- `clamped rotation`: the same beam slice, but with the tip bending rotation
  constrained as a kinematic control against the continuum top-face
  displacement boundary

That distinction turned out to matter a lot more in the nonlinear regime than
I expected.

In the elastic monotonic slice (`0.5 mm`), the continuum still matches the
free-rotation beam slightly better than the clamped one:

- `Hex20 2x2x2 embedded`: RMS path error `~9.87e-3` vs free, `~1.20e-2` vs clamped
- `Hex20 2x2x10 embedded`: RMS path error `~8.72e-3` vs free, `~1.23e-2` vs clamped
- `Hex20 2x2x10 bias=3 embedded`: RMS path error `~8.62e-3` vs free, `~1.24e-2` vs clamped

So the elastic bridge does not justify promoting the clamped beam as the new
structural reference.

In the nonlinear monotonic pushover (`20 mm`), the picture flips completely.
The continuum is much closer to the clamped structural control than to the
free-rotation beam:

- `Hex20 2x2x2 embedded`: RMS path error `~9.97e-1` vs free, but only `~1.79e-1` vs clamped
- `Hex20 2x2x2 host-only`: RMS path error `~7.18e-1` vs free, but only `~2.52e-1` vs clamped

This is a strong physical clue. Once the host cracks and softens, comparing a
solid whose whole top face is translated against a beam whose tip bending
rotation remains free is no longer a benign modeling choice; it materially
biases the equivalence audit.

That does **not** mean the clamped beam replaces the structural reduced-column
reference. It means the clamped beam is the correct kinematic control for this
specific monotonic beam-vs-continuum audit.

The same audit also says something uncomfortable but useful about refinement:
the nonlinear `Hex20 2x2x10` cases are still outside the present runtime
budget on this machine. So longitudinal refinement beyond `2x2x2` remains an
open operational frontier for the continuum pilot, not yet a promoted physical
improvement. Right now the best-supported reading is:

- elastic `NZ=10` refinement reduces the embedding-gap metric but barely moves the global force path
- nonlinear `NZ=10` is still too expensive to use as the first promoted
  continuum baseline
- the first continuum reference should therefore remain `Hex20 2x2x2`, with
  the clamped beam slice used as the monotonic kinematic audit control

The audit figures are now frozen in:

- [continuum monotonic equivalence overlay](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_continuum_monotonic_equivalence_overlay.png)
- [continuum monotonic equivalence error](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_continuum_monotonic_equivalence_error.png)
- [continuum monotonic equivalence timing](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_continuum_monotonic_equivalence_timing.png)

That still left an important open question: was Ko-Bathe 3D itself the wrong
local concrete model, or had we simply been benchmarking it through an
over-expensive baseline? The local-model baseline audit now freezes that
answer in:

- `scripts/run_reduced_rc_continuum_local_model_baseline_audit.py`
- [continuum local-model baseline summary](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_continuum_local_model_baseline_audit/continuum_local_model_baseline_summary.json)

The read is now much cleaner:

- the old continuum benchmark was paying for a hidden combination of
  `adaptive-central-difference-with-secant-fallback`, a fixed `lb = 100 mm`,
  and the pre-fix quadratic longitudinal bias geometry
- once the baseline is promoted to `production-stabilized +
  fracture-secant + explicit crack-band semantics`, the same `Hex20 2x2x2`
  monotonic bridge keeps essentially the same global response while cutting
  solve time from about `58.22 s` to about `24.50 s`
- so the present evidence does **not** justify replacing Ko-Bathe 3D yet;
  it justifies replacing the old baseline

The same audit also gives us a more useful local slice for the next stage:

- `Hex20 2x2x2 uniform` remains the cheapest nonlinear control
- `Hex20 2x2x2 bias=3` improves the clamped-beam monotonic bridge modestly
  (`RMS ~ 0.3221 -> 0.3177`) with no change in the order of cost
- the `fixed-end-longitudinal-host-edge` crack-band mode is now the more
  honest benchmark-specific regularization choice when we intentionally bias
  the host toward the fixed base, even though on this particular global
  monotonic bridge it does not materially move the force path relative to the
  same biased mesh with mean-edge `lb`
- the `host-only` slice remains clearly worse than the embedded-bar slice,
  which supports keeping embedded longitudinal reinforcement in the promoted
  local RC model

The refined-host frontier is also now explicit instead of anecdotal:

- `Hex20 2x2x10` elasticized closes comfortably in about `9.88â€“12.17 s`
- `Hex20 2x2x6` and `Hex20 2x2x10` nonlinear, both with base bias and
  fixed-end crack-band length, still time out at the current `300 s` budget

That separation matters scientifically. It tells us the present barrier is not
"continuous geometry is too expensive" and not "embedded trusses are broken in
general"; it is a **nonlinear local-material frontier** of the refined
continuum slice.

I also added a constitutive regression to make sure this interpretation is not
hand-wavy:

- [test_ko_bathe_concrete_3d.cpp](/c:/MyLibs/fall_n/tests/test_ko_bathe_concrete_3d.cpp) now verifies directly that a shorter
  crack-band length retains more tensile stress on the 3D post-peak branch
- [test_reduced_rc_column_continuum_baseline.cpp](/c:/MyLibs/fall_n/tests/test_reduced_rc_column_continuum_baseline.cpp) now checks that the promoted
  continuum baseline actually reports the fracture-secant tangent and the
  mesh-evaluated longitudinal characteristic length

The audit figures are now frozen in:

- [continuum local-model baseline overlay](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_continuum_local_model_baseline_overlay.png)
- [continuum local-model baseline error/timing](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_continuum_local_model_baseline_error_timing.png)
- [continuum local-model baseline frontier](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_continuum_local_model_baseline_frontier.png)

With that baseline promoted, I also reopened the monotonic amplitude sweep on
the **right** local continuum chain and froze it in:

- `scripts/run_reduced_rc_continuum_monotonic_amplitude_audit.py`
- [continuum monotonic amplitude summary](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_continuum_monotonic_amplitude_audit_v2_promoted/continuum_monotonic_amplitude_summary.json)

That second pass is useful because it stops treating the promoted slice as a
single `20 mm` anecdote. On the clamped-beam comparator, the promoted embedded
`Hex20 2x2x2 bias=3 + fixed-end lb` chain improves the RMS base-shear path
error **consistently** across the whole `2.5-20 mm` monotonic window:

- `2.5 mm`: `~3.3897e-1 -> ~3.3635e-1`
- `5.0 mm`: `~2.5472e-1 -> ~2.5167e-1`
- `10.0 mm`: `~2.0323e-1 -> ~2.0011e-1`
- `15.0 mm`: `~1.8673e-1 -> ~1.8378e-1`
- `20.0 mm`: `~1.7915e-1 -> ~1.7647e-1`

Just as importantly, it does **not** buy that improvement through a hidden cost
explosion. The promoted slice stays in the same cost class as the uniform
control and is often slightly cheaper (`~13.50 s` vs `~15.32 s` at `2.5 mm`,
`~17.25 s` vs `~19.34 s` at `10 mm`, essentially tied at `20 mm`). The
host-only biased slice remains faster, but still clearly worse physically, so
the embedded-bar chain remains the right promoted local RC model.

The amplitude figures are now frozen in:

- [continuum monotonic amplitude base shear](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_continuum_monotonic_amplitude_base_shear.png)
- [continuum monotonic amplitude error](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_continuum_monotonic_amplitude_error.png)
- [continuum monotonic amplitude timing](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_continuum_monotonic_amplitude_timing.png)

That promoted monotonic slice was still hiding one uncomfortable issue,
though: the preload semantics of the embedded-bar chain were never audited as a
first-class object. I reopened that question explicitly in
`scripts/run_reduced_rc_embedded_preload_transfer_audit.py` and froze the new
bundle in
[embedded preload transfer summary](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_embedded_preload_transfer_audit_v2_divtol/embedded_preload_transfer_summary.json).
The audit compares two things separately:

- how the axial preload enters the composite section:
  - `host_surface_only`
  - `composite_section_force_split`
- whether the open `Truss<3>` frontier on the continuum pilot is truly
  mechanical or only a PETSc divergence-test artifact.

The first answer is useful and honest. On the promoted `Hex20 2x2x2 + Truss<2>`
pilot, splitting the preload explicitly between the host face and the top rebar
end does improve the axial end mismatch, but only modestly:

- top rebar-face axial gap: `~3.46e-5 m -> ~2.93e-5 m`
- embedding-gap norm: `~3.47e-5 m -> ~2.93e-5 m`
- wall time stays in the same class: `~0.63 s -> ~0.70 s`

So the host-only preload path was not fully faithful, but the improved path
does not magically close the interface either. That matters for the future
multiscale use-case: the embedded local model needs explicit preload semantics,
not an implicit assumption that the steel and host stay automatically aligned.

The second answer is even more important. The open quadratic-bar frontier is
now sharply localized:

- `Hex20 + Truss<3> + preload` still fails before the first accepted step
- the same `Truss<3>` path succeeds on `Hex27 1x1x1`
- the same `Truss<3>` path also succeeds on `Hex20` when the preload is zero

That means the frontier is no longer honestly describable as
â€œquadratic embedded rebar is unstableâ€. The accurate statement is narrower:
the currently open slice is
`Hex20 + embedded quadratic rebar + axial preload`.

That narrower statement is now backed by a standalone benchmark rather than by
an informal unit-test intuition. The canonical bundle
[reboot_truss3_cyclic_compression_baseline](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_truss3_cyclic_compression_baseline/runtime_manifest.json)
drives a single `TrussElement<3,3>` through a purely compressive
Menegotto-Pinto return protocol using the **same steel factory** that feeds the
reduced RC fiber section and the promoted embedded continuum bars. The imposed
kinematics are affine (`u_m = 0.5 u_r`), so the exact solution is uniform axial
strain and any mismatch should come from the truss formulation itself.

The closure is essentially at machine precision:

- `max_abs_stress_error_mpa ~ 3.41e-13`
- `max_abs_tangent_error_mpa ~ 4.51e-10`
- `max_abs_element_tangent_error_mpa ~ 3.35e-10`
- `max_abs_gp_strain_spread ~ 3.04e-18`
- `max_abs_middle_node_force_mn ~ 6.81e-17`

That audit also corrected an easy-to-misread part of the validation logic: for
the quadratic bar, the equivalent axial tangent is not the raw `K_rr` entry.
It must be projected along the imposed affine path and then mapped back with
`L/A`. Once read that way, the element-level tangent closes with the direct
Menegotto tangent just as tightly as the stress history itself.

One constitutive detail is worth making explicit because it can easily be
misread as an element defect. In the same canonical bundle, the final return to
zero total strain still carries about `+3.63e2 MPa` of residual stress. The
direct material point and the standalone quadratic truss reproduce the same
value to machine precision, so this is a feature of the current
Menegotto-Pinto parameterization under that protocol, not a bug in
`TrussElement<3,3>`.

The canonical standalone truss figures are now frozen in:

- [quadratic truss Menegotto equivalence](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_truss3_cyclic_menegotto_equivalence.png)
- [quadratic truss force-displacement loop](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_truss3_cyclic_force_displacement.png)

That standalone closure made it possible to reopen the full steel chain without
guessing where the mismatch lived. The canonical artifact is now
[steel chain audit summary](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_steel_chain_audit_75mm/steel_chain_summary.json),
which compares, on the same `75 mm` cyclic window:

- direct Menegotto-Pinto replay,
- standalone `Truss<3>` replay,
- the promoted continuum embedded bar, and
- the matched structural steel trace used in the beam-vs-continuum bridge.

The useful read is now stronger and more nuanced:

- on the promoted continuum bar, `material -> truss -> embedded bar` collapses
  essentially to machine precision
  (`rms_abs_stress_error_mpa ~ 2.30e-7`, active relative RMS `~ 1.34e-9`);
- on the already interpolated matched structural trace, direct material and
  standalone truss replays still show a much larger gap
  (`rms_abs_stress_error_mpa ~ 12.41`, active relative RMS `~ 1.14e-1`);
- the same replay run on the actual upper structural site is still essentially
  exact, but the lower site is no longer tight at this amplitude
  (`rms_abs_stress_error_mpa ~ 3.74`, active relative RMS `~ 6.89e-2`); and
- interpolating those **site replays** still closes far better than replaying
  the already matched beam trace directly
  (`rms_abs_stress_error_mpa ~ 1.87`, active relative RMS `~ 4.27e-2`).

That changes the diagnosis in an important way. The large structural-side steel
gap is no longer a credible sign that the Menegotto assignment or the
`Truss<3>` carrier is wrong. It is primarily a trace-construction effect: the
matched beam trace is an axial interpolation between two section fiber
histories, not a single constitutive material point. At `75 mm`, the same audit
also shows something physically useful: the lower structural station is now the
one drifting first, while the upper station still tracks the constitutive
replay almost exactly. So the open frontier is no longer on the continuum steel
carrier at all; it is on the structural-side distributed section response under
large reversal. The steel-chain figures that preserve that reasoning are now
frozen in:

- [steel chain replay 75 mm](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_steel_chain_replay_75mm.png)
- [steel chain drift 75 mm](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_steel_chain_drift_75mm.png)
- [steel chain structural sites 75 mm](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_steel_chain_structural_sites_75mm.png)

I also reopened the PETSc side carefully instead of assuming `SNES_DIVERGED_DTOL`
was definitive. The continuum benchmark now exposes
`--snes-divergence-tolerance` as a typed benchmark surface, and the same
embedded-preload audit was rerun with `divtol=unlimited` on the minimal
`Hex20 1x1x1 + Truss<3>` host. That did **not** rescue the case. It simply
turned the early `DTOL` failure into a long stagnation that exhausted the
`300 s` audit budget. That is exactly the kind of result we want in a rebooted
validation chain: it tells us the frontier is not a false positive created by
PETSc's default divergence test. The typed solver seam was still worth adding,
because now that conclusion is explicit, reproducible, and reusable by both the
static and future dynamic pilots.

The new preload/frontier figure is frozen in:

- [embedded preload transfer frontier](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_embedded_preload_transfer_frontier.png)

At this point the promoted continuum read is much cleaner:

- `Hex20 -> Truss<2>` remains the promoted efficient local RC baseline
- `Hex27 -> Truss<3>` remains a viable higher-fidelity host/bar route
- `Hex20 -> Truss<3>` should not be promoted until the preload frontier is
  closed for real, not papered over by solver settings

That same frontier also forced a useful audit of the host-bar coupling itself.
The current penalty-coupling setup does anchor interior quadratic rebar nodes to
the continuum host: the new regression in
[`tests/test_reduced_rc_column_continuum_baseline.cpp`](/c:/MyLibs/fall_n/tests/test_reduced_rc_column_continuum_baseline.cpp)
freezes the distinction explicitly. On a single `Hex20` host, a two-node
embedded truss leaves no interior bar DOF to couple, while a three-node
embedded truss does couple its midpoint. So the open `Hex20 + Truss<3> +
preload` slice should not be read as a fictitious slip caused by forgotten
interior bar nodes. The stronger kinematic closure check is already frozen in
[structural continuum steel hysteresis summary](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_structural_continuum_promoted_cyclic_30mm_audit/structural_continuum_steel_hysteresis_summary.json):

- `max_abs_host_bar_axial_strain_gap ~ 3.37e-10`
- `rms_abs_host_bar_axial_strain_gap ~ 1.21e-10`
- `max_abs_projected_gap_norm_m ~ 2.85e-11`

So the embedded `Truss<3>` midpoint is not only present; it is kinematically
locked to the host on the promoted path to essentially machine precision.

I also reopened the predictor/seeding question instead of assuming that a
secant guess must help near fracture. The canonical audit is now frozen in
[continuum predictor audit summary](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_continuum_predictor_audit_v4_full/continuum_predictor_audit_summary.json)
on the promoted local continuum slice (`Hex20 2x2x2`, embedded bars,
`production-stabilized + fracture-secant + fixed-end lb`, `newton-l2-only`).
The read is now materially stronger than the earlier `v1` pass:

- all five typed policies complete the full promoted window
  (`2.5, 5, 10, 20 mm`);
- none of them moves first-crack drift or the macroscopic base-shear response;
- `secant` is still best on the easiest pre-crack case (`2.5 mm`, `~18.94 s`);
- `hybrid-secant-linearized` is fastest at `5, 10, 20 mm`
  (`~19.71 s`, `~26.87 s`, `~44.60 s`);
- across the whole promoted window, `hybrid-secant-linearized` is now the
  cheapest policy on average (`~28.71 s` mean wall time), ahead of
  `secant` (`~29.21 s`) and `current-state-only` (`~29.45 s`);
- `linearized-equilibrium` remains slightly heavier on the promoted slice
  (`~30.65 s` mean wall time), but it is no longer just a conceptual branch.

That last point matters because the direct frontier probes on
`Hex20 + Truss<3> + preload` changed the architectural conclusion. The old
baseline `current-state-only` still fails before the first accepted step, but
both of the new typed seeds reopen that pathological slice:

- [current-state probe](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_hex20_truss3_preload_current_state_probe/runtime_manifest.json):
  `completed_successfully = false`
- [linearized-equilibrium probe](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_hex20_truss3_preload_linearized_seed_probe/runtime_manifest.json):
  `completed_successfully = true`
- [hybrid secant + linearized fallback probe](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_hex20_truss3_preload_hybrid_seed_probe/runtime_manifest.json):
  `completed_successfully = true`

That is why the continuum benchmark default was promoted from
`current_state_only` to `secant_with_linearized_fallback`. The new predictor
seam in `src/analysis/NLAnalysis.hh` is now doing two jobs at once:

- it stays cheap on the already-promoted `Hex20 2x2x2` window, where secant
  history is enough most of the time; and
- it provides a robust first-step rescue path on the still-open
  `Hex20 + Truss<3> + preload` frontier, where no secant history exists yet.

The corresponding timing/iteration figures are now also frozen in:

- [continuum predictor timing](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_continuum_predictor_timing.png)
- [continuum predictor Newton workload](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_continuum_predictor_newton.png)
- [continuum predictor fracture onset](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_continuum_predictor_fracture_onset.png)

The next pass then stepped back from predictors and audited the physical
foundation of the promoted continuum slice itself instead of assuming that the
remaining cost or crack-opening pathologies came from Ko-Bathe "being too
heavy". The canonical artifact is now the
[continuum cover/core foundation summary](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_continuum_cover_core_audit/continuum_cover_core_foundation_summary.json),
driven by `scripts/run_reduced_rc_continuum_cover_core_foundation_audit.py`.
That audit changed three parts of the local RC baseline together:

- the host continuum can now be zoned explicitly as `cover_core_split`;
- the transverse mesh can now be `cover_aligned`, so outer unconfined strips
  and the confined core are represented by actual host elements; and
- the reinforced continuum can now carry an explicit
  `boundary_matched_eight_bar` layout instead of forcing all longitudinal bars
  into the interior host cells.

The new read is stronger than a simple mesh-sensitivity anecdote:

- on `Hex20 4x4x2` without steel, moving from `uniform_reference` to
  `cover_core_split + cover_aligned` changes peak base shear only modestly
  (`~17.06 kN -> ~17.44 kN`), which means the zoning is physically worth
  keeping but is not by itself the dominant source of the global gap;
- on the same `4x4x2` host, the interior-bar and boundary-bar layouts are both
  kinematically admissible, but they are **not** equivalent:
  `embedded interior` reaches `~21.07 kN`, while `embedded boundary` reaches
  `~23.59 kN`, so boundary bars must remain an explicit comparison branch and
  cannot be silently promoted as "the same local model";
- an explicit `cover_core_interface_eight_bar` probe collapses exactly to the
  same response as the promoted `structural_matched_eight_bar` branch for the
  current canonical reduced-column geometry, because the canonical eight steel
  paths already lie on the cover/core interfaces; and
- on a strongly biased `Hex20 4x4x4` host (`bias=3`), the old
  `mean_longitudinal_host_edge_mm` crack-band length produces an absurd crack
  opening (`~1.083 m`) while leaving the global base-shear trace almost
  unchanged (`~8.84 kN` in both runs); and
- replacing it with `fixed_end_longitudinal_host_edge_mm` restores a physically
  plausible crack opening (`~3.87e-4 m`) with essentially the same global force
  path.

That last point matters a lot. It means the problem was not simply "Ko-Bathe is
wrong" or "the continuum is too nonlinear"; the problematic part of the chain
was the interaction between a strongly biased longitudinal mesh and a
characteristic-length policy that stopped being admissible on that mesh. The
correct promotion is therefore:

- keep Ko-Bathe on the current local RC continuum path;
- allow strong fixed-end bias only together with the fixed-end host-edge crack
  length; and
- treat `Hex20 4x4x10` as an operational frontier, not as a promoted local
  baseline yet.

The corresponding figures are now frozen in:

- [cover/core monotonic overlay](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_continuum_cover_core_monotonic_overlay.png)
- [characteristic-length sensitivity](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_continuum_characteristic_length_sensitivity.png)
- [cover/core timing](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_continuum_cover_core_timing.png)

That same foundation question was then rerun in a narrower and more falsifiable
mesh audit, frozen in
[continuum monotonic mesh-foundation summary](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_continuum_monotonic_mesh_foundation_audit/continuum_monotonic_mesh_foundation_summary.json).
This pass kept the same nonlinear concrete profile, the same embedded interior
bars, the same axial preload, and the same monotonic push-over to `20 mm`,
while comparing:

- `Hex20 4x4x2 uniform` as the current promoted baseline,
- `Hex20 6x6x2 uniform` as a pure transverse-refinement control, and
- `Hex20 4x4x4 bias=3 fixed-end` as a longitudinal-refinement control.

The result is useful because it removes another tempting explanation. Pure
transverse refinement barely moves the physics:

- `Hex20 4x4x2`: `~21.069 kN`, RMS vs clamped structural control `~3.13e-1`
- `Hex20 6x6x2`: `~21.059 kN`, RMS vs clamped structural control `~3.14e-1`

So, on this monotonic `20 mm` slice, the promoted baseline is not obviously
too coarse in the section. The longitudinally refined branch is more
interesting physically, but also more expensive:

- `Hex20 4x4x4 bias=3 fixed-end`: `~20.637 kN`, RMS `~3.25e-1`,
  `642` cracked Gauss points, `~0.761 mm` crack opening, and
  `~115.2 MPa` peak steel stress.

That is not enough to promote it as a new baseline yet, but it is enough to
say something cleaner: if a larger physical shift is still needed, it is more
likely to come from the longitudinal direction than from simply packing more
elements across the section. The corresponding figures are now frozen in:

- [continuum monotonic mesh-foundation overlay](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_continuum_monotonic_mesh_foundation_overlay.png)
- [continuum monotonic mesh-foundation overview](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_continuum_monotonic_mesh_foundation_overview.png)

With that foundation in place, the next honest question was whether the same
read would survive the first short cyclic window. The new canonical artifact is
the
[continuum cover/core cyclic summary](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_continuum_cover_core_cyclic_audit/continuum_cover_core_cyclic_summary.json),
driven by `scripts/run_reduced_rc_continuum_cover_core_cyclic_audit.py`. The
short cyclic window keeps the same axial preload, the same `Hex20` host family,
and the same cover/core-aware section, while comparing:

- `continuum-only`
- `embedded interior`
- `embedded boundary`
- and a first `4x4x4 bias=3 + fixed-end lb` exploratory branch

The results are already quite informative:

- the three `Hex20 4x4x2` branches all close the cyclic window up to `10 mm`;
- `continuum-only` reaches `~8.91 kN`, `264` cracked Gauss points, and
  `max crack opening ~ 3.06e-4 m`;
- `embedded interior` reaches `~10.71 kN`, `252` cracked Gauss points, and
  `peak steel stress ~ 44.8 MPa`;
- `embedded boundary` reaches `~11.97 kN`, `252` cracked Gauss points, and
  `peak steel stress ~ 57.3 MPa`;
- the boundary-bar branch is again cheaper (`~878 s`) than the promoted
  interior-bar branch (`~1239 s`), but it also shows a larger host-bar axial
  strain gap (`~1.41e-4` vs `~9.84e-5`); and
- the first higher-cost branch,
  `Hex20 4x4x4 bias=3 + fixed-end lb + embedded interior`, still does not close
  the cyclic window within the present `4800 s` budget.

So the monotonic conclusion survives reversal: boundary bars are a valid and
useful comparison branch, but they remain physically different from the
interior embedded chain and should not be promoted as the same local model.
The promoted efficient local baseline remains the cover/core-aware
`Hex20 4x4x2` host with interior embedded bars; the strongly biased `4x4x4`
branch stays an operational frontier.

The next closure pass then asked a narrower question than â€œshould Ko-Bathe be
replaced?â€: whether the promoted local slice actually benefits from changing
only the crack-stabilization profile while keeping the same constitutive
family. The new canonical artifact is the
[continuum cover/core profile probe summary](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_continuum_cover_core_profile_probe/continuum_cover_core_profile_probe_summary.json),
which compares `benchmark_reference` against `production_stabilized` on the
promoted `Hex20 4x4x2 + cover_core_split + cover_aligned + structural bars`
slice in both monotonic and short cyclic windows. The read is much cleaner than
the old â€œKo-Bathe is too heavyâ€ suspicion:

- both profiles produce the same peak base shear in monotonic and short cyclic
  windows (`monotonic_force_gap_fraction = 0`, `cyclic_force_gap_fraction = 0`);
- peak crack opening and peak steel stress also remain essentially unchanged;
- `production_stabilized` materially reduces the counted active cracked Gauss
  points (`282 -> 258` monotonic, `432 -> 252` cyclic); and
- timing stays at least competitive (`152.7 s` vs `163.4 s` monotonic, nearly
  tied in the short cyclic window).

So the stabilized profile is not behaving like a hidden hard-coded patch. It
keeps the same constitutive family, preserves the macro response on the
promoted local slice, and regularizes the internal crack activity enough to be
the right efficient default for the future multiscale route.

That is also the point where the Spiliopoulos-Lykidis solid RC methodology is
actually useful to adapt. Three ideas transfer well to `fall_n`:

- explicit embedded `Truss<3>` reinforcement as a first-class continuum path;
- loading/unloading crack-state updates inside the nonlinear iteration, rather
  than treating reversal as only a load-step event; and
- a dynamic viewpoint in which time integration becomes a sequence of nonlinear
  equilibrium solves, which lines up naturally with the typed PETSc
  solve-policy seam already opened in the code.

What the paper does **not** justify here is a blind constitutive replacement.
On the promoted local slice, the stronger conclusion is narrower: keep the
family, keep the stabilized profile, and continue removing cost from the
host/bar chain and the characteristic-length policy before changing the
material model itself.

The corresponding cyclic figures are now frozen in:

- [cover/core cyclic hysteresis 20 mm](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_continuum_cover_core_cyclic_hysteresis_20mm.png)
- [cover/core cyclic crack opening 20 mm](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_continuum_cover_core_cyclic_crack_opening_20mm.png)
- [cover/core cyclic rebar stress 20 mm](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_continuum_cover_core_cyclic_rebar_stress_20mm.png)
- [cover/core profile monotonic overlay](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_continuum_cover_core_profile_monotonic_overlay.png)
- [cover/core profile cyclic overlay](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_continuum_cover_core_profile_cyclic_overlay.png)
- [cover/core profile timing](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_continuum_cover_core_profile_timing.png)

Before reopening the next large nonlinear window, we also stopped and audited
two foundation issues that were easy to miss because they live below the
headline constitutive model.

First, the support resultant exported by the continuum benchmark is still read
as a **discrete reaction sum**, not as a post-hoc face traction integral. That
choice is fine, even for `Hex20` and `Hex27`, as long as the support set is the
actual constrained support and not just the host face. The new regression in
[`tests/test_reduced_rc_column_continuum_baseline.cpp`](/c:/MyLibs/fall_n/tests/test_reduced_rc_column_continuum_baseline.cpp)
now freezes exactly that:

- `Hex20` and `Hex27` recover the applied top-face traction magnitude through
  the base reaction, despite the higher-order consistent nodal force pattern;
- the promoted embedded branch now includes the constrained rebar-endcap nodes
  in the support resultant, so `base_shear` and `base_axial_reaction` no
  longer undercount the support force when axial preload is split between host
  and steel.

That correction matters. From this point on, any old embedded-continuum bundle
generated before the support-resultant fix should be treated as **historical**,
not as a quantitative reference for base-shear comparison.

The next audit exposed a deeper PETSc coupling issue. The embedded-bar
penalty Jacobian was previously assembled as if a global point offset could be
advanced by `offset + component`. That is not safe when PETSc compresses
partially constrained nodes. The fix is now component-wise: each local rebar or
host component is mapped through the DM local-to-global map, and only truly
constrained components are omitted. With that correction the benchmark can keep
the rebar end nodes coupled without creating artificial axial slip:

- `Hex8 2x2x4`, corotational, `200 mm`: `|V|max ~= 157.65 kN`,
  `max embedding gap ~= 2.5e-10 m`, `|sigma_s|max ~= 434.5 MPa`,
  solve time `~12.2 s`.
- `Hex8 4x4x8`, corotational, `200 mm`: `|V|max ~= 60.39 kN`,
  `max embedding gap ~= 3.4e-10 m`, `|sigma_s|max ~= 439.0 MPa`,
  solve time `~123.6 s`.

Preliminary corrected hysteresis curves are stored at
[component-wise coupling hysteresis 200 mm](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_continuum_componentwise_coupling_200mm.svg).

That fix changes the next structural--continuum question.  The embedded bars
are no longer allowed to slip numerically, so the remaining gap is a true local
equivalence problem: can the continuum sampling and material regularization
reproduce the steel loop seen by the Timoshenko fiber section?  The promoted
comparison script now records the required section-axis transform explicitly:
the structural active coordinate maps with opposite sign to continuum
`RebarBar::ly` (the prism local x coordinate), while the passive coordinate maps
directly to `RebarBar::lz`.  Without this map the benchmark can accidentally
pair a bar with the physically opposite structural fiber.

The first corrected `50,100,150,200 mm` local-equivalence matrix is:

- `Hex8 2x2x4`, uniform, first steel GP at `z ~= 169.1 mm`:
  `W_c/W_s ~= 0.23`, global peak-normalized RMS error `~= 1.12`,
  steel stress RMS error `~= 0.39`, solve `~59.5 s`.
- `Hex8 2x2x12`, longitudinal bias `2.0`, first steel GP at
  `z ~= 4.7 mm`: `W_c/W_s ~= 0.77`, global peak-normalized RMS error
  `~= 0.39`, steel stress RMS error `~= 0.25`, solve `~231.8 s`.
- `Hex8 4x4x8`, cover/core aligned, uniform in height, first steel GP at
  `z ~= 84.5 mm`: `W_c/W_s ~= 0.11`, global peak-normalized RMS error
  `~= 0.43`, steel stress RMS error `~= 0.39`, solve `~496.1 s`.
- `Hex8 4x4x8`, cover/core aligned, bias `1.5`, first steel GP at
  `z ~= 29.9 mm`: `W_c/W_s ~= 0.36`, global peak-normalized RMS error
  `~= 0.35`, steel stress RMS error `~= 0.26`, solve `~542.8 s`.
- `Hex8 4x4x12`, cover/core aligned, bias `2.0`, first steel GP at
  `z ~= 4.7 mm`: `W_c/W_s ~= 0.69`, global peak-normalized RMS error
  `~= 0.37`, steel stress RMS error `~= 0.25`, solve `~767.1 s`.

The strongest promoted candidate in this pass is therefore
`Hex8 4x4x12` with cover/core transverse fidelity and a `bias = 2.0`
longitudinal grid.  It does not yet close the structural loop perfectly, but it
is the first branch that combines a near-base steel integration point, a
physically meaningful cover/core section, closed embedded transfer, and a
global hysteresis error comparable to the cheaper uniform branch.  The remaining
gap is now a material/localization calibration problem, not a fake bar-slip
problem.  The compact comparison is stored at
[local equivalence matrix 200 mm](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_local_equivalence_matrix_200mm.png)
and the numeric rows are in
[`local_equivalence_matrix_summary.json`](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_structural_continuum_componentwise_fullpenalty_200mm_equivalence_matrix/local_equivalence_matrix_summary.json).

The promoted branch was then replotted with the kinematic policy in the
artifact name, so the comparison cannot be mistaken for a small-strain or TL
run.  The `Hex8 4x4x12 bias=2` cover/core case is explicitly
`corotational`, with `tensile_crack_band_damage_proxy`, full component-wise
embedded coupling and the fixed-end crack-band length.  The full `200 mm`
cyclic protocol completed with zero failed attempts, peak-normalized global
RMS error `~= 0.37`, peak-normalized steel-stress RMS error `~= 0.25`, steel
loop ratio `W_c/W_s ~= 0.69`, and maximum embedded gap `~= 3.35e-10 m`.

- [corotational global hysteresis overlay](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_structural_continuum_hysteresis_overlay_corotational_tensile_crack_band_damage_proxy_et0p1_4x4x12_bias2_200mm.png)
- [corotational steel hysteresis overlay](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_structural_continuum_steel_hysteresis_overlay_corotational_tensile_crack_band_damage_proxy_et0p1_4x4x12_bias2_200mm.png)
- [corotational embedded-transfer strain audit](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_structural_continuum_embedded_transfer_strain_corotational_tensile_crack_band_damage_proxy_et0p1_4x4x12_bias2_200mm.png)

The same stabilization pass also froze the steel-area contract across the
structural and continuum models. The promoted structural section and the
promoted continuum branch `structural_matched_eight_bar` both now report the
same resolved reinforcement block in `runtime_manifest.json`:

- bar count: `8`
- single bar area: `2.010619298e-4 m^2`
- total longitudinal steel area: `1.608495439e-3 m^2`
- gross section area: `6.25e-2 m^2`
- longitudinal steel ratio: `2.573592702e-2`

The continuum manifest additionally exports
`area_equivalent_to_structural_baseline`. That flag is `true` for the promoted
eight-bar interior branch and intentionally `false` for the future
`enriched_twelve_bar` branch, whose total steel area is larger by construction.

Second, we introduced a deliberately cheap host control branch instead of
guessing from the full Ko-Bathe pilot:

- new constitutive relation:
  [`src/materials/constitutive_models/lineal/OrthotropicBimodularConcreteProxy.hh`](/c:/MyLibs/fall_n/src/materials/constitutive_models/lineal/OrthotropicBimodularConcreteProxy.hh)
- new benchmark surface mode:
  `orthotropic_bimodular_proxy`
- canonical audit:
  [continuum proxy host summary](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_continuum_proxy_host_audit/continuum_proxy_host_summary.json)

This proxy host is intentionally narrow:

- compression-side modulus is kept at the usual code-level proxy
  `Ec ~= 4700 sqrt(f'c)` MPa,
- tension-side stiffness is reduced by a configurable ratio,
- no Poisson coupling is carried in the proxy tangent, and
- the embedded steel stays fully nonlinear Menegotto-Pinto.

That makes it useful as a **cheap control branch**, not as a silent replacement
for the promoted Ko-Bathe host. The audit already says something concrete:

- embedded interior, `30 mm`, `tr = 0.10`: `~77.7 s`, `~164.7 MPa`
- embedded interior, `50 mm`, `tr = 0.10`: `~102.4 s`, `~274.7 MPa`
- embedded interior, `75 mm`, `tr = 0.10`: `~204.3 s`, `~412.2 MPa`
- boundary bars, `75 mm`, `tr = 0.10`: `~202.5 s`, `~419.6 MPa`

So the cheap host does something valuable: it reaches a near-yield steel cycle
far earlier and far more cheaply than the current Ko-Bathe host. But it also
stays uncracked by construction and leaves a much larger host-bar strain gap
than the promoted local baseline. That is why the right promotion is narrow:

- keep `Ko-Bathe + production_stabilized + embedded interior` as the promoted
  local physics baseline;
- keep `orthotropic_bimodular_proxy + nonlinear steel` as an accelerated
  control branch for solver, protocol, and hysteresis exploration;
- keep `boundary` as a physical comparison branch, not as the future microscale
  baseline.

The proxy-host figures now frozen are:

- [proxy host hysteresis overlay](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_continuum_proxy_host_hysteresis_overlay.png)
- [proxy host steel hysteresis](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_continuum_proxy_host_rebar_stress.png)
- [proxy host timing](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_continuum_proxy_host_timing.png)
- [proxy host steel frontier](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_continuum_proxy_host_steel_frontier.png)

The first structural-vs-continuum proxy-host cyclic audit has also been pushed
to `200 mm` with the same eight-bar steel area and nonlinear Menegotto-Pinto
steel. The canonical `tr = 0.10` bundle is:

- [200 mm proxy structural-continuum summary](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_structural_continuum_proxy_host_cyclic_200mm_audit/structural_continuum_steel_hysteresis_summary.json)
- [200 mm base-shear overlay](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_structural_continuum_hysteresis_overlay_orthotropic_bimodular_proxy_et0p1_200mm.png)
- [200 mm steel hysteresis overlay](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_structural_continuum_steel_hysteresis_overlay_orthotropic_bimodular_proxy_et0p1_200mm.png)

Important outcome: the embedded steel is **not** trapped in the elastic range in
this branch. It reaches `|sigma_s| ~= 429 MPa` and
`|epsilon_s| ~= 6.59e-3`. The selected embedded-bar/host transfer gap is also
essentially closed (`max |Delta epsilon| ~= 2.1e-10` for the matched audit
trace), so the remaining discrepancy is not credibly explained by fictitious
slip between the truss and the continuum host.

The tension-stiffness sensitivity at `200 mm` is:

| `Et/Ec` | Continuum solve time | Global base-shear RMS rel. error | Steel stress RMS rel. error | Continuum steel loop work |
|---:|---:|---:|---:|---:|
| `0.10` | `65.0 s` | `1.30` | `0.72` | `4.65 MPa` |
| `0.03` | `68.1 s` | `1.13` | `0.80` | `3.06 MPa` |
| `0.01` | `83.7 s` | `1.13` | `0.88` | `0.90 MPa` |

This is a useful negative result. Reducing the tensile stiffness can soften the
global force path, but it also suppresses the local steel loop work instead of
making the proxy physically equivalent to the cracked structural model. The
proxy branch is therefore retained as a fast, traceable solver/material control
case; the physically promoted local continuum still needs a cracking host
(Ko-Bathe or a later XFEM/DG family) once this cheap chain has finished its
algorithmic audit.

The next inexpensive cracking control branch is now implemented as
`tensile_crack_band_damage_proxy`:

- constitutive relation:
  [`src/materials/constitutive_models/non_lineal/TensileCrackBandDamageConcreteProxy3D.hh`](/c:/MyLibs/fall_n/src/materials/constitutive_models/non_lineal/TensileCrackBandDamageConcreteProxy3D.hh)
- unit regression:
  [`tests/test_tensile_crack_band_damage_concrete_proxy_3d.cpp`](/c:/MyLibs/fall_n/tests/test_tensile_crack_band_damage_concrete_proxy_3d.cpp)
- first 6x6x12 cyclic audit:
  [damage-proxy 200 mm summary](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_structural_continuum_damage_proxy_cyclic_200mm_6x6x12_hex8_audit/structural_continuum_steel_hysteresis_summary.json)

This material is deliberately one notch more physical than the path-independent
bimodular proxy, but still much cheaper than the Ko-Bathe fixed-crack host. It
uses a Mazars-style positive-principal-strain scalar driver,
`kappa = sqrt(sum(<eps_i>_+^2))`, and an exponential crack-band softening law
regularized by `G_f / l_c`. Normal compression stiffness is recovered when the
diagnostic crack closes; normal tension and shear are degraded through a
secant tangent so the global solve avoids an extra local Newton problem.

The first full `50,100,150,200 mm` cyclic run used `Hex8 6x6x12`,
`cover_core_split`, `cover_aligned`, the same eight-bar steel area, nonlinear
Menegotto-Pinto steel, and `Et/Ec = 0.10`. It completed successfully, but it
also sharpened the next bottleneck:

| Observable | Value |
|---|---:|
| continuum solve time | `1653.2 s` |
| accepted runtime steps | `49 / 52` |
| average / maximum Newton iterations | `31.1 / 104` |
| peak base shear | `44.7 kN` |
| peak steel stress | `444.7 MPa` |
| peak steel strain | `1.45e-2` |
| peak cracked Gauss points | `3020 / 3456` |
| maximum crack opening strain | `1.65e-2` |
| max nearest-host damage around selected bar | `0.9999` |
| steel area equivalent to structural baseline | `true` |

The physical read is useful but not yet promotable. The host now cracks and
the steel clearly yields, so the old "steel stays elastic" concern is gone.
However, the continuum steel loop work remains far below the structural fiber
loop (`15.1 MPa` versus `120.4 MPa` in the matched trace), and the runtime is
too high for a future FE2 local model. The next continuum work should therefore
target the global nonlinear solve/assembly cost and the spatial distribution
of host damage, not another blind reduction of tensile stiffness.

The generated figures are already available in both documentation trees:

- [damage-proxy base shear 200 mm](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_structural_continuum_hysteresis_overlay_tensile_crack_band_damage_proxy_et0p1_200mm.png)
- [damage-proxy steel hysteresis 200 mm](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_structural_continuum_steel_hysteresis_overlay_tensile_crack_band_damage_proxy_et0p1_200mm.png)
- [damage-proxy steel stress vs drift 200 mm](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_structural_continuum_steel_stress_vs_drift_tensile_crack_band_damage_proxy_et0p1_200mm.png)
- [damage-proxy embedded transfer 200 mm](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_structural_continuum_embedded_transfer_strain_tensile_crack_band_damage_proxy_et0p1_200mm.png)

### PETSc factorization tuning and fixed-end longitudinal bias

The PETSc side was reopened at the matrix/linear-solve layer instead of
assuming that the remaining cost was purely material. The important
architectural change is that linear tuning now belongs to
`NonlinearSolveProfile`, not to ambient global PETSc options. The new
validation preset `newton-l2-lu-symbolic-reuse-only` keeps the same nonlinear
method (`SNESNEWTONLS` with L2/secant line search and `KSPPREONLY + PCLU`) but
sets:

- `PCFactorSetMatOrderingType(..., MATORDERINGRCM)`;
- `PCFactorSetReuseOrdering(..., PETSC_TRUE)`;
- `PCFactorSetReuseFill(..., PETSC_TRUE)`.

It deliberately does **not** enable `KSPSetReusePreconditioner` by default,
because that can reuse a stale numeric preconditioner across changed tangent
values. The safe default here is to reuse symbolic factorization information
for the fixed FE sparsity pattern, while still refactorizing the current
Jacobian values.

Microbenchmarks so far are conservative:

| Case | Policy | Solve time | Physical response |
|---|---:|---:|---|
| `Hex8 4x4x2`, monotonic `20 mm` | `newton-l2-only` | `2.293 s` | identical |
| `Hex8 4x4x2`, monotonic `20 mm` | `newton-l2-lu-symbolic-reuse-only` | `2.364 s` | identical |
| `Hex8 6x6x4`, monotonic `50 mm` | `newton-l2-only` | `51.799 s` | identical |
| `Hex8 6x6x4`, monotonic `50 mm` | `newton-l2-lu-symbolic-reuse-only` | `51.605 s` | identical |

So this is not yet promoted as a decisive speedup. Its value is that it makes
PETSc storage/factorization policy explicit, reproducible, and reusable by the
future dynamic `TS` path. More aggressive options such as lagged Jacobians,
lagged preconditioners, iterative KSPs, block AIJ storage, or symmetric storage
remain available only after a matrix-property audit; they should not be turned
on just because the continuum benchmark is expensive.

The bigger physical finding came from longitudinal sampling. The uniform
`Hex8 6x6x12` damage-proxy mesh places the first rebar Gauss point at
`z = 56.35 mm`; at `50 mm` monotonic drift, the largest base-layer steel
strain was only `~2.4e-3`. With the same `6x6x12` mesh but
`longitudinal_bias_power = 2`, the first point moves to `z = 4.70 mm` and the
same monotonic probe reaches `~1.1e-2`. That is a real modeling lesson: the
local steel discrepancy was not primarily a host-bar slip problem, because the
selected host-bar strain gap stayed around `1e-9`; it was largely a sampling
problem in the fixed-end plastic/cracking zone.

The biased cyclic audit completed the full `50,100,150,200 mm` protocol:

- artifact:
  [bias2 damage-proxy summary](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_structural_continuum_damage_proxy_cyclic_200mm_6x6x12_hex8_bias2_audit/structural_continuum_steel_hysteresis_summary.json)
- continuum solve time: `1345.6 s` versus `1653.2 s` for the uniform run;
- continuum steel loop work: `119.6 MPa` versus `15.1 MPa` for the uniform run;
- matched structural loop work: `169.0 MPa`;
- selected host-bar axial strain gap: `max ~2.5e-9`.

That narrows the physical gap substantially without changing the steel area,
material model, or embedded transfer. The remaining gap is now better framed:
the structural element enforces section kinematics, while the continuum permits
3D warping/crack localization. The next promoted continuum branch should
therefore keep fixed-end-biased longitudinal sampling, then compare Hex8/Hex20
and Ko-Bathe/XFEM-style hosts under the same explicit PETSc solver policy.

Updated figures:

- [bias2 damage-proxy base shear 200 mm](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_structural_continuum_hysteresis_overlay_tensile_crack_band_damage_proxy_et0p1_6x6x12_bias2_200mm.png)
- [bias2 damage-proxy steel hysteresis 200 mm](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_structural_continuum_steel_hysteresis_overlay_tensile_crack_band_damage_proxy_et0p1_6x6x12_bias2_200mm.png)
- [uniform damage-proxy steel hysteresis 200 mm](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_structural_continuum_steel_hysteresis_overlay_tensile_crack_band_damage_proxy_et0p1_6x6x12_200mm.png)

### Hex20/Hex27 escalation and PETSc linear-solver audit

The continuum driver now writes a `discretization` block in every runtime
manifest. That block records domain nodes/elements, host and rebar element
counts, embedded-node count, global PETSc DOFs, and matrix nonzeros. This was
added because high-order host elements change the cost by graph connectivity,
not just by the visible mesh dimensions.

Zero-amplitude setup probes on the same `6x6x12`, `cover_aligned`,
`longitudinal_bias_power = 2` mesh give:

| Host | PETSc DOFs | Matrix nonzeros | Setup/zero-step time |
|---|---:|---:|---:|
| `Hex8` | `1987` | `108783` | `23.6 s` |
| `Hex20` + 3-node bars | `6979` | `906447` | `112.0 s` |
| `Hex27` + 3-node bars | `12559` | `1953579` | `119.2 s` |

So the high-order switch is supported, but it is not a free refinement. A
`Hex20 4x4x8` monotonic `50 mm` damage-proxy probe completed with both LU
policies and identical response:

| Policy | Result | Solve time | Notes |
|---|---:|---:|---|
| `newton-l2-only` | completed | `182.6 s` | `2463` DOFs, `268539` nonzeros |
| `newton-l2-lu-symbolic-reuse-only` | completed | `184.0 s` | same response; timing difference is within run noise |
| `newton-l2-gmres-ilu1-only` | failed | `515.2 s` | KSP hit `1000` GMRES iterations and returned `KSP_DIVERGED_ITS` |

This makes the immediate recommendation conservative: use `Hex20` as the
next high-order verification branch on smaller/intermediate meshes, but do
not promote `Hex20/Hex27 6x6x12` to the full cyclic validation until we add a
stronger scalable preconditioner or reduce the penalty-coupling conditioning
problem. `Hex27` remains valuable as a geometry/interpolation control, but the
current direct-LU route is too expensive for it to be the default local model.

### Hex8 fixed-end adaptive longitudinal refinement

The next long-run audit kept `Hex8` and refined only along the column axis near
the fixed end. This is the right refinement direction for the present local
column: the steel/plastic hinge discrepancy is concentrated near the
empotramiento, and uniform refinement far from that region mostly buys cost.

Two lessons emerged. First, an overly aggressive bias is counterproductive:
`6x6x18` with `longitudinal_bias_power = 2.5` places the first steel Gauss
points at `0.49, 1.84, 4.62 mm` and drives the crack-band characteristic
length down to `2.33 mm`; the full `200 mm` cyclic run did not finish within
the long-run window. Second, keeping `bias = 2` and increasing `nz` is much
better behaved.

| Case | Status | Solve time | Peak base shear | Peak steel stress | Peak steel strain | Max crack opening |
|---|---:|---:|---:|---:|---:|---:|
| `Hex8 6x6x12 bias=2` | completed | `1345.6 s` | `31.1 kN` | `586.3 MPa` | `8.53e-2` | `9.94e-2` |
| `Hex8 6x6x16 bias=2` | completed | `3053.8 s` | `29.1 kN` | `558.3 MPa` | `7.13e-2` | `8.48e-2` |
| `Hex8 6x6x20 bias=2` | timeout | `> 5 h` | `n/a` | `n/a` | `n/a` | `n/a` |

The current completed high-resolution candidate is therefore
`Hex8 6x6x16 bias=2`, not because convergence is mathematically closed, but
because it is the most refined completed case before the present
solver/regularization chain becomes impractical. The `6x6x20` and
`bias=2.5` frontiers should be revisited only after improving penalty scaling,
preconditioning, or the crack-band regularization strategy.

Artifact:
[Hex8 fixed-end refinement summary](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_continuum_hex8_fixed_end_refinement_audit/hex8_fixed_end_refinement_summary.json)

### Continuum kinematic policy selection

The continuum reduced-column benchmark now exposes the finite-kinematics seam
directly through:

```powershell
--continuum-kinematics small-strain|total-lagrangian|updated-lagrangian|corotational
```

The default remains `small-strain` for backward-compatible baselines. The
runtime dispatcher now exposes `total-lagrangian`, `updated-lagrangian`, and a
partial `corotational` solid branch while keeping the element assembly
templated by the kinematic policy. This is deliberately done at the
`RunSpec`/manifest layer, not as a hidden benchmark flag, so future Python or
Julia wrappers can select the same policy without touching solver internals.

The immediate physical motivation is rigid-motion objectivity. A pure rigid
rotation has \(F = R\) and the Total Lagrangian strain is
\(E = \frac12(R^T R-I)=0\), so TL does not convert rigid rotation into
spurious strain. The current large-drift continuum column can therefore be
audited against the previous `small-strain` path without conflating material
damage with geometric strain pollution. UL is available as an executable
spatial branch, but it remains more experimental in the audit taxonomy because
its current implementation still needs broader nonlinear validation before it
becomes the default local-model route.

`corotational` now extracts the polar rotation \(R\) from \(F=R U\), drives the
material with the corotated Biot-like strain
\(\hat e=\operatorname{sym}(R^T F)-I\), and rotates stress/tangent back to the
assembly frame. Its tangent currently freezes \(R\), so it is a useful
benchmarking and architecture branch, not the reference finite-kinematics
solid formulation. TL remains the scientific reference until the missing
\(\partial R/\partial u\) terms are audited.

Smoke evidence:

| Policy | Status | Peak base shear in 1x1x1 elastic smoke | Solve time |
|---|---:|---:|---:|
| `small-strain` | completed | `0.13247 MN` | `0.145 s` |
| `total-lagrangian` | completed | `0.13247 MN` | `0.423 s` |
| `updated-lagrangian` | completed | `0.13247 MN` | `0.390 s` |
| `corotational` | completed, partial audit | `0.13247 MN` | `0.492 s` |

The small-displacement agreement is now covered by
`fall_n_reduced_rc_column_continuum_baseline_test`, and the rigid-rotation
objectivity of both TL and the corotational rotation filter remains covered by
`fall_n_kinematic_test`.

### Intermediate cyclic crack-band concrete branch

The `tensile_crack_band_damage_proxy` branch was useful for cheap steel-host
audits, but it had a physical limitation: it used the reduced tensile modulus
`Et = 0.10 Ec` before cracking. The structural Kent-Park fiber section cracks
from the full initial concrete modulus, so the proxy delayed crack initiation
and mixed two distinct effects: initial tensile stiffness and post-crack
tension stiffening.

The new `cyclic-crack-band-concrete` material mode separates those effects:

- tension and compression start from the Kent-Park-compatible `Ec`;
- tensile damage is driven by a Mazars-style norm of positive principal
  strains;
- post-peak tension is regularized by `Gf / lc` following crack-band logic;
- crack closure is history-aware, so a large open crack does not recover the
  full Kent-Park compression branch as soon as the strain crosses zero;
- a small residual compression-transfer floor keeps open cracks numerically
  stable without reintroducing full contact too early;
- `--concrete-tension-stiffness-ratio` now acts as a post-crack
  tension-stiffening floor for this branch, not as the initial tensile modulus.

This implementation is deliberately an intermediate validation material. It is
not a replacement for Ko-Bathe, XFEM, or a future fixed/rotating-crack model
with explicit shear transfer and cyclic compressive damage. Its purpose is to
test the missing physics between the bimodular proxy and the full local
concrete law without paying for a local return-mapping solve at every Gauss
point.

The first `Hex8 2x2x12 bias=2`, corotational, full-penalty embedded-bar audit
to `200 mm` completed with `cyclic-crack-band-concrete`. A dedicated sweep now
keeps every modeling choice fixed and varies only the post-crack
tension-stiffening floor:

| Case | Solve time | Base-shear RMS error | Steel-stress RMS error | Steel loop ratio |
|---|---:|---:|---:|---:|
| `tensile_crack_band_damage_proxy Et/Ec=0.10` | `231.8 s` | `0.386` | `0.254` | `0.690` |
| `cyclic_crack_band_concrete ts=0.000` | `332.0 s` | `0.420` | `0.262` | `1.313` |
| `cyclic_crack_band_concrete ts=0.005` | `317.8 s` | `0.419` | `0.301` | `0.791` |
| `cyclic_crack_band_concrete ts=0.010` | `291.9 s` | `0.417` | `0.344` | `0.566` |
| `cyclic_crack_band_concrete ts=0.020` | `486.7 s` | `0.429` | `0.304` | `0.435` |

So the new material clarified the physics but is not promoted yet. The
history-aware closure removed the non-physical reaction spikes seen in the
first full-`Ec` attempt, and the secant/closure tangent made the branch
tractable. The sweep also shows something more important than a single
calibrated value: `ts=0.005` is the best point in this small family, but the
global base-shear RMS remains almost flat near `0.42`. The tension-stiffening
floor changes local steel energy; it does not by itself close the structural
vs continuum equivalence gap.

To isolate whether that remaining gap was caused by using a different concrete
law, a second diagnostic material was added:
`componentwise-kent-park-concrete`. This branch reuses the exact uniaxial
Kent-Park response used by the structural fibers, once per 3D normal component,
while exporting a secant-positive tangent to PETSc in softening branches. It is
a response-equivalence control branch, not a production 3D concrete law.

| Componentwise Kent-Park control | Solve time | Base-shear RMS error | Steel-stress RMS error | Steel loop ratio |
|---|---:|---:|---:|---:|
| `Hex8 2x2x12 bias=2 uniform` | `172.7 s` | `0.522` | `0.308` | `2.257` |
| `Hex8 4x4x12 bias=2 cover-aligned` | `735.0 s` | `0.472` | `0.179` | `2.009` |

This is an important negative result. Making the continuum host response
Kent-Park-equivalent improves neither the global hysteresis nor the steel
energy enough; refining the section to a cover-aligned `4x4` grid improves the
local stress RMS, but the loop work is still about twice the structural
reference and the cost increases sharply. The current frontier is therefore
not simply "choose a better concrete proxy". The next validation step should
focus on continuum kinematics/localization and the local boundary value
problem: plane-section structural kinematics versus deformable 3D host,
localization width near the fixed end, and whether an enriched XFEM/DG or
force-based/local constraint strategy is needed to obtain a representative
multiscale cell without excessive mesh refinement.

Artifacts:

- [cyclic crack-band summary, ts=0.02](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_structural_continuum_cyclic_crack_band_tensionstiff02_hex8_2x2x12_bias2_uniform_200mm/structural_continuum_steel_hysteresis_summary.json)
- [cyclic crack-band tension sweep summary](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_cyclic_crack_band_tension_sweep_200mm/cyclic_crack_band_tension_sweep_summary.json)
- [cyclic crack-band tension sweep metrics](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_cyclic_crack_band_tension_sweep_200mm_metrics.png)
- [componentwise Kent-Park 4x4 cover-aligned summary](/c:/MyLibs/fall_n/data/output/cyclic_validation/reboot_structural_continuum_componentwise_kentpark_hex8_4x4x12_bias2_covercore_200mm/structural_continuum_steel_hysteresis_summary.json)
- [cyclic crack-band base shear, ts=0.02](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_structural_continuum_hysteresis_overlay_corotational_cyclic_crack_band_concrete_ts0p02_2x2x12_bias2_200mm.png)
- [cyclic crack-band steel hysteresis, ts=0.02](/c:/MyLibs/fall_n/doc/figures/validation_reboot/reduced_rc_structural_continuum_steel_hysteresis_overlay_corotational_cyclic_crack_band_concrete_ts0p02_2x2x12_bias2_200mm.png)
- [`CyclicCrackBandConcrete3D` unit test](/c:/MyLibs/fall_n/tests/test_cyclic_crack_band_concrete_3d.cpp)
- [`ComponentwiseKentParkConcrete3D` unit test](/c:/MyLibs/fall_n/tests/test_componentwise_kent_park_concrete_3d.cpp)

### Global XFEM local-model branch

The latest validation pass adds a stronger conclusion to the continuum story:
standard smeared 3D continuum elements remain useful as controls, but they do
not close the structural-vs-local equivalence by mesh refinement alone. The
closest branch so far is the shifted-Heaviside global XFEM column with
Menegotto-Pinto truss bars tied to the enriched host and
`CyclicCrackBandConcrete3D` as the nonlinear concrete host.

Two implementation details matter for reproducibility:

- the benchmark exposes `--global-xfem-crack-z-m`, so the physical crack-plane
  location is no longer an accidental midpoint of the first host element;
- prescribed crack planes are rejected if they coincide with a mesh level,
  because that would make the shifted-Heaviside split degenerate.

Against the high-resolution structural reference (`N=10`, Lobatto stations,
fine fiber section, `0.02 MN` axial compression), the first protocol-aligned
coarse XFEM comparison reached the same global force scale up to `200 mm`.
After the documented base-reaction sign alignment, the peak-normalized RMS
gap was about `7.39e-2` and the XFEM/structural peak-shear ratio was `1.107`.
That is the first positive evidence that the enriched local branch is
physically closer to the structural model than the basic continuum branch.

The follow-up refinement matrix fixed the crack plane at `zc = 0.60 m` and
varied uniform and fixed-end-biased Hex8 host meshes. The best completed case
was `2x2x2` with fixed-end bias:

| XFEM host case | Reduced PETSc DOFs | Wall time | Peak ratio | Peak-normalized RMS |
|---|---:|---:|---:|---:|
| `1x1x2 uniform` | `60` | `23.6 s` | `1.153` | `0.175` |
| `1x1x4 uniform` | `132` | `114.2 s` | `1.610` | `0.0978` |
| `2x2x2 uniform` | `95` | `103.6 s` | `1.673` | `0.136` |
| `2x2x2 fixed-end bias2` | `95` | `67.9 s` | `1.374` | `0.0959` |

The `1x1` biased cases and the `2x2x4` biased case timed out under the current
secant Newton cascade. They are kept as algorithmic evidence, not promoted as
converged physics. The healthy conclusion is deliberately conservative:
XFEM moves `fall_n` much closer to a representative local RC-column model, but
the remaining closure depends on a small physical convergence matrix over
crack location, cohesive fracture energy, and opening-dependent shear transfer,
not on blindly adding host DOFs.

Artifacts:

- [coarse XFEM vs structural hysteresis](/c:/MyLibs/fall_n/doc/figures/validation_reboot/xfem_global_secant_vs_structural_n10_lobatto_200mm_hysteresis.png)
- [XFEM refinement matrix](/c:/MyLibs/fall_n/doc/figures/validation_reboot/xfem_global_secant_structural_refinement_matrix_200mm.png)
- [XFEM refinement summary](/c:/MyLibs/fall_n/doc/figures/validation_reboot/xfem_global_secant_structural_refinement_matrix_200mm_summary.json)
- [`ShiftedHeavisideSolidElement` implementation](/c:/MyLibs/fall_n/src/xfem/ShiftedHeavisideSolidElement.hh)
- [`run_xfem_structural_refinement_matrix.py`](/c:/MyLibs/fall_n/scripts/run_xfem_structural_refinement_matrix.py)

### Repository hygiene before the next master push

Generated validation outputs are intentionally not part of the source tree.
`data/output/` is ignored, while promoted evidence should live under
`doc/figures/validation_reboot/` and, when needed by the thesis, under
`PhD_Thesis/Figuras/validation_reboot/`. The cleanup helper
`scripts/cleanup_validation_artifacts.py` now has a safe `--skip-figures`
mode for commit preparation: it prunes disposable output bundles without
touching promoted figures.

The tracked `build_stage8_validation/` build directory has also been removed
from the Git index. Local build trees should stay local; this keeps binary
executables, Ninja logs, and CTest temporary files out of the next commit.

## Bottom Line

`fall_n` is best understood today as an actively hardened research library: ambitious, already technically rich, and increasingly explicit about what is mature, what is experimental, and what still needs scientific or architectural closure. The multiscale subsystem is currently the clearest example of that direction.
