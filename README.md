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
- A local stabilization gate exists in `scripts/ci_multiscale_stabilization.ps1`, and the intended CI surface is frozen in `.github/workflows/multiscale-stability.yml`.
- The GitHub Windows/MSYS2 gate now builds under the UCRT64 MSYS shell but runs regression executables from PowerShell with `ucrt64\\bin` injected into `PATH`, which avoids the `exit 127` native-loader failures observed when launching some UCRT executables directly from the MSYS shell.
- A reproducible predefinitive physical-validation harness exists in `scripts/run_predefinitive_physical_validation.ps1`; it records both the current Case 4 short-run milestone and the current Case 5 frontier honestly.
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
ninja -C build fall_n_table_cyclic_validation
```

The repository also contains many executable validation drivers, including:

- `fall_n_lshaped_multiscale`
- `fall_n_lshaped_multiscale_16`
- `fall_n_table_multiscale`
- `fall_n_table_cyclic_validation`
- `fall_n_rc_beam_validation`

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
- The dedicated `fall_n_case5_frontier_probe_test` target is currently affected by a Windows/MSYS2 linker issue involving duplicated Eigen-generated symbols. The main FE2 drivers and the API regressions remain usable, but the cheap probe still needs a target-level link cleanup before it can be treated as a stable CI artifact.
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
- The next concrete validation task is now sharper than before: use the new macro SNES and section-operator diagnostics to determine whether the first cracked `Case 5` FE2 failure is driven mainly by a grossly indefinite injected tangent, by an overly large affine force intercept, or by the interaction between those operators and the macro incremental/bisection solver.
- The optional consistent-tangent override remains an audit path, not a promotion path. The earlier negative result is preserved as historical evidence, but after the FE2 setup-lifetime correction it should be treated as a benchmark that needs re-audit before any final scientific claim is made.

## Bottom Line

`fall_n` is best understood today as an actively hardened research library: ambitious, already technically rich, and increasingly explicit about what is mature, what is experimental, and what still needs scientific or architectural closure. The multiscale subsystem is currently the clearest example of that direction.
