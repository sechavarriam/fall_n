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
- A local stabilization gate exists in `scripts/ci_multiscale_stabilization.ps1`, and the intended CI surface is frozen in `.github/workflows/multiscale-stability.yml`.
- A reproducible predefinitive physical-validation harness exists in `scripts/run_predefinitive_physical_validation.ps1`; it records both the current Case 4 short-run milestone and the current Case 5 frontier honestly.

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
- The distributed MPI micro-solve engine is not implemented yet beyond contracts and communicator ownership.
- The root `CMakeLists.txt` is still monolithic.
- `header_files.hh` still causes avoidable build coupling and PCH invalidation.
- The repository mixes naming styles such as `non_lineal` / `nonlinear` and `Homogenisation` / `Homogenized`.
- The top-level documentation still contains historical drift and LaTeX warning debt.
- Some heavy validation drivers remain expensive to compile and run; on the latest audited cyclic-validation pass, touching `main_table_cyclic_validation.cpp` rebuilt in about `15.3 s`, while touching `TableCyclicValidationFE2.cpp` still took about `457.6 s`.

## Quick Wins

The fastest improvements with a good effort/impact ratio are:

1. Keep shrinking the use of `header_files.hh` and move toward module-local umbrellas.
2. Split `CMakeLists.txt` into smaller module-oriented fragments.
3. Continue breaking up the FE2-heavy cyclic-validation translation units, because the local umbrella/PCH cleanup improved dependency hygiene but did not yet solve the heavy compile frontier.
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
- Case 5 iterated two-way FE2 still aborts at the first cracked point in the short-run matrix with explicit rollback and failed-site reporting; that is documented as a frontier, not hidden as a pass.

## Bottom Line

`fall_n` is best understood today as an actively hardened research library: ambitious, already technically rich, and increasingly explicit about what is mature, what is experimental, and what still needs scientific or architectural closure. The multiscale subsystem is currently the clearest example of that direction.
