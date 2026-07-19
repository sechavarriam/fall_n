# fall_n

`fall_n` is a C++23 finite-element research library for nonlinear structural
analysis, with a focus on reinforced-concrete structures and beam-to-continuum
multiscale (FE²) coupling. It is developed as part of a doctoral research
project at the Universidad Nacional de Colombia.

## Highlights

- **Elements**: Timoshenko beams with configurable axis quadrature
  (Gauss–Legendre, Gauss–Lobatto, Gauss–Radau), MITC shells, trusses,
  fiber sections, and 3D Lagrangian/serendipity continuum elements
  (hex8/hex20/hex27), with small-strain, total-Lagrangian,
  updated-Lagrangian, and corotational kinematic policies.
- **Constitutive models**: a Ko–Bathe 3D concrete model with octahedral
  invariants, crack band regularization and cyclic crack closure;
  Menegotto–Pinto steel; Kent–Park concrete; damage and plasticity
  compositions with explicit commit/rollback state chains.
- **Solvers**: PETSc-backed nonlinear statics (SNES), implicit dynamics
  (Newmark), arc-length continuation, Levenberg–Marquardt and regularized
  Newton continuation, and TAO-based energy continuation, with a
  Newton↔LM state machine for hard cyclic problems.
- **Multiscale (FE²)**: one-way downscaling, lagged feedback coupling, and
  iterated two-way FE² between structural beam models and 3D continuum
  submodels, including embedded rebar coupling and XFEM-like local
  enrichment for cracks.
- **Metaheuristics**: a cultural algorithm for tuning solver/continuation
  parameter sets over benchmark campaigns.
- **Compile-time audit layer**: the supported combinations of element
  family × kinematic formulation × analysis route are encoded as concepts
  and constexpr catalogs, so unsupported combinations are rejected at
  compile time instead of failing silently at runtime.

The architecture favors C++20/23 concepts over inheritance or CRTP for
polymorphism; CRTP appears only in static mixin classes that inject
operations into strong types.

## Requirements

- CMake ≥ 3.21 and Ninja (or another generator)
- A C++23 compiler (GCC 14+ is the reference toolchain)
- MPI
- [PETSc](https://petsc.org) (with TAO)
- [Eigen3](https://eigen.tuxfamily.org)
- [VTK](https://vtk.org) (IO/XML and filter modules)
- OpenMP (optional, used when available)

## Build

### Windows (MSYS2 / UCRT64)

Install the toolchain and dependencies from an MSYS2 shell:

```bash
pacman -S mingw-w64-ucrt-x86_64-{gcc,cmake,ninja,eigen3,vtk,petsc,msmpi}
```

Configure and build from PowerShell (or any shell with
`C:\msys64\ucrt64\bin` on `PATH`):

```powershell
$env:PATH = "C:\msys64\ucrt64\bin;" + $env:PATH
cmake -S . -B build -G Ninja
cmake --build build
```

The CI workflow in `.github/workflows/multiscale-stability.yml` documents
the exact dependency set and the runtime-DLL staging needed on Windows
runners.

### Linux

With PETSc discoverable through `pkg-config` (set `PETSC_DIR` /
`PETSC_ARCH` if using a custom build):

```bash
cmake -S . -B build -G Ninja
cmake --build build
```

## Tests

The test suite registers 145 CTest cases labeled by cost:

```bash
ctest --test-dir build -L unit                 # fast feedback (seconds)
ctest --test-dir build -L 'unit|integration'   # adds solver-level tests
ctest --test-dir build                         # full suite
```

The heaviest cases (`sensitivity_study`, FE² frontier probes) benefit from
an otherwise idle machine or an increased `--timeout`.

## Getting started

The library is header-first: add the repository root to your include path,
link MPI + PETSc + Eigen + VTK (the `fall_n::multiscale_api` interface
target bundles exactly that), and include the umbrella for the surface you
need, e.g. `src/analysis/MultiscaleAPI.hh` for the FE² coupling layer.

Executable entry points at the repository root show complete workflows,
from small to large:

- `main.cpp` — minimal smoke driver.
- `main_seismic.cpp` — linear-dynamic frame under the El Centro record.
- `main_multiscale.cpp` — basic beam ↔ continuum FE² coupling.
- `main_table_cyclic_validation.cpp` — progressive 5-case cyclic
  validation suite of the FE² pipeline.
- `main_lshaped_multiscale_16storey.cpp` — 16-storey height-irregular
  building under a 3-component seismic record with FE² columns.

Each has a matching CMake target (see `CMakeLists.txt`).

## Repository layout

| Directory | Content |
|---|---|
| `src/algorithms` | metaheuristics (cultural algorithm) |
| `src/analysis` | linear/nonlinear/dynamic analysis, multiscale API, audit catalogs |
| `src/continuum` | kinematic policies, continuum semantics, strain/stress carriers |
| `src/domain`, `src/mesh` | PETSc DMPlex-backed domain and Gmsh import |
| `src/elements` | element formulations (beam, shell, truss, continuum) |
| `src/materials` | constitutive models and update strategies |
| `src/numerics` | quadrature, tensors, linear-algebra adapters |
| `src/petsc` | RAII wrappers over PETSc objects |
| `src/reconstruction` | local submodel evolution and field reconstruction |
| `src/validation` | benchmark baselines and validation campaign drivers |
| `src/xfem` | local enrichment (cracks, cohesive laws) |
| `tests/` | CTest suite |
| `scripts/` | campaign launchers and plotting utilities |
| `data/input` | meshes and ground-motion records used by drivers/tests |

## Configuration

Some validation drivers and the Ko–Bathe concrete path read `KOBATHE_*`
environment variables that select solver policies and physical variants.
The reference table lives in
[docs/kobathe_env_vars.md](docs/kobathe_env_vars.md); unset variables
always fall back to the documented defaults.

## History

The detailed development log that previously lived in this README —
architecture evolution, validation campaigns, and design decisions — is
preserved in [CHANGELOG.md](CHANGELOG.md).

## License

`fall_n` is released under the GNU General Public License v3.0 (see
[LICENSE](LICENSE)). PETSc, Eigen, and VTK are used under their respective
compatible licenses.
