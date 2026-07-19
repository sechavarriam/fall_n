# `KOBATHE_*` configuration reference

The Ko–Bathe concrete material and the reduced RC-column continuum
validation driver read a family of `KOBATHE_*` switches. This table is the
single reference for all 97 of them. Unset variables always fall back to
the defaults listed here, which reproduce the committed baseline behavior
bit for bit.

Two kinds of knobs exist and they must not be confused:

- **Compile-time macros** (3): passed as `-DNAME=value` to the compiler.
  They change the material law itself and require a rebuild.
- **Environment variables** (94): read at runtime with `getenv`. The three
  material-level ones are latched per process on first use (a "material
  gene"); changing them mid-run has no effect. The driver-level ones are
  read when the corresponding solver branch starts.

Boolean variables take `1`/`0` unless noted. Numeric variables are parsed
with `atof`/`atoi`.

## Material law — `src/materials/constitutive_models/non_lineal/KoBatheConcrete3D.hh`

These alter physical results silently. Any executable that instantiates
`KoBatheConcrete3D` is affected, including the FE² submodels.

### Compile-time macros

| Macro | Default | Effect |
|---|---|---|
| `KOBATHE_MULTI_CRACK` | `0` | Enables the experimental fixed multi-directional crack model with an angular threshold (de Borst & Nauta 1985) instead of the forced-orthogonality crack system. |
| `KOBATHE_CRACK_ANGLE_DEG` | `30.0` | Angular threshold in degrees for opening a new non-orthogonal crack when `KOBATHE_MULTI_CRACK=1`. |
| `KOBATHE_ANCHOR_FT` | `1` | Clamps the continuity-bridge formation anchor to the tensile strength `f_t` ("fix C"). Build with `0` to reproduce the pre-fix behavior for controlled A/B runs. |

### Environment variables (latched per process)

| Variable | Type | Default | Effect |
|---|---|---|---|
| `KOBATHE_ANCHOR_CAP_RATIO` | double | `1.0` | Scales the ceiling of the formation-anchor clamp to `ratio * f_t`. `1.0` is the calibrated fix; large values progressively restore the original octahedral overshoot (over-stiff envelope). |
| `KOBATHE_CLOSURE_STIFF_FRAC` | double | `1.0` | Fraction of the elastic normal stiffness recovered when a crack closes. Values below 1 damp the reversal spike at the cost of a softer closed crack in compression. The loading branch is unchanged. |
| `KOBATHE_VISCOUS_RATCHET_BETA` | double | `1.0` (clamped to `[0,1]`) | Duvaut–Lions viscous relaxation factor of the crack-strain ratchet. `1.0` reproduces the rate-independent original; smaller values smooth the C0 kinks that create reversal multi-equilibria. |

## Validation driver — `src/validation/ReducedRCColumnContinuumBaseline.cpp`

Read by the targets that compile this translation unit
(`fall_n_reduced_rc_column_continuum_reference_benchmark`, the publication
probes, and `fall_n_reduced_rc_column_continuum_baseline_test`). Each
family is dormant until its gate variable enables the branch.

### Dynamic relaxation (gate `KOBATHE_DYNAMIC=1`)

Newmark + Rayleigh dynamic branch used to cross the ~88 mm yield limit
point that the quasi-static path cannot pass.

| Variable | Type | Default | Effect |
|---|---|---|---|
| `KOBATHE_DYNAMIC` | bool | `0` | Enables the dynamic branch (replaces the quasi-static incremental loop). |
| `KOBATHE_DYN_RHO` | double | `2.4e-3` | Material density used to assemble the mass matrix. |
| `KOBATHE_DYN_TOTAL_TIME` | double | `1.0` | Total physical time `T` mapped onto the loading protocol. |
| `KOBATHE_DYN_DT` | double | `5.0e-5` | Time step. |
| `KOBATHE_DYN_ALPHA_M` | double | `60.0` | Rayleigh mass-proportional damping coefficient. |
| `KOBATHE_DYN_BETA_K` | double | `1.0e-3` | Rayleigh stiffness-proportional damping coefficient. |
| `KOBATHE_DYN_PRELOAD_FRAC` | double | `0.15` (clamped to `[0,0.9]`) | Fraction of `T` spent ramping the axial preload at zero drift. Only active when the spec has axial compression. |
| `KOBATHE_DYN_RELAX` | bool | `0` | Per-increment dynamic-relaxation-to-static mode. The protocol is discretized and each increment is relaxed until the base shear settles, so recorded states are static. |
| `KOBATHE_DYN_RELAX_INCREMENTS` | int | `500` | Number of lateral increments of the discretized protocol. |
| `KOBATHE_DYN_RELAX_PRELOAD_STEPS` | int | `12` | Preload increments (only with axial compression). |
| `KOBATHE_DYN_RELAX_VTOL` | double | `1.0e-3` | Relative settling tolerance on the base-shear change per step. |
| `KOBATHE_DYN_RELAX_FLOOR` | double | `1.0e-6` | Shear floor in MN for the relative criterion (during preload `V ~ 0` and the criterion would never trigger without it). |
| `KOBATHE_DYN_RELAX_MAXSTEPS` | int | `3000` | Maximum relaxation steps per increment. |
| `KOBATHE_DYN_RELAX_MINSTEPS` | int | `8` | Minimum relaxation steps per increment. |
| `KOBATHE_DYN_RELAX_STABLE` | int | `3` | Consecutive settled steps required before accepting the state as static. |
| `KOBATHE_DYN_RELAX_KINETIC` | bool | `1` | Kinetic damping (velocity reset at kinetic-energy peaks) to drain momentum faster. |
| `KOBATHE_DYN_RELAX_TRACE` | int | `0` | Per-step trace of the first relaxation, for calibrating `dt`/damping. |
| `KOBATHE_DYN_VTK_DRIFT_MM` | double | `1.0` | Drift interval in mm between VTK output frames. |
| `KOBATHE_DYN_PREDICT` | bool | `0` | Tangent-extrapolation predictor at protocol reversals, seeding the unloading (lower) branch. |
| `KOBATHE_DYN_PREDICT_GAIN` | double | `1.0` | Gain applied to the predictor extrapolation ratio. |
| `KOBATHE_DYN_UNLOAD_DT` | double | `0.0` | Large time step used during unloading so the implicit step degenerates to a static Newton solve near the seeded branch. `0` disables. |

### Arc-length continuation (gate `KOBATHE_ARCLEN=1`)

Bordered mixed-control Newton with Crisfield constraint, tracing `(u, p)`
across the load limit point.

| Variable | Type | Default | Effect |
|---|---|---|---|
| `KOBATHE_ARCLEN` | bool | `0` | Enables the arc-length branch. |
| `KOBATHE_ARCLEN_PSI` | double | `0.0` | Crisfield load-scaling factor `psi` (`0` = cylindrical constraint). |
| `KOBATHE_ARCLEN_MU` | double | `0.0` | Seed of the per-iteration LM regularization inside the bordered corrector. `>0` activates it. |
| `KOBATHE_ARCLEN_ACCEPT_FLOOR` | double | `0.0` | Residual floor acceptance. A step is accepted if the residual sits at the floor and the load parameter advanced (the softening plateau stalls near `1e-3` because of the penalty terms). |
| `KOBATHE_ARCLEN_DS` | double | `5.0e-3` | Initial arc-length increment. |
| `KOBATHE_ARCLEN_DS_MIN` | double | `1.0e-5` | Minimum arc-length increment. |
| `KOBATHE_ARCLEN_DS_MAX` | double | `5.0e-2` | Maximum arc-length increment. |
| `KOBATHE_ARCLEN_MAXSTEPS` | int | `50000` | Maximum continuation steps. |
| `KOBATHE_ARCLEN_RTOL` | double | `1.0e-4` | Corrector residual tolerance (relaxed to the physical residual floor of the penalized problem). |
| `KOBATHE_ARCLEN_CTOL` | double | `1.0e-8` | Constraint tolerance. |
| `KOBATHE_ARCLEN_MAXIT` | int | `60` | Maximum corrector iterations. |
| `KOBATHE_ARCLEN_MU_MAX` | double | `1.0e-1` | Cap of the adaptive LM regularization (fraction of `\|diag(K)\|`). |
| `KOBATHE_ARCLEN_MU_GROW` | double | `4.0` | LM growth factor on failure. |
| `KOBATHE_ARCLEN_MU_DROP` | double | `0.25` | LM drop factor on success. |
| `KOBATHE_ARCLEN_FD_ORDER` | int | `2` | Finite-difference order of the load column `dR/dlambda`. |
| `KOBATHE_ARCLEN_PRED_ORDER` | int | `2` | Predictor order (`2` adds the curvature/Taylor term to the secant). |
| `KOBATHE_ARCLEN_TURN_AWARE` | bool | `1` | Detects protocol reversals and crosses them with a fixed-control window instead of the secant predictor. |
| `KOBATHE_ARCLEN_TURN_HOLD` | int | `6` | Number of fixed-control steps held after a detected reversal. |
| `KOBATHE_ARCLEN_DIAG` | presence | unset | If set (any value), prints rejected-step diagnostics to stderr. |

### Regularized Newton / Levenberg–Marquardt continuation (gate `KOBATHE_LMNEWTON=1`)

Solves `(K + mu I) du = -R` with adaptive `mu` so the near-singular
tangent at the limit point remains solvable, decaying to plain Newton on
well-conditioned branches.

| Variable | Type | Default | Effect |
|---|---|---|---|
| `KOBATHE_LMNEWTON` | bool | `0` | Enables the LM continuation branch. |
| `KOBATHE_LM_MU0` | double | `1.0e-2` | Initial regularization `mu`. |
| `KOBATHE_LM_GROW` | double | `4.0` | `mu` growth factor when an iteration fails to reduce the residual. |
| `KOBATHE_LM_DROP` | double | `0.3` | `mu` drop factor on success. |
| `KOBATHE_LM_MUMIN` | double | `0.0` | Lower bound of `mu`. |
| `KOBATHE_LM_MAXIT` | int | `80` | Maximum Newton iterations per step. |
| `KOBATHE_LM_ATOL` | double | `1.0e-8` | Absolute residual tolerance. |
| `KOBATHE_LM_RTOL` | double | `1.0e-8` | Relative residual tolerance. |
| `KOBATHE_LM_MUMAXFRAC` | double | `1.0e-1` | Cap of `mu` relative to `\|diag(K)\|` (prevents `mu` explosions that freeze the step at the limit point). |
| `KOBATHE_LM_STAG` | int | `12` | Stagnation iteration budget before giving up the step. |
| `KOBATHE_LM_ACCEPT_FLOOR` | double | `1.0e-5` | Residual floor at which a stagnated step is still accepted. |
| `KOBATHE_LM_PREDICT` | bool | `1` | Secant predictor between steps (kills the reversal spike). `0` falls back to pure warm-start. |
| `KOBATHE_LM_PREDICT_MAXSCALE` | double | `4.0` | Maximum scaling of the secant extrapolation. |
| `KOBATHE_LM_LINESEARCH` | bool | `0` | Incremental-energy line search that rejects basin-jumping LM steps. |
| `KOBATHE_LM_LS_BACKTRACKS` | int | `4` | Maximum line-search backtracks. |
| `KOBATHE_LM_PROXIMAL` | double | `0.0` | Proximal continuation weight `kappa` (phase A minimizes `Pi + kappa/2 \|u-u_n\|^2`, phase B polishes). `0` disables. |
| `KOBATHE_LM_SWITCH_STEP` | int | `0` | Static Newton→LM switch: standard Newton (`mu = 0`) for global steps below this index, LM afterwards. `0` runs LM from the start. |
| `KOBATHE_LM_ADAPTIVE_SWITCH` | bool | `0` | Newton↔LM state machine with per-step checkpoint/restore. When on, the static switch is ignored. |
| `KOBATHE_LM_ASW_CLEAN_M` | int | `3` | Consecutive converged LM steps with degenerate `mu` required to switch back to Newton mode. |
| `KOBATHE_LM_ASW_CLEAN_MU` | double | `1.0e-12` | `mu` threshold under which an LM step counts as "clean Newton". |
| `KOBATHE_LM_ASW_LADDER` | int | `3` | Bisection-ladder depth: a non-converged step is retried sub-incremented (x2, x4, ...) before floor acceptance. |
| `KOBATHE_LM_REVSUBDIV` | int | `1` (min `1`) | Sub-increments applied to the first post-reversal steps so the tangent evolves smoothly through the unloading transition. |
| `KOBATHE_LM_REVSUBN` | int | `3` | Number of post-reversal steps that receive the sub-incrementation. |
| `KOBATHE_LM_DEFLATE` | bool | `0` | Sherman–Morrison deflation with detect–restore–retry at reversal spikes. |
| `KOBATHE_LM_DEFLATE_SPIKE` | double | `1.5` | Spike detector threshold relative to the pre-reversal shear envelope. |
| `KOBATHE_LM_DEFLATE_MAXRETRY` | int | `2` | Maximum deflated retries per step. |
| `KOBATHE_LM_DEFLATE_ON_NOCONV` | bool | `0` | Extends the spike detector to non-converged stalled steps. |
| `KOBATHE_LM_DEFLATE_RESID_FACTOR` | double | `10.0` | A deflated retry is kept only if its residual does not exceed the base attempt by more than this factor. |
| `KOBATHE_LM_DEFLATE_POWER` | double | `2.0` | Deflation operator exponent. |
| `KOBATHE_LM_DEFLATE_SHIFT` | double | `1.0` | Deflation operator shift. |
| `KOBATHE_LM_DEFLATE_TAUMAX` | double | `50.0` | Cap of the deflation factor `tau`. |
| `KOBATHE_LM_REPLAY_WINDOW` | string | unset | Window `"k0:m"` for the CA replay mode. Required (and validated) when `KOBATHE_CA=1`. |

### TAO energy continuation (inside the LM branch)

Trust-region minimization of the incremental energy. Both gates require
the `production_stabilized` concrete profile (the hard crack closure of
`paper_reference` breaks the C1 smoothness the trust region needs).

| Variable | Type | Default | Effect |
|---|---|---|---|
| `KOBATHE_TAO_REVERSAL` | bool | `0` | TAO step only at protocol reversals. |
| `KOBATHE_TAO` | bool | `0` | TAO step on every increment. |
| `KOBATHE_TAO_POLISH` | bool | `0` | LM polish after TAO leaves the iterate in the right basin without closing the residual. |
| `KOBATHE_TAO_MAXIT` | int | `200` | Maximum TAO iterations. |
| `KOBATHE_TAO_GATOL` | double | `1.0e-8` | Absolute gradient tolerance. |
| `KOBATHE_TAO_GRTOL` | double | `1.0e-8` | Relative gradient tolerance. |
| `KOBATHE_TAO_ACCEPT_FLOOR` | double | `1.0e-6` | Gradient floor at which a TAO step is accepted. |
| `KOBATHE_TAO_TYPE` | string | kernel default | Explicit TAO algorithm name. If unset and the symmetry check detects `\|K-K^T\|/\|K\| > 1e-8`, the driver falls back to `bqnls` (gradient-only). |
| `KOBATHE_TAO_SYMCHECK` | bool | `0` | One-shot Hessian symmetry check `\|K-K^T\|_F/\|K\|_F` printed to stderr. |

### Cultural-algorithm window tuner (gate `KOBATHE_CA=1`)

Replays a window of the protocol per candidate to tune the LM trait
vector. Runs the protocol up to `k0-1`, checkpoints model and solver, and
maximizes the window fitness. Replay mode persists its artifacts
(`ca_history.csv`, `ca_best_genome.json`) and does not continue the
protocol. `KOBATHE_LM_DEFLATE` and `KOBATHE_TAO*` are ignored in this
mode.

| Variable | Type | Default | Effect |
|---|---|---|---|
| `KOBATHE_CA` | bool | `0` | Enables the CA replay tuner (requires `KOBATHE_LM_REPLAY_WINDOW="k0:m"`). |
| `KOBATHE_CA_POP` | int | `8` | Population size. |
| `KOBATHE_CA_GEN` | int | `10` | Number of generations. |
| `KOBATHE_CA_SEED` | uint64 | `20260715` | RNG seed. |
| `KOBATHE_CA_TOPFRAC` | double | `0.25` | Acceptance fraction feeding the belief space. |
| `KOBATHE_CA_VTARGET` | double | `0.0` | Fitness v2: penalizes the shear peak relative to this physical plateau (MN) instead of the run's own envelope. `0` keeps fitness v1. |
| `KOBATHE_CA_WZZ` | double | `0.0` | Fitness v2: weight of the zig-zag penalty (fraction of `dV` sign flips on monotone drift segments). |
| `KOBATHE_CA_SWITCH_GENE` | bool | `0` | Adds the 9th gene `g8 = switch_step`. The 8-gene default keeps earlier campaigns bit-for-bit reproducible (same RNG sequence). |
| `KOBATHE_CA_ASW_GENES` | bool | `0` | Adds the 3 genes of the adaptive Newton↔LM machine: `clean_m` in `[1,6]`, `ladder` in `[0,4]`, `log10(clean_mu)` in `[-16,-6]`. Composable with the switch gene. |
| `KOBATHE_CA_SOURCES` | int | `2` | Knowledge sources in the belief space. `2` = normative + situational (historical configuration); `>=5` = full literature set (adds domain, history, topographic). |

## Reproducing a tuned configuration

`ca_best_genome.json` is written with keys equal to the environment
variables that reproduce the best candidate, so a tuned run can be
replayed by exporting that file's entries verbatim.
