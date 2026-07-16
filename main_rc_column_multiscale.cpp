// =============================================================================
//  main_rc_column_multiscale.cpp
// =============================================================================
//
//  Single RC cantilever column — quasi-static cyclic FE² driver.
//
//  Macro scale : 3.2 m vertical cantilever, 8 two-node Timoshenko beam
//                elements with REDUCED (1-point) integration — the pattern
//                validated in tests/test_timoshenko_cantilever_benchmark.cpp
//                (Test 4: 2-node elements + GaussLegendreCellIntegrator<1>;
//                full 2-point integration shear-locks the linear element,
//                adding a parasitic kGA·l²/12 to EI).  RC fiber section from
//                ReducedRCColumnReferenceSpec (0.25 × 0.25 m, f'c = 30 MPa).
//  Micro scale : one persistent KoBathe RVE (Hex27 + 8 embedded rebar bars)
//                coupled at the BASE element (plastic-hinge region).
//
//  Purpose: probe whether macro/RVE information passing (FE²) smooths the
//  noisy hysteresis loops the monolithic continuum baseline shows at cyclic
//  reversals, and compare the section-force homogenisation operator
//  (boundary reactions vs volume average) as the lever.
//
//  Protocol (mirrors the monolithic reduced-column driver defaults):
//    - constant axial preload of 0.02 MN, equilibrated over 4 steps,
//    - imposed lateral tip displacement cycling ±50/100/150/200 mm,
//      0 → +A → −A → 0 per amplitude in linear segments, 8 steps per
//      segment × substep factor 2  →  4 + 4·3·8·2 = 196 targets.
//
//  Output: <out>/fe2_column_hysteresis.csv
//      step,p,drift_m,base_shear_MN,staggered_iters,macro_converged
//
//  Units: [m, MN, MPa = MN/m²]
//
// =============================================================================

#include "header_files.hh"
#include "src/validation/ReducedRCColumnReferenceSpec.hh"
#include "src/validation/ReducedRCColumnSolveControl.hh"

#include <Eigen/Dense>

#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numbers>
#include <print>
#include <span>
#include <string>
#include <vector>

using namespace fall_n;

// =============================================================================
//  Constants
// =============================================================================
namespace {

// ── Column geometry / materials ──────────────────────────────────────────────
//  Identical to ReducedRCColumnReferenceSpec (reference monolithic column).
static constexpr double COL_HEIGHT = 3.2;     // column height [m]
static constexpr double COL_B      = 0.25;    // section width  [m]
static constexpr double COL_H      = 0.25;    // section height [m]
static constexpr double COL_CVR    = 0.03;    // cover [m]
static constexpr double COL_BAR    = 0.016;   // longitudinal bar ∅ [m]
static constexpr double COL_FPC    = 30.0;    // f'c [MPa]
static constexpr double STEEL_E    = 200000.0;
static constexpr double STEEL_FY   = 420.0;
static constexpr double STEEL_B    = 0.01;
static constexpr double NU_RC      = 0.20;

static const double EC_COL = 4700.0 * std::sqrt(COL_FPC);
static const double GC_COL = EC_COL / (2.0 * (1.0 + NU_RC));

// ── Cyclic protocol (monolithic reduced-column driver defaults) ─────────────
static constexpr std::array<double, 4> AMPLITUDES_M{0.050, 0.100, 0.150, 0.200};
static constexpr int    STEPS_PER_SEGMENT      = 8;
static constexpr int    SEGMENT_SUBSTEP_FACTOR = 2;   // monolithic default
static constexpr int    AXIAL_PRELOAD_STEPS    = 4;   // monolithic default
static constexpr double AXIAL_COMPRESSION_MN   = 0.02;
static constexpr int    MAX_BISECTIONS         = 8;

//  12 segments × 8 steps × 2 = 192 lateral targets (+4 preload = 196 total)
static const int LATERAL_STEPS =
    fall_n::cyclic_step_count(AMPLITUDES_M.size(), STEPS_PER_SEGMENT)
    * SEGMENT_SUBSTEP_FACTOR;
static const int TOTAL_STEPS = AXIAL_PRELOAD_STEPS + LATERAL_STEPS;

// ── RVE mesh (base hinge sub-model) ─────────────────────────────────────────
static constexpr int SUB_NX = 2;
static constexpr int SUB_NY = 2;
static constexpr int SUB_NZ = 4;

// ── Macro node / element layout ──────────────────────────────────────────────
//  N_ELEMS two-node elements stacked along z: nodes 0 (base) … N_ELEMS (top).
//  Element 0 (z in [0, H/8] = hinge region) hosts the FE² coupling site.
static constexpr int N_ELEMS = 8;
static constexpr std::size_t BASE_NODE = 0;
static constexpr std::size_t TOP_NODE  = N_ELEMS;
static constexpr std::size_t HINGE_ELEMENT = 0;

// ── Type aliases (table-driver pattern) ──────────────────────────────────────
static constexpr std::size_t NDOF = 6;
using StructPolicy = SingleElementPolicy<StructuralElement>;
using StructModel  = Model<TimoshenkoBeam3D, continuum::SmallStrain, NDOF, StructPolicy>;
using BeamElemT    = BeamElement<TimoshenkoBeam3D, 3, beam::SmallRotation>;
using ShellElemT   = MITC16Shell<>;
using MacroSolver  = NonlinearAnalysis<TimoshenkoBeam3D, continuum::SmallStrain,
                                       NDOF, StructPolicy>;
using MacroBridge  = BeamMacroBridge<StructModel, BeamElemT>;
using ColumnFE2Analysis = MultiscaleAnalysis<
    MacroSolver, MacroBridge, NonlinearSubModelEvolver, SerialExecutor>;

// ── CLI ──────────────────────────────────────────────────────────────────────
enum class CouplingChoice { MacroOnly, OneWay, TwoWay };
enum class HomogChoice { Boundary, Volume };

struct CliOptions {
    std::string output_dir{};
    CouplingChoice coupling{CouplingChoice::TwoWay};
    HomogChoice homog{HomogChoice::Boundary};
    int    max_staggered_iter{6};
    double staggered_tol{0.03};
    double staggered_relax{0.7};
    int    coupling_start_step{1};
    //  Warmup escalonado del two-way: los primeros N pasos corren en
    //  one-way (el RVE trackea y COMITEA la historia sin retroalimentar,
    //  fisurando gradualmente fuera del lazo acoplado) y el two-way arranca
    //  CALIENTE en N+1 — sin el salto frio del primer feedback de un RVE
    //  virgen ni el transitorio de primera fisuracion dentro del lazo.
    //  0 = two-way desde el primer paso.
    int    two_way_warmup_steps{0};
    int    steps_cap{0};   // 0 = full protocol
    //  Two-way stabilisers (MultiscaleAnalysis built-ins).  The default
    //  tangent regularisation mirrors the L-shaped 16-storey campaign
    //  (cutback_retry_blend_spectral) with the framework default ratios
    //  (blend_alpha 0.35, eigen floors 0.25/0.02).
    TwoWayTangentRegularizationMode tangent_reg{
        TwoWayTangentRegularizationMode::BlendSpectral};
    int    macro_cutback_attempts{3};
    double macro_cutback_factor{0.5};
    int    macro_backtrack_attempts{3};
    double macro_backtrack_factor{0.5};
    //  B1: macro-guided interior kinematics of the RVE (element-layer planes)
    bool interior_kinematics{false};
    //  D: local nonlinear engine for the RVE ramp sub-increments
    LocalSolveEngine local_engine{LocalSolveEngine::Snes};
};

[[nodiscard]] TwoWayTangentRegularizationMode
parse_tangent_regularization_mode(std::string_view raw)
{
    std::string value{raw};
    std::ranges::replace(value, '-', '_');
    if (value == "none" || value == "off") {
        return TwoWayTangentRegularizationMode::None;
    }
    if (value == "blend") {
        return TwoWayTangentRegularizationMode::Blend;
    }
    if (value == "spectral_floor" || value == "spectral") {
        return TwoWayTangentRegularizationMode::SpectralFloor;
    }
    if (value == "blend_spectral" || value == "blend_spectral_floor") {
        return TwoWayTangentRegularizationMode::BlendSpectral;
    }
    if (value == "secant_column_floor" || value == "secant_columns" ||
        value == "column_floor") {
        return TwoWayTangentRegularizationMode::SecantColumnFloor;
    }
    throw std::invalid_argument(
        "unknown --tangent-reg: use none, blend, spectral-floor, "
        "blend-spectral, or secant-column-floor");
}

void print_usage(const char* prog)
{
    std::println(stderr,
        "Usage: {} --output-dir DIR [--coupling macro-only|one-way|two-way]\n"
        "          [--homog boundary|volume] [--max-staggered-iter N]\n"
        "          [--staggered-tol x] [--staggered-relax x]\n"
        "          [--coupling-start-step N] [--steps-cap N]\n"
        "          [--tangent-reg none|blend|spectral-floor|blend-spectral|"
        "secant-column-floor]\n"
        "          [--kin-transfer faces|layers] [--local-engine snes|energy-lm]",
        prog);
}

[[nodiscard]] CliOptions parse_cli(int argc, char* argv[])
{
    CliOptions opts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument(arg + " requires a value");
            }
            return argv[++i];
        };

        if (arg == "--output-dir") {
            opts.output_dir = next();
        } else if (arg == "--coupling") {
            const auto value = next();
            if (value == "macro-only")   opts.coupling = CouplingChoice::MacroOnly;
            else if (value == "one-way") opts.coupling = CouplingChoice::OneWay;
            else if (value == "two-way") opts.coupling = CouplingChoice::TwoWay;
            else throw std::invalid_argument("unknown --coupling: " + value);
        } else if (arg == "--homog") {
            const auto value = next();
            if (value == "boundary")    opts.homog = HomogChoice::Boundary;
            else if (value == "volume") opts.homog = HomogChoice::Volume;
            else throw std::invalid_argument("unknown --homog: " + value);
        } else if (arg == "--max-staggered-iter") {
            opts.max_staggered_iter = std::stoi(next());
        } else if (arg == "--staggered-tol") {
            opts.staggered_tol = std::stod(next());
        } else if (arg == "--staggered-relax") {
            opts.staggered_relax = std::stod(next());
        } else if (arg == "--coupling-start-step") {
            opts.coupling_start_step = std::stoi(next());
        } else if (arg == "--two-way-warmup") {
            opts.two_way_warmup_steps = std::stoi(next());
        } else if (arg == "--steps-cap") {
            opts.steps_cap = std::stoi(next());
        } else if (arg == "--tangent-reg") {
            opts.tangent_reg = parse_tangent_regularization_mode(next());
        } else if (arg == "--kin-transfer") {
            const auto value = next();
            if (value == "faces")       opts.interior_kinematics = false;
            else if (value == "layers") opts.interior_kinematics = true;
            else throw std::invalid_argument(
                "unknown --kin-transfer: " + value + " (use faces|layers)");
        } else if (arg == "--local-engine") {
            const auto value = next();
            if (value == "snes") {
                opts.local_engine = LocalSolveEngine::Snes;
            } else if (value == "energy-lm") {
                opts.local_engine = LocalSolveEngine::EnergyLM;
            } else {
                throw std::invalid_argument(
                    "unknown --local-engine: " + value
                    + " (use snes|energy-lm)");
            }
        } else if (arg == "--macro-cutback-attempts") {
            opts.macro_cutback_attempts = std::stoi(next());
        } else if (arg == "--macro-backtrack-attempts") {
            opts.macro_backtrack_attempts = std::stoi(next());
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }

    if (opts.output_dir.empty()) {
        print_usage(argv[0]);
        throw std::invalid_argument("--output-dir is required");
    }
    return opts;
}

[[nodiscard]] const char* to_label(CouplingChoice c) noexcept
{
    switch (c) {
        case CouplingChoice::MacroOnly: return "macro-only";
        case CouplingChoice::OneWay:    return "one-way";
        case CouplingChoice::TwoWay:    return "two-way";
    }
    return "?";
}

[[nodiscard]] const char* to_label(HomogChoice h) noexcept
{
    return h == HomogChoice::Boundary ? "boundary" : "volume";
}

// ── Control path: 4 preload targets, then 192 lateral targets ────────────────
[[nodiscard]] double preload_completion_p() noexcept
{
    return static_cast<double>(AXIAL_PRELOAD_STEPS)
         / static_cast<double>(TOTAL_STEPS);
}

[[nodiscard]] double lateral_progress(double runtime_p) noexcept
{
    const double p0 = preload_completion_p();
    if (runtime_p <= p0) {
        return 0.0;
    }
    return std::clamp((runtime_p - p0) / (1.0 - p0), 0.0, 1.0);
}

[[nodiscard]] double target_drift(double runtime_p) noexcept
{
    return fall_n::cyclic_displacement(
        lateral_progress(runtime_p),
        std::span<const double>{AMPLITUDES_M});
}

// ── Base shear = Σ f_int,x at the base node (structural-baseline mechanism) ─
[[nodiscard]] double extract_base_shear_x(const StructModel& model)
{
    Vec f_int;
    VecDuplicate(model.state_vector(), &f_int);
    VecSet(f_int, 0.0);

    auto& mut_model = const_cast<StructModel&>(model);
    for (auto& elem : mut_model.elements()) {
        elem.compute_internal_forces(model.state_vector(), f_int);
    }
    VecAssemblyBegin(f_int);
    VecAssemblyEnd(f_int);

    PetscScalar val{};
    const PetscInt idx = static_cast<PetscInt>(
        model.get_domain().node(BASE_NODE).dof_index()[0]);
    VecGetValues(f_int, 1, &idx, &val);

    VecDestroy(&f_int);
    return static_cast<double>(val);
}

}  // anonymous namespace


// =============================================================================
//  Main
// =============================================================================
//  All PETSc-owning objects live inside this function so their destructors
//  run BEFORE PetscFinalize() (otherwise VecDestroy fires after MPI teardown).
static int run_column_fe2(const CliOptions& opts)
{
    std::println("========================================================");
    std::println("  fall_n — RC cantilever column, quasi-static cyclic FE²");
    std::println("  coupling={}  homog={}  targets={} (preload {} + lateral {})",
                 to_label(opts.coupling), to_label(opts.homog),
                 TOTAL_STEPS, AXIAL_PRELOAD_STEPS, LATERAL_STEPS);
    std::println("========================================================");

    // ─────────────────────────────────────────────────────────────────────
    //  1. Macro domain: N_ELEMS two-node beams with 1-point (reduced)
    //     integration — benchmark-validated pattern (Test 4).  Full 2-point
    //     integration on the linear element shear-locks the cantilever
    //     (parasitic EI += kGA·l²/12 ≈ 119 MN·m² per 1.6 m element).
    // ─────────────────────────────────────────────────────────────────────
    Domain<3> domain;
    PetscInt tag = 0;

    for (int i = 0; i <= N_ELEMS; ++i) {
        domain.add_node(i, 0.0, 0.0,
                        COL_HEIGHT * static_cast<double>(i) / N_ELEMS);
    }

    for (int e = 0; e < N_ELEMS; ++e) {
        PetscInt conn[2] = {e, e + 1};
        auto& geom = domain.make_element<LagrangeElement3D<2>>(
            GaussLegendreCellIntegrator<1>{}, tag++, conn);
        geom.set_physical_group("Columns");
    }

    domain.assemble_sieve();

    // ─────────────────────────────────────────────────────────────────────
    //  2. Macro fiber-section material (reduced-column reference spec)
    // ─────────────────────────────────────────────────────────────────────
    const auto col_mat =
        validation_reboot::make_default_reduced_rc_column_section_material();

    auto builder = StructuralModelBuilder<
        BeamElemT, ShellElemT,
        TimoshenkoBeam3D, MindlinReissnerShell3D>{};
    builder.set_frame_material("Columns", col_mat);

    StructModel model{domain, builder.build_elements(domain)};

    //  Base clamped (6 DOFs), lateral tip displacement imposed (DOF x).
    model.fix_z(0.0);
    model.constrain_dof(TOP_NODE, 0, 0.0);
    model.setup();

    //  Constant axial preload (compression) at the tip; the control scheme
    //  ramps it during the preload stage and holds it afterwards.
    model.apply_node_force(
        TOP_NODE, 0.0, 0.0, -AXIAL_COMPRESSION_MN, 0.0, 0.0, 0.0);

    std::println("  Macro: {} nodes, {} beam elements, H={} m, section {}x{} m",
                 domain.num_nodes(), model.elements().size(),
                 COL_HEIGHT, COL_B, COL_H);

    //  Reference elastic lateral stiffness from the pristine section tangent
    //  (frozen copy, baseline pattern): k = [H³/(3EI) + H/(kGA)]⁻¹.
    //  Used after the first lateral step to catch macro stiffness defects
    //  (e.g. shear locking or spuriously clamped rotations).
    double k_theory = 0.0;
    {
        const auto* hinge_beam =
            model.elements()[HINGE_ELEMENT].as<BeamElemT>();
        const auto u0 = hinge_beam->local_state_vector(model.state_vector());
        const auto strain0 = hinge_beam->sample_generalized_strain_at_gp(0, u0);
        auto frozen_section = hinge_beam->sections()[0];
        const auto D0 = frozen_section.tangent(strain0);
        const double EI_y = D0(1, 1);
        const double kGA  = D0(4, 4);
        double flexibility = 0.0;
        if (EI_y > 0.0) {
            flexibility += COL_HEIGHT * COL_HEIGHT * COL_HEIGHT / (3.0 * EI_y);
        }
        if (kGA > 0.0) {
            flexibility += COL_HEIGHT / kGA;
        }
        k_theory = flexibility > 0.0 ? 1.0 / flexibility : 0.0;
        std::println("  Section tangent: EI_y = {:.4f} MN·m², kGA = {:.2f} MN"
                     "  →  k_lateral (3EI/H³ + shear) = {:.4f} MN/m",
                     EI_y, kGA, k_theory);
    }

    // ─────────────────────────────────────────────────────────────────────
    //  3. Sub-model at the base (hinge) element — skipped for macro-only
    // ─────────────────────────────────────────────────────────────────────
    MultiscaleCoordinator coordinator;
    std::vector<NonlinearSubModelEvolver> nl_evolvers;

    if (opts.coupling != CouplingChoice::MacroOnly) {
        auto extract_beam_kinematics =
            [&](std::size_t e_idx) -> ElementKinematics {
            const auto& se = model.elements()[e_idx];
            const auto* beam = se.as<BeamElemT>();
            const auto u_e = se.extract_element_dofs(model.state_vector());

            auto kin_A = extract_section_kinematics(*beam, u_e, -1.0);
            auto kin_B = extract_section_kinematics(*beam, u_e, +1.0);
            kin_A.E = EC_COL; kin_A.G = GC_COL; kin_A.nu = NU_RC;
            kin_B.E = EC_COL; kin_B.G = GC_COL; kin_B.nu = NU_RC;

            ElementKinematics ek;
            ek.element_id   = e_idx;
            ek.kin_A        = kin_A;
            ek.kin_B        = kin_B;
            ek.endpoint_A   = beam->geometry().map_local_point(std::array{-1.0});
            ek.endpoint_B   = beam->geometry().map_local_point(std::array{+1.0});
            ek.up_direction = std::array<double, 3>{1.0, 0.0, 0.0};
            return ek;
        };

        coordinator.add_critical_element(extract_beam_kinematics(HINGE_ELEMENT));

        //  8-bar rebar layout matching the fiber section (4 corners + 4 mid-face)
        const double bar_a = std::numbers::pi / 4.0 * COL_BAR * COL_BAR;
        const double y0 = -COL_B / 2.0 + COL_CVR + COL_BAR / 2.0;
        const double y1 =  COL_B / 2.0 - COL_CVR - COL_BAR / 2.0;
        const double z0 = -COL_H / 2.0 + COL_CVR + COL_BAR / 2.0;
        const double z1 =  COL_H / 2.0 - COL_CVR - COL_BAR / 2.0;

        std::vector<SubModelSpec::RebarBar> bars = {
            {y0, z0, bar_a, COL_BAR}, {y1, z0, bar_a, COL_BAR},
            {y0, z1, bar_a, COL_BAR}, {y1, z1, bar_a, COL_BAR},
            {0.0, z0, bar_a, COL_BAR}, {0.0, z1, bar_a, COL_BAR},
            {y0, 0.0, bar_a, COL_BAR}, {y1, 0.0, bar_a, COL_BAR},
        };

        coordinator.build_sub_models(SubModelSpec{
            .section_width  = COL_B,
            .section_height = COL_H,
            .nx = SUB_NX,
            .ny = SUB_NY,
            .nz = SUB_NZ,
            .hex_order = HexOrder::Quadratic,
            .rebar_bars = std::move(bars),
            .rebar_E  = STEEL_E,
            .rebar_fy = STEEL_FY,
            .rebar_b  = STEEL_B,
        });

        {
            const auto rpt = coordinator.report();
            std::println("  RVE:   {} sub-model(s), {} hex/sub, {} nodes/sub",
                         rpt.num_elements, rpt.total_elements, rpt.total_nodes);
        }

        const std::string evol_dir = opts.output_dir + "/sub_models";
        std::filesystem::create_directories(evol_dir);

        for (auto& sub : coordinator.sub_models()) {
            //  vtk_interval = 0 disables all sub-model VTK output.
            nl_evolvers.emplace_back(sub, COL_FPC, evol_dir, 0);
            nl_evolvers.back().set_incremental_params(60, 8);
            nl_evolvers.back().set_penalty_alpha(EC_COL * 10.0);
            nl_evolvers.back().set_snes_params(100, 2.0, 1e-3);
            if (opts.homog == HomogChoice::Volume) {
                nl_evolvers.back().set_homogenization_operator(
                    HomogenizationOperator::VolumeAverage);
            }
            //  B1: macro-guided interior section planes (layers mode).
            nl_evolvers.back().set_interior_kinematics(
                opts.interior_kinematics);
            //  D: local engine for the ramp sub-increments.
            nl_evolvers.back().set_local_solve_engine(opts.local_engine);
        }

        std::println("  RVE:   kin_transfer={}  local_engine={}",
                     opts.interior_kinematics ? "layers" : "faces",
                     opts.local_engine == LocalSolveEngine::EnergyLM
                         ? "energy-lm" : "snes");
    }

    // ─────────────────────────────────────────────────────────────────────
    //  4. MultiscaleModel + MultiscaleAnalysis
    // ─────────────────────────────────────────────────────────────────────
    MultiscaleModel<MacroBridge, NonlinearSubModelEvolver> ms_model{
        MacroBridge{model}};

    for (auto& ev : nl_evolvers) {
        const auto eid = ev.parent_element_id();
        ms_model.register_local_model(
            ms_model.macro_bridge().default_site(eid),
            std::move(ev));
    }

    std::unique_ptr<CouplingAlgorithm> algorithm =
        (opts.coupling == CouplingChoice::TwoWay)
            ? std::unique_ptr<CouplingAlgorithm>(
                  std::make_unique<IteratedTwoWayFE2>(opts.max_staggered_iter))
            : std::unique_ptr<CouplingAlgorithm>(
                  std::make_unique<OneWayDownscaling>());

    MacroSolver nl{&model};

    //  Macro solver configuration aligned with the VALIDATED reduced-column
    //  structural baseline (ReducedRCColumnStructuralBaseline.cpp), which
    //  completed this same cyclic protocol on the same fiber column:
    //    - canonical Newton profile cascade (basic → backtracking → l2 →
    //      trust region) with the small-residual acceptance band,
    //    - cutback-based increment adaptation around the softening peaks.
    nl.set_solve_profiles(
        validation_reboot::make_reduced_rc_validation_solve_profiles(
            validation_reboot::ReducedRCColumnSolverPolicyKind::
                canonical_newton_profile_cascade));
    {
        using Adapt = MacroSolver::IncrementAdaptationSettings;
        const double nominal_increment =
            1.0 / static_cast<double>(TOTAL_STEPS);
        nl.set_increment_adaptation(Adapt{
            .enabled = true,
            .min_increment_size =
                std::ldexp(nominal_increment, -(MAX_BISECTIONS + 3)),
            .max_increment_size = nominal_increment,
            .cutback_factor = 0.5,
            .growth_factor = 1.15,
            .max_cutbacks_per_step = std::max(8, MAX_BISECTIONS * 2),
            .easy_newton_iterations = 6,
            .difficult_newton_iterations = 12,
            .easy_steps_before_growth = 2,
        });
    }

    ColumnFE2Analysis analysis(
        nl,
        std::move(ms_model),
        std::move(algorithm),
        std::make_unique<ForceAndTangentConvergence>(
            opts.staggered_tol, opts.staggered_tol),
        std::make_unique<ConstantRelaxation>(opts.staggered_relax),
        SerialExecutor{});
    analysis.set_coupling_start_step(opts.coupling_start_step);
    analysis.set_section_dimensions(COL_B, COL_H);

    //  When the staggered loop oscillates between the fiber-section and the
    //  RVE operators (typical at cyclic reversals — the very phenomenon this
    //  driver probes), advance the step with the last converged homogenised
    //  feedback instead of aborting the whole protocol.
    if (opts.coupling == CouplingChoice::TwoWay) {
        TwoWayFailureRecoveryPolicy recovery{};
        recovery.mode = TwoWayFailureRecoveryMode::HybridObservationWindow;
        recovery.max_hybrid_steps = 0;   // unlimited observation window
        //  Without this the RVE freezes at the last strict step while the
        //  macro cycles on: every hybrid step then imposes a growing
        //  total-kinematics jump on a stale material state and the staggered
        //  loop can never recover (res_F pinned at ~1). Evolving + committing
        //  the locals under the final macro state keeps the micro history
        //  tracking the protocol and refreshes the affine feedback anchor.
        recovery.evolve_locals_in_hybrid = true;
        //  En reversas con cinemática de capas, la tangente condensada del
        //  RVE muy fisurado puede ser demasiado degradada para el Newton
        //  macro incluso en ventana híbrida: como último recurso el paso se
        //  reintenta con la inyección limpia (one-way de emergencia).
        recovery.clear_feedback_on_hybrid_macro_failure = true;
        analysis.set_two_way_failure_recovery_policy(recovery);

        //  Stabilisers for the injected cracked-RVE operator, so a macro
        //  divergence at a reversal retries with a regularised tangent and
        //  bounded feedback jumps instead of aborting the protocol.
        TwoWayTangentRegularizationSettings tangent_reg{};
        tangent_reg.mode = opts.tangent_reg;
        analysis.set_two_way_tangent_regularization(tangent_reg);

        TwoWayFeedbackLimiterSettings feedback_limiter{};
        feedback_limiter.enabled = true;   // work_gap_limit 0.05, min_alpha 0.01
        analysis.set_two_way_feedback_limiter(feedback_limiter);

        TwoWayForceFeedbackLimiterSettings force_limiter{};
        force_limiter.enabled = true;      // force_gap_limit 0.05
        analysis.set_two_way_force_feedback_limiter(force_limiter);
    }

    //  On macro failure, retry with a reduced increment and with the
    //  predictor blended back toward the last accepted feedback.
    analysis.set_macro_step_cutback(
        opts.macro_cutback_attempts, opts.macro_cutback_factor);
    analysis.set_macro_failure_backtracking(
        opts.macro_backtrack_attempts, opts.macro_backtrack_factor);

    std::println("  FE²:   staggered max_iter={}, tol={}, relax={}, start_step={}",
                 opts.max_staggered_iter, opts.staggered_tol,
                 opts.staggered_relax, opts.coupling_start_step);
    std::println("  FE²:   tangent_reg={} (alpha=0.35, floors 0.25/0.02), "
                 "feedback limiters on (gap 0.05), macro cutback {}x{}, "
                 "macro backtracking {}x{}",
                 to_string(opts.tangent_reg),
                 opts.macro_cutback_attempts, opts.macro_cutback_factor,
                 opts.macro_backtrack_attempts, opts.macro_backtrack_factor);

    //  Seed the RVE at the unloaded state so the first coupled step starts
    //  from the homogenised elastic operator instead of an empty predictor
    //  (same pattern as main_table_multiscale: initialize, then evolve).
    if (opts.coupling != CouplingChoice::MacroOnly) {
        std::println("  Initializing RVE at the unloaded state...");
        if (!analysis.initialize_local_models()) {
            std::println(stderr,
                "[fe2-column] RVE initialization FAILED ({} sub-models)",
                analysis.last_report().failed_submodels);
            PetscFinalize();
            return 1;
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    //  5. Incremental displacement control: preload ramp, then cyclic drift
    // ─────────────────────────────────────────────────────────────────────
    auto scheme = make_control(
        [](double p, Vec f_full, Vec f_ext, StructModel* m) {
            const double p0 = preload_completion_p();
            if (p <= p0) {
                //  Preload stage: ramp the axial force, keep the tip fixed.
                VecCopy(f_full, f_ext);
                VecScale(f_ext, std::clamp(p / p0, 0.0, 1.0));
                m->update_imposed_value(TOP_NODE, 0, 0.0);
                return;
            }
            //  Lateral stage: full axial force + imposed cyclic tip drift.
            VecCopy(f_full, f_ext);
            m->update_imposed_value(TOP_NODE, 0, target_drift(p));
        });

    nl.begin_incremental(TOTAL_STEPS, MAX_BISECTIONS, scheme);

    const int executed_steps =
        (opts.steps_cap > 0) ? std::min(TOTAL_STEPS, opts.steps_cap)
                             : TOTAL_STEPS;

    // ─────────────────────────────────────────────────────────────────────
    //  6. Step loop + hysteresis CSV (flushed row by row)
    // ─────────────────────────────────────────────────────────────────────
    std::filesystem::create_directories(opts.output_dir);
    std::ofstream csv(opts.output_dir + "/fe2_column_hysteresis.csv");
    csv << "step,p,drift_m,base_shear_MN,staggered_iters,macro_converged\n";
    csv << std::scientific << std::setprecision(8);

    bool ok = true;
    bool stiffness_checked = false;
    int completed_steps = 0;
    int nonconverged_rows = 0;
    int strict_two_way_steps = 0;
    int hybrid_window_steps = 0;
    int one_way_steps = 0;
    double abs_shear_max = 0.0;
    for (int step = 1; step <= executed_steps; ++step) {
        //  Baseline drive pattern: aim each macro step at the exact protocol
        //  target so internal cutbacks never desynchronise the 196 targets.
        const double target =
            static_cast<double>(step) / static_cast<double>(TOTAL_STEPS);
        const double delta = target - nl.current_time();
        if (delta <= 1.0e-14) {
            completed_steps = step;
            continue;
        }
        nl.set_increment_size(delta);

        //  Warmup escalonado: conmuta la política de recuperación por paso.
        //  Durante el warmup el modo OneWayOnly hace que el two-way degrade
        //  a one-way (el RVE evoluciona y comitea sin retroalimentar); al
        //  cruzar N vuelve la ventana híbrida y el acople arranca con la
        //  historia local completa y consistente.
        if (opts.coupling == CouplingChoice::TwoWay &&
            opts.two_way_warmup_steps > 0) {
            TwoWayFailureRecoveryPolicy staged{};
            staged.mode = step <= opts.two_way_warmup_steps
                ? TwoWayFailureRecoveryMode::OneWayOnly
                : TwoWayFailureRecoveryMode::HybridObservationWindow;
            staged.max_hybrid_steps = 0;
            staged.evolve_locals_in_hybrid = true;
            staged.clear_feedback_on_hybrid_macro_failure = true;
            analysis.set_two_way_failure_recovery_policy(staged);
        }

        const bool step_ok = analysis.step();
        const auto& report = analysis.last_report();

        if (!step_ok) {
            //  No CSV row for the failed trial (the state was rolled back and
            //  its shear would be stale); report to stderr and abort.
            const double p_fail = report.attempted_state_valid
                ? report.attempted_macro_time
                : target;
            std::println(stderr,
                "[fe2-column] step {} FAILED at p={:.6f} (target drift "
                "{:+.5e} m): reason={} failed_submodels={} rollback={}",
                step, p_fail, target_drift(target),
                to_string(report.termination_reason),
                report.failed_submodels,
                report.rollback_performed ? "yes" : "no");
            ok = false;
            break;
        }

        completed_steps = step;
        const double p = nl.current_time();
        const double drift = model.prescribed_value(TOP_NODE, 0);
        const double shear = extract_base_shear_x(model);
        const int staggered_iters = analysis.last_staggered_iterations();
        if (!report.converged) {
            ++nonconverged_rows;
        }
        switch (report.coupling_regime) {
            case CouplingRegime::StrictTwoWay:            ++strict_two_way_steps; break;
            case CouplingRegime::HybridObservationWindow: ++hybrid_window_steps;  break;
            case CouplingRegime::OneWayOnly:              ++one_way_steps;        break;
        }
        abs_shear_max = std::max(abs_shear_max, std::abs(shear));

        csv << step << ","
            << p << ","
            << drift << ","
            << shear << ","
            << staggered_iters << ","
            << (report.converged ? 1 : 0) << "\n" << std::flush;

        std::println(stderr,
            "[fe2-column] step={:3d}/{}  p={:.6f}  drift={:+.5e} m  "
            "V={:+.5e} MN  stag_iters={}  converged={}  "
            "res_F={:.3e}  res_K={:.3e}  regime={}",
            step, executed_steps, p, drift, shear, staggered_iters,
            report.converged ? 1 : 0,
            report.max_force_residual_rel,
            report.max_tangent_residual_rel,
            to_string(report.coupling_regime));

        //  Elastic-stiffness sanity check on the first lateral step:
        //  the secant k = V/drift must be of the order of 3EI/H³.
        if (!stiffness_checked
            && std::abs(drift) > 1.0e-12 && std::isfinite(shear))
        {
            stiffness_checked = true;
            const double k_measured = std::abs(shear / drift);
            const double ratio =
                k_theory > 0.0 ? k_measured / k_theory
                               : std::numeric_limits<double>::quiet_NaN();
            std::println(
                "  [stiffness-check] first lateral step: k = |V/drift| = "
                "{:.4f} MN/m  vs  3EI/H³(+shear) = {:.4f} MN/m  (ratio {:.2f})",
                k_measured, k_theory, ratio);
            if (!(ratio > 1.0 / 3.0 && ratio < 3.0)) {
                std::println(stderr,
                    "[fe2-column] WARNING: elastic lateral stiffness differs "
                    "from 3EI/H³ by more than 3x (k={:.4f}, k_ref={:.4f}) — "
                    "check element locking / spurious DOF constraints",
                    k_measured, k_theory);
            }
        }
    }
    csv.close();

    for (auto& ev : analysis.model().local_models()) {
        ev.finalize();
    }

    std::println("  Result: {}  ({} of {} protocol steps completed)",
                 ok ? "COMPLETED" : "ABORTED", completed_steps, TOTAL_STEPS);
    std::println("  |V|max: {:.6f} MN   non-converged rows: {}",
                 abs_shear_max, nonconverged_rows);
    std::println("  Regimes: strict_two_way={}  hybrid_window={}  one_way={}",
                 strict_two_way_steps, hybrid_window_steps, one_way_steps);
    std::println("  CSV:    {}/fe2_column_hysteresis.csv", opts.output_dir);

    return ok ? 0 : 1;
}


// =============================================================================
//  Entry point
// =============================================================================
int main(int argc, char* argv[])
{
    setvbuf(stdout, nullptr, _IONBF, 0);

    CliOptions opts;
    try {
        opts = parse_cli(argc, argv);
    } catch (const std::exception& ex) {
        std::println(stderr, "[fe2-column] CLI error: {}", ex.what());
        return 2;
    }

    PetscInitialize(&argc, &argv, nullptr, nullptr);
    PetscOptionsSetValue(nullptr, "-snes_monitor_cancel", "");
    PetscOptionsSetValue(nullptr, "-ksp_monitor_cancel",  "");
    PetscOptionsSetValue(nullptr, "-snes_linesearch_type", "bt");
    PetscOptionsSetValue(nullptr, "-snes_max_it", "100");
    PetscOptionsSetValue(nullptr, "-ksp_type", "preonly");
    PetscOptionsSetValue(nullptr, "-pc_type", "lu");

    const int rc = run_column_fe2(opts);

    PetscFinalize();
    return rc;
}
