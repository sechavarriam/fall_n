// =============================================================================
//  main_table_cyclic_validation.cpp
//
//  Progressive incremental validation of the FE² pipeline using a
//  cyclic-loaded "4-legged table" geometry.  Five cases of increasing
//  complexity:
//
//    Case 0        — Linear elastic reference (N=3 TimoshenkoBeamN)
//    Case 1a–1i  — Single cantilever beam (N=2..10 node TimoshenkoBeamN)
//    Case 2a/2b/2c — Single continuum column (Hex8/Hex20/Hex27 + rebar)
//    Case 3        — Full table structural model (4 beams + MITC16 shell)
//    Case 4        — Table + FE² sub-models, 1 direction  (TODO)
//    Case 5        — Table + FE² sub-models, 2 directions (TODO)
//
//  All cases use displacement-controlled cyclic loading with geometric
//  amplitudes: ±2.5, ±5, ±10, ±20 mm (triangular protocol).
//
//  Output:
//    data/output/cyclic_validation/{case_name}/hysteresis.csv
//    data/output/cyclic_validation/{case_name}/vtk/
//
//  Build:
//    cmake --build build --target fall_n_table_cyclic_validation
//
//  Run:
//    ./fall_n_table_cyclic_validation --case 1a
//    ./fall_n_table_cyclic_validation --case all
//
//  Units: [m, MN, MPa = MN/m², s]
//
// =============================================================================

#include "header_files.hh"

#include "src/analysis/PenaltyCoupling.hh"
#include "src/elements/TrussElement.hh"

#include <Eigen/Dense>

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <format>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numbers>
#include <print>
#include <string>
#include <vector>

#include <petsc.h>

using namespace fall_n;


// =============================================================================
//  Constants (shared table geometry & material)
// =============================================================================
namespace {

#ifdef FALL_N_SOURCE_DIR
static const std::string BASE = std::string(FALL_N_SOURCE_DIR) + "/";
#else
static const std::string BASE = "./";
#endif

static const std::string OUT_ROOT = BASE + "data/output/cyclic_validation/";

// ── Table geometry ──────────────────────────────────────────────────
static constexpr double LX = 5.0;          // plan X [m]
static constexpr double LY = 5.0;          // plan Y [m]
static constexpr double H  = 3.2;          // column height [m]

// ── Column section 0.25 × 0.25 m ───────────────────────────────────
static constexpr double COL_B   = 0.25;
static constexpr double COL_H   = 0.25;
static constexpr double COL_CVR = 0.03;
static constexpr double COL_BAR = 0.016;
static constexpr double COL_TIE = 0.08;
static constexpr double COL_FPC = 30.0;    // f'c [MPa]

// ── Shell slab ──────────────────────────────────────────────────────
static constexpr double SLAB_T = 0.20;
static const     double SLAB_E = 4700.0 * std::sqrt(COL_FPC);

// ── RC steel ─────────────────────────────────────────────────────────
static constexpr double STEEL_E  = 200000.0;
static constexpr double STEEL_FY = 420.0;
static constexpr double STEEL_B  = 0.01;
static constexpr double TIE_FY   = 420.0;
static constexpr double NU_RC    = 0.20;

// ── Yield & cyclic protocol ──────────────────────────────────────────
static constexpr double EPS_YIELD = STEEL_FY / STEEL_E;   // ≈ 0.0021
static constexpr int    NUM_STEPS = 120;                   // 10 steps per segment × 12
static constexpr int    MAX_BISECT = 6;

// ── Continuum mesh (Cases 2, 4, 5) ──────────────────────────────────
static constexpr int C_NX = 2;
static constexpr int C_NY = 2;
static constexpr int C_NZ = 12;
// ── Effective column elastic moduli ─────────────────────────────
static const double EC_COL = 4700.0 * std::sqrt(COL_FPC);
static const double GC_COL = EC_COL / (2.0 * (1.0 + NU_RC));

// ── Sub-model mesh (Cases 4, 5) ─────────────────────────────────
static constexpr int SUB_NX = 2;
static constexpr int SUB_NY = 2;
static constexpr int SUB_NZ = 1;

// ── FE² coupling parameters ─────────────────────────────────────
static constexpr int    FE2_STEPS           = 60;
static constexpr int    MAX_STAGGERED_ITER  = 3;
static constexpr double STAGGERED_TOL       = 0.05;
static constexpr double STAGGERED_RELAX     = 0.8;
static constexpr int    COUPLING_START_STEP = 1;
// ── Type aliases ─────────────────────────────────────────────────────
static constexpr std::size_t NDOF = 6;
using StructPolicy = SingleElementPolicy<StructuralElement>;
using StructModel  = Model<TimoshenkoBeam3D, continuum::SmallStrain, NDOF, StructPolicy>;
using BeamElemT2   = BeamElement<TimoshenkoBeam3D, 3, beam::SmallRotation>;
using ShellElemT   = MITC16Shell<>;

} // anonymous namespace


// =============================================================================
//  Cyclic displacement protocol  (from CyclicProtocol.hh)
// =============================================================================

using fall_n::cyclic_displacement;


// =============================================================================
//  Step record + CSV writer
// =============================================================================

struct StepRecord {
    int    step;
    double p;
    double drift;        // lateral displacement [m]
    double base_shear;   // reaction force in X [MN]
};

static void write_csv(const std::string& path,
                      const std::vector<StepRecord>& records)
{
    std::ofstream ofs(path);
    ofs << "step,p,drift_m,base_shear_MN\n";
    ofs << std::scientific << std::setprecision(8);
    for (const auto& r : records) {
        ofs << r.step << ","
            << r.p << ","
            << r.drift << ","
            << r.base_shear << "\n";
    }
    std::println("  CSV: {} ({} records)", path, records.size());
}


// =============================================================================
//  Reaction extraction: base shear in X at specified nodes
// =============================================================================

template <typename ModelT>
double extract_base_shear_x(const ModelT& model,
                            const std::vector<std::size_t>& base_nodes)
{
    Vec f_int;
    VecDuplicate(model.state_vector(), &f_int);
    VecSet(f_int, 0.0);

    auto& mut_model = const_cast<ModelT&>(model);
    for (auto& elem : mut_model.elements())
        elem.compute_internal_forces(model.state_vector(), f_int);
    VecAssemblyBegin(f_int);
    VecAssemblyEnd(f_int);

    double shear = 0.0;
    for (auto nid : base_nodes) {
        PetscScalar val;
        PetscInt idx = static_cast<PetscInt>(
            model.get_domain().node(nid).dof_index()[0]);
        VecGetValues(f_int, 1, &idx, &val);
        shear += val;
    }

    VecDestroy(&f_int);
    return shear;
}


// =============================================================================
//  Case 0: Linear elastic reference — TimoshenkoBeamN<3>
// =============================================================================
//
//  Same geometry as Case 1b (N=3 quadratic beam) but with a purely elastic
//  section (E, A, I) — no fiber integration, no material nonlinearity.
//  Provides an elastic backbone for overlay comparison.

static std::vector<StepRecord> run_case0(const std::string& out_dir)
{
    std::println("\n  Case 0: Linear elastic reference (N=3 beam)");

    constexpr std::size_t N = 3;
    using BeamPolicy = SingleElementPolicy<TimoshenkoBeamN<N>>;
    using BeamModel  = Model<TimoshenkoBeam3D, continuum::SmallStrain, NDOF, BeamPolicy>;

    // ── Domain: 3 nodes along column axis (Z) ────────────────────
    Domain<3> domain;
    PetscInt tag = 0;

    for (std::size_t i = 0; i < N; ++i) {
        double z = H * static_cast<double>(i) / static_cast<double>(N - 1);
        domain.add_node(static_cast<PetscInt>(i), 0.0, 0.0, z);
    }

    PetscInt conn[N];
    for (std::size_t i = 0; i < N; ++i)
        conn[i] = static_cast<PetscInt>(i);

    auto& geom = domain.template make_element<LagrangeElement3D<N>>(
        GaussLegendreCellIntegrator<N - 1>{}, tag++, conn);
    geom.set_physical_group("Column");

    domain.assemble_sieve();

    // ── Elastic section: E, G, A, Iy, Iz, J ─────────────────────
    const double Ec  = 4700.0 * std::sqrt(COL_FPC);
    const double Gc  = Ec / (2.0 * (1.0 + NU_RC));
    const double A   = COL_B * COL_H;
    const double Iy  = COL_B * std::pow(COL_H, 3) / 12.0;
    const double Iz  = COL_H * std::pow(COL_B, 3) / 12.0;
    const double J   = 0.1406 * COL_B * std::pow(COL_H, 3);
    const double k   = 5.0 / 6.0;

    TimoshenkoBeamMaterial3D rel{Ec, Gc, A, Iy, Iz, J, k, k};
    auto col_mat = Material<TimoshenkoBeam3D>{rel, ElasticUpdate{}};

    // ── Build element + model ─────────────────────────────────────
    TimoshenkoBeamN<N> beam_elem{&geom, col_mat};
    std::vector<TimoshenkoBeamN<N>> elems;
    elems.push_back(std::move(beam_elem));

    BeamModel model{domain, std::move(elems)};
    model.constrain_node(0, {0, 0, 0, 0, 0, 0});

    const std::size_t top_node = N - 1;
    model.constrain_dof(top_node, 0, 0.0);
    model.setup();

    // ── Solver ────────────────────────────────────────────────────
    PetscOptionsSetValue(nullptr, "-snes_linesearch_type", "bt");
    PetscOptionsSetValue(nullptr, "-snes_max_it", "100");
    PetscOptionsSetValue(nullptr, "-ksp_type",  "preonly");
    PetscOptionsSetValue(nullptr, "-pc_type",   "lu");

    NonlinearAnalysis<TimoshenkoBeam3D, continuum::SmallStrain, NDOF,
                      BeamPolicy> nl{&model};

    // ── Recording ─────────────────────────────────────────────────
    std::vector<StepRecord> records;
    records.reserve(NUM_STEPS + 1);
    records.push_back({0, 0.0, 0.0, 0.0});

    const std::vector<std::size_t> base_nodes = {0};

    nl.set_step_callback(
        [&](int step, double p, const BeamModel& m) {
            double d     = cyclic_displacement(p);
            double shear = extract_base_shear_x(m, base_nodes);
            records.push_back({step, p, d, shear});

            if (step % 20 == 0 || step == NUM_STEPS) {
                std::println("    step={:3d}  p={:.4f}  d={:+.4e} m  V={:+.4e} MN",
                             step, p, d, shear);
            }
        });

    auto scheme = make_control(
        [top_node](double p, Vec /*f_full*/, Vec f_ext, BeamModel* m) {
            VecSet(f_ext, 0.0);
            m->update_imposed_value(top_node, 0, cyclic_displacement(p));
        });

    bool ok = nl.solve_incremental(NUM_STEPS, MAX_BISECT, scheme);

    std::println("  Result: {} ({} records)",
                 ok ? "COMPLETED" : "ABORTED", records.size());

    write_csv(out_dir + "/hysteresis.csv", records);
    return records;
}


// =============================================================================
//  Case 1: Single cantilever beam — TimoshenkoBeamN<N>
// =============================================================================
//
//  Single vertical column from z=0 (clamped) to z=H (displacement-
//  controlled in X).  RC fiber section with KentPark concrete and
//  MenegottoPinto steel.
//
//  N = 2 (linear, 1 GP), 3 (quadratic, 2 GPs), 4 (cubic, 3 GPs).

template <std::size_t N>
std::vector<StepRecord> run_case1(const std::string& out_dir)
{
    static_assert(N >= 2 && N <= 10, "TimoshenkoBeamN supports N=2..10");
    const char sub = static_cast<char>('a' + (N - 2));   // N=2→a, N=3→b, … N=10→i
    std::println("\n  Case 1{}: TimoshenkoBeamN<{}>  ({}-node beam, {} GPs)",
                 sub, N, N, N - 1);

    // ── Type aliases for direct TimoshenkoBeamN (no StructuralElement wrapper)
    using BeamPolicy = SingleElementPolicy<TimoshenkoBeamN<N>>;
    using BeamModel  = Model<TimoshenkoBeam3D, continuum::SmallStrain, NDOF, BeamPolicy>;

    // ── Domain: N nodes along column axis (Z) ────────────────────
    Domain<3> domain;
    PetscInt tag = 0;

    for (std::size_t i = 0; i < N; ++i) {
        double z = H * static_cast<double>(i) / static_cast<double>(N - 1);
        domain.add_node(static_cast<PetscInt>(i), 0.0, 0.0, z);
    }

    PetscInt conn[N];
    for (std::size_t i = 0; i < N; ++i)
        conn[i] = static_cast<PetscInt>(i);

    auto& geom = domain.template make_element<LagrangeElement3D<N>>(
        GaussLegendreCellIntegrator<N - 1>{}, tag++, conn);
    geom.set_physical_group("Column");

    domain.assemble_sieve();

    // ── RC fiber section ──────────────────────────────────────────
    const auto col_mat = make_rc_column_section({
        .b = COL_B, .h = COL_H, .cover = COL_CVR,
        .bar_diameter = COL_BAR, .tie_spacing = COL_TIE,
        .fpc = COL_FPC, .nu = NU_RC,
        .steel_E = STEEL_E, .steel_fy = STEEL_FY, .steel_b = STEEL_B,
        .tie_fy = TIE_FY,
    });

    // ── Build element + model ─────────────────────────────────────
    TimoshenkoBeamN<N> beam_elem{&geom, col_mat};

    std::vector<TimoshenkoBeamN<N>> elems;
    elems.push_back(std::move(beam_elem));

    BeamModel model{domain, std::move(elems)};

    // BCs: clamp base node (node 0)
    model.constrain_node(0, {0, 0, 0, 0, 0, 0});

    // Constrain top node DOF 0 (u_x) for cyclic displacement control
    const std::size_t top_node = N - 1;
    model.constrain_dof(top_node, 0, 0.0);

    model.setup();

    // ── Solver ────────────────────────────────────────────────────
    PetscOptionsSetValue(nullptr, "-snes_linesearch_type", "bt");
    PetscOptionsSetValue(nullptr, "-snes_max_it", "100");
    PetscOptionsSetValue(nullptr, "-ksp_type",  "preonly");
    PetscOptionsSetValue(nullptr, "-pc_type",   "lu");

    NonlinearAnalysis<TimoshenkoBeam3D, continuum::SmallStrain, NDOF,
                      BeamPolicy> nl{&model};

    // ── Recording ─────────────────────────────────────────────────
    std::vector<StepRecord> records;
    records.reserve(NUM_STEPS + 1);
    records.push_back({0, 0.0, 0.0, 0.0});

    const std::vector<std::size_t> base_nodes = {0};

    nl.set_step_callback(
        [&](int step, double p, const BeamModel& m) {
            double d     = cyclic_displacement(p);
            double shear = extract_base_shear_x(m, base_nodes);
            records.push_back({step, p, d, shear});

            if (step % 20 == 0 || step == NUM_STEPS) {
                std::println("    step={:3d}  p={:.4f}  d={:+.4e} m  V={:+.4e} MN",
                             step, p, d, shear);
            }
        });

    // ── Cyclic control scheme ─────────────────────────────────────
    auto scheme = make_control(
        [top_node](double p, Vec /*f_full*/, Vec f_ext, BeamModel* m) {
            VecSet(f_ext, 0.0);
            m->update_imposed_value(top_node, 0, cyclic_displacement(p));
        });

    bool ok = nl.solve_incremental(NUM_STEPS, MAX_BISECT, scheme);

    std::println("  Result: {} ({} records)",
                 ok ? "COMPLETED" : "ABORTED", records.size());

    // ── VTK (skipped for direct TimoshenkoBeamN — needs StructuralElement model)
    // TODO: add VTK export once StructuralElement type-erasure bridge is fixed

    write_csv(out_dir + "/hysteresis.csv", records);
    return records;
}


// =============================================================================
//  Case 2: Single continuum column — Hex + embedded truss rebar
// =============================================================================
//
//  Prismatic column (COL_B × COL_H × H) discretised with hexahedral
//  elements + embedded 1D rebar trusses.  KoBathe concrete + Menegotto-
//  Pinto steel.  Bottom face clamped, top face u_x controlled.
//
//  2a = Hex8 (linear), 2b = Hex20 (serendipity), 2c = Hex27 (quadratic).
//
//  Uses PenaltyCoupling to enforce displacement compatibility between
//  rebar nodes and host hex elements (α = 10·Ec).

static std::vector<StepRecord>
run_case2(HexOrder order, const std::string& out_dir)
{
    const char* label = (order == HexOrder::Linear)     ? "Hex8"  :
                        (order == HexOrder::Serendipity) ? "Hex20" : "Hex27";
    const char sub = (order == HexOrder::Linear)     ? 'a' :
                     (order == HexOrder::Serendipity) ? 'b' : 'c';

    std::println("\n  Case 2{}: {} reinforced column  ({}×{}×{} mesh)",
                 sub, label, C_NX, C_NY, C_NZ);

    // ── Rebar layout: 8 bars (same as FE² sub-model) ─────────────
    const double cvr   = COL_CVR;
    const double bar_d = COL_BAR;
    const double bar_a = std::numbers::pi / 4.0 * bar_d * bar_d;
    const double y0    = -COL_B / 2.0 + cvr + bar_d / 2.0;
    const double y1    =  COL_B / 2.0 - cvr - bar_d / 2.0;
    const double z0    = -COL_H / 2.0 + cvr + bar_d / 2.0;
    const double z1    =  COL_H / 2.0 - cvr - bar_d / 2.0;

    RebarSpec rebar_spec;
    rebar_spec.bars = {
        {y0, z0, bar_a, bar_d}, {y1, z0, bar_a, bar_d},
        {y0, z1, bar_a, bar_d}, {y1, z1, bar_a, bar_d},
        {0.0, z0, bar_a, bar_d}, {0.0, z1, bar_a, bar_d},
        {y0, 0.0, bar_a, bar_d}, {y1, 0.0, bar_a, bar_d},
    };

    // ── Reinforced prismatic domain (hex + rebar) ────────────────
    PrismaticSpec spec{
        .width  = COL_B,
        .height = COL_H,
        .length = H,
        .nx = C_NX, .ny = C_NY, .nz = C_NZ,
        .hex_order = order,
    };

    auto result = make_reinforced_prismatic_domain(spec, rebar_spec);
    auto& domain = result.domain;
    auto& grid   = result.grid;

    // ── Build mixed element vector (hex + truss) ──────────────────
    KoBatheConcrete3D concrete_impl{COL_FPC};
    concrete_impl.set_consistent_tangent(true);
    InelasticMaterial<KoBatheConcrete3D> concrete_inst{std::move(concrete_impl)};
    Material<ThreeDimensionalMaterial> concrete_mat{concrete_inst, InelasticUpdate{}};

    MenegottoPintoSteel steel_impl{STEEL_E, STEEL_FY, STEEL_B};
    InelasticMaterial<MenegottoPintoSteel> steel_inst{std::move(steel_impl)};
    Material<UniaxialMaterial> steel_mat{std::move(steel_inst), InelasticUpdate{}};

    using HexElem = ContinuumElement<ThreeDimensionalMaterial, 3,
                                     continuum::SmallStrain>;
    using MixedModel = Model<ThreeDimensionalMaterial, continuum::SmallStrain,
                             3, MultiElementPolicy>;

    std::vector<FEM_Element> elements;
    elements.reserve(domain.num_elements());

    // Hex elements (first N elements, before rebar range)
    for (std::size_t i = 0; i < result.rebar_range.first; ++i)
        elements.emplace_back(HexElem{&domain.element(i), concrete_mat});

    // Truss rebar elements
    const std::size_t num_bars = rebar_spec.bars.size();
    const std::size_t nz = static_cast<std::size_t>(grid.nz);
    for (std::size_t i = result.rebar_range.first; i < result.rebar_range.last; ++i) {
        std::size_t bar_idx = (i - result.rebar_range.first) / nz;
        double area = rebar_spec.bars[bar_idx].area;
        elements.emplace_back(TrussElement<3>{&domain.element(i), steel_mat, area});
    }

    MixedModel model{domain, std::move(elements)};

    // ── BCs ───────────────────────────────────────────────────────
    // Bottom face: fully clamped (hex + rebar nodes at z=0)
    // Note: nodes_on_face() returns ALL grid-level nodes including
    // mid-face-center points that don't exist in serendipity (Hex20)
    // elements.  Skip nodes with num_dof()==0 to avoid PETSc errors.
    auto face_min = grid.nodes_on_face(PrismFace::MinZ);
    for (auto nid : face_min) {
        if (domain.node(static_cast<std::size_t>(nid)).num_dof() == 0) continue;
        model.constrain_node(static_cast<std::size_t>(nid), {0.0, 0.0, 0.0});
    }

    // Rebar nodes at MinZ face: also clamped
    {
        const int step = grid.step;
        const std::size_t rpb = static_cast<std::size_t>(step * grid.nz + 1);
        for (std::size_t b = 0; b < num_bars; ++b) {
            // First rebar node of each bar (iz=0)
            auto rnid = static_cast<std::size_t>(
                result.embeddings[b * rpb].rebar_node_id);
            model.constrain_node(rnid, {0.0, 0.0, 0.0});
            // Last rebar node of each bar (iz=rpb-1, MaxZ face)
            // Constrain all 3 DOFs: u_x will be driven, u_y=u_z=0.
            // (Penalty coupling skips boundary faces, so transverse
            //  rebar DOFs would otherwise have zero stiffness.)
            auto rnid_top = static_cast<std::size_t>(
                result.embeddings[b * rpb + rpb - 1].rebar_node_id);
            model.constrain_node(rnid_top, {0.0, 0.0, 0.0});
        }
    }

    // Top face: constrain u_x for cyclic control (u_y, u_z free)
    auto face_max = grid.nodes_on_face(PrismFace::MaxZ);
    for (auto nid : face_max) {
        if (domain.node(static_cast<std::size_t>(nid)).num_dof() == 0) continue;
        model.constrain_dof(static_cast<std::size_t>(nid), 0, 0.0);
    }

    model.setup();

    // ── Penalty coupling ──────────────────────────────────────────
    fall_n::PenaltyCoupling coupling;
    coupling.setup(domain, grid, result.embeddings, num_bars,
                   EC_COL * 10.0, /*skip_minz_maxz=*/true, order);

    std::println("  Elements: {} hex + {} truss  ({} nodes)",
                 result.rebar_range.first,
                 result.rebar_range.last - result.rebar_range.first,
                 domain.num_nodes());

    // ── Solver ────────────────────────────────────────────────────
    PetscOptionsSetValue(nullptr, "-snes_linesearch_type", "bt");
    PetscOptionsSetValue(nullptr, "-snes_max_it", "100");
    PetscOptionsSetValue(nullptr, "-ksp_type",  "preonly");
    PetscOptionsSetValue(nullptr, "-pc_type",   "lu");

    NonlinearAnalysis<ThreeDimensionalMaterial, continuum::SmallStrain, 3,
                      MultiElementPolicy> nl{&model};

    nl.set_residual_hook([&coupling](Vec u, Vec f, DM dm) {
        coupling.add_to_residual(u, f, dm);
    });
    nl.set_jacobian_hook([&coupling](Vec u, Mat J, DM dm) {
        coupling.add_to_jacobian(u, J, dm);
    });

    // ── Recording ─────────────────────────────────────────────────
    std::vector<StepRecord> records;
    records.reserve(NUM_STEPS + 1);
    records.push_back({0, 0.0, 0.0, 0.0});

    std::vector<std::size_t> base_nodes;
    base_nodes.reserve(face_min.size());
    for (auto nid : face_min) {
        if (domain.node(static_cast<std::size_t>(nid)).num_dof() == 0) continue;
        base_nodes.push_back(static_cast<std::size_t>(nid));
    }

    // ── PVD time-series setup ────────────────────────────────────
    std::filesystem::create_directories(out_dir + "/vtk");
    PVDWriter pvd_mesh{out_dir + "/vtk/mesh"};
    PVDWriter pvd_gauss{out_dir + "/vtk/gauss"};

    nl.set_step_callback(
        [&](int step, double p, const MixedModel& m) {
            double d     = cyclic_displacement(p);
            double shear = extract_base_shear_x(m, base_nodes);
            records.push_back({step, p, d, shear});

            // Per-step VTK snapshot
            {
                fall_n::vtk::VTKModelExporter exporter{
                    const_cast<MixedModel&>(m)};
                exporter.set_displacement();
                exporter.compute_material_fields();

                auto mesh_file  = std::format("{}/vtk/mesh_{:04d}.vtu",
                                              out_dir, step);
                auto gauss_file = std::format("{}/vtk/gauss_{:04d}.vtu",
                                              out_dir, step);
                exporter.write_mesh(mesh_file);
                exporter.write_gauss_points(gauss_file);
                pvd_mesh.add_timestep(p, mesh_file);
                pvd_gauss.add_timestep(p, gauss_file);
            }

            if (step % 20 == 0 || step == NUM_STEPS) {
                std::println("    step={:3d}  p={:.4f}  d={:+.4e} m  V={:+.4e} MN",
                             step, p, d, shear);
            }
        });

    // ── Cyclic control scheme ─────────────────────────────────────
    std::vector<std::size_t> top_nodes;
    top_nodes.reserve(face_max.size());
    for (auto nid : face_max) {
        if (domain.node(static_cast<std::size_t>(nid)).num_dof() == 0) continue;
        top_nodes.push_back(static_cast<std::size_t>(nid));
    }

    // Also include rebar nodes at MaxZ face in control set
    {
        const int step = grid.step;
        const std::size_t rpb = static_cast<std::size_t>(step * grid.nz + 1);
        for (std::size_t b = 0; b < num_bars; ++b) {
            auto rnid = static_cast<std::size_t>(
                result.embeddings[b * rpb + rpb - 1].rebar_node_id);
            top_nodes.push_back(rnid);
        }
    }

    auto scheme = make_control(
        [&top_nodes](double p, Vec /*f_full*/, Vec f_ext, MixedModel* m) {
            VecSet(f_ext, 0.0);
            double d = cyclic_displacement(p);
            for (auto nid : top_nodes)
                m->update_imposed_value(nid, 0, d);
        });

    bool ok = nl.solve_incremental(NUM_STEPS, MAX_BISECT, scheme);

    std::println("  Result: {} ({} records)",
                 ok ? "COMPLETED" : "ABORTED", records.size());

    // ── Finalize PVD time series ─────────────────────────────────
    pvd_mesh.write();
    pvd_gauss.write();

    write_csv(out_dir + "/hysteresis.csv", records);
    return records;
}


// =============================================================================
//  Case 3: Full table structural model (4 beams + MITC16 shell)
// =============================================================================
//
//  4 vertical RC columns at corners of a 5×5 m plan, height 3.2 m.
//  1 bicubic MITC16 shell slab at z = H.
//  Same domain as main_table_multiscale.cpp.
//
//  Base nodes (z=0) clamped.  Slab corner nodes (4, 7, 16, 19)
//  displacement-controlled in X for a uniform drift pushover.

static std::vector<StepRecord> run_case3(const std::string& out_dir)
{
    std::println("\n  Case 3: Full table (4 × BeamElement + 1 × MITC16)");

    // ── Build table domain (same topology as main_table_multiscale) ──
    Domain<3> domain;
    PetscInt tag = 0;

    // Base nodes: 0–3
    domain.add_node(0, 0.0, 0.0, 0.0);
    domain.add_node(1,  LX, 0.0, 0.0);
    domain.add_node(2, 0.0,  LY, 0.0);
    domain.add_node(3,  LX,  LY, 0.0);

    // Slab nodes: 4–19 (4×4 grid at z = H)
    {
        PetscInt nid = 4;
        for (int jy = 0; jy < 4; ++jy)
            for (int jx = 0; jx < 4; ++jx)
                domain.add_node(nid++,
                                LX * jx / 3.0,
                                LY * jy / 3.0,
                                H);
    }

    // Column elements: 2-node beams
    const std::array<std::pair<PetscInt, PetscInt>, 4> col_pairs = {{
        {0, 4}, {1, 7}, {2, 16}, {3, 19}
    }};
    for (auto [base, top] : col_pairs) {
        PetscInt conn[2] = {base, top};
        auto& g = domain.make_element<LagrangeElement3D<2>>(
            GaussLegendreCellIntegrator<2>{}, tag++, conn);
        g.set_physical_group("Columns");
    }

    // MITC16 slab
    {
        PetscInt slab_conn[16];
        for (int i = 0; i < 16; ++i) slab_conn[i] = 4 + i;
        auto& g = domain.make_element<LagrangeElement3D<4, 4>>(
            GaussLegendreCellIntegrator<4, 4>{}, tag++, slab_conn);
        g.set_physical_group("Slabs");
    }

    domain.assemble_sieve();

    // ── RC fiber section for columns ──────────────────────────────
    const auto col_mat = make_rc_column_section({
        .b = COL_B, .h = COL_H, .cover = COL_CVR,
        .bar_diameter = COL_BAR, .tie_spacing = COL_TIE,
        .fpc = COL_FPC, .nu = NU_RC,
        .steel_E = STEEL_E, .steel_fy = STEEL_FY, .steel_b = STEEL_B,
        .tie_fy = TIE_FY,
    });

    // ── Elastic shell material for slab ───────────────────────────
    MindlinShellMaterial slab_relation{SLAB_E, NU_RC, SLAB_T};
    const auto slab_mat = Material<MindlinReissnerShell3D>{
        slab_relation, ElasticUpdate{}};

    // ── Build structural model ────────────────────────────────────
    auto builder = StructuralModelBuilder<
        BeamElemT2, ShellElemT,
        TimoshenkoBeam3D, MindlinReissnerShell3D>{};
    builder.set_frame_material("Columns", col_mat);
    builder.set_shell_material("Slabs",   slab_mat);

    auto elements = builder.build_elements(domain);

    StructModel model{domain, std::move(elements)};

    // BCs: clamp all base nodes (z = 0)
    model.fix_z(0.0);

    // Constrain slab corner nodes DOF 0 (u_x) for cyclic control
    const std::array<std::size_t, 4> slab_corners = {4, 7, 16, 19};
    for (auto nid : slab_corners)
        model.constrain_dof(nid, 0, 0.0);

    model.setup();

    std::println("  Nodes: {}  Elements: {}  (4 beams + 1 shell)",
                 domain.num_nodes(), model.elements().size());

    // ── Solver ────────────────────────────────────────────────────
    PetscOptionsSetValue(nullptr, "-snes_linesearch_type", "bt");
    PetscOptionsSetValue(nullptr, "-snes_max_it", "100");
    PetscOptionsSetValue(nullptr, "-ksp_type",  "preonly");
    PetscOptionsSetValue(nullptr, "-pc_type",   "lu");

    NonlinearAnalysis<TimoshenkoBeam3D, continuum::SmallStrain, NDOF,
                      StructPolicy> nl{&model};

    // ── Recording ─────────────────────────────────────────────────
    std::vector<StepRecord> records;
    records.reserve(NUM_STEPS + 1);
    records.push_back({0, 0.0, 0.0, 0.0});

    const std::vector<std::size_t> base_nodes = {0, 1, 2, 3};

    // ── PVD time-series setup ────────────────────────────────────
    std::filesystem::create_directories(out_dir + "/vtk");
    PVDWriter pvd_table{out_dir + "/vtk/table"};

    auto beam_profile  = fall_n::reconstruction::RectangularSectionProfile<2>{COL_B, COL_H};
    auto shell_profile = fall_n::reconstruction::ShellThicknessProfile<3>{};

    nl.set_step_callback(
        [&](int step, double p, const StructModel& m) {
            double d     = cyclic_displacement(p);
            double shear = extract_base_shear_x(m, base_nodes);
            records.push_back({step, p, d, shear});

            // Per-step VTM snapshot
            {
                fall_n::vtk::StructuralVTMExporter vtm{
                    const_cast<StructModel&>(m),
                    beam_profile, shell_profile};
                vtm.set_displacement(m.state_vector());
                vtm.set_yield_strain(EPS_YIELD);
                auto vtm_file = std::format("{}/vtk/table_{:04d}.vtm",
                                            out_dir, step);
                vtm.write(vtm_file);
                pvd_table.add_timestep(p, vtm_file);
            }

            if (step % 20 == 0 || step == NUM_STEPS) {
                std::println("    step={:3d}  p={:.4f}  d={:+.4e} m  V={:+.4e} MN",
                             step, p, d, shear);
            }
        });

    // ── Cyclic control scheme ─────────────────────────────────────
    auto scheme = make_control(
        [&slab_corners](double p, Vec /*f_full*/, Vec f_ext, StructModel* m) {
            VecSet(f_ext, 0.0);
            double d = cyclic_displacement(p);
            for (auto nid : slab_corners)
                m->update_imposed_value(nid, 0, d);
        });

    bool ok = nl.solve_incremental(NUM_STEPS, MAX_BISECT, scheme);

    std::println("  Result: {} ({} records)",
                 ok ? "COMPLETED" : "ABORTED", records.size());

    // ── Finalize PVD time series ─────────────────────────────────
    pvd_table.write();

    write_csv(out_dir + "/hysteresis.csv", records);
    return records;
}


// =============================================================================
//  Cases 4/5: Full table + FE² sub-models (1-dir / 2-dir)
// =============================================================================
//
//  Same table geometry as Case 3 (4 beams + MITC16 shell), but with
//  NonlinearSubModelEvolver continuum sub-models at each column.
//  Global fiber sections provide the initial response; after step 1
//  the FE² staggered coupling injects homogenised tangent + forces.
//
//  Case 4: One-way coupling (macro → micro, no feedback)
//  Case 5: Two-way staggered coupling (macro ↔ micro)

static std::vector<StepRecord>
run_case_fe2(bool two_way, const std::string& out_dir)
{
    const char* label = two_way ? "5 (FE², two-way)" : "4 (FE², one-way)";
    std::println("\n  Case {}: Full table + FE² sub-models", label);

    // ── Build table domain (identical to Case 3) ──────────────────
    Domain<3> domain;
    PetscInt tag = 0;

    domain.add_node(0, 0.0, 0.0, 0.0);
    domain.add_node(1,  LX, 0.0, 0.0);
    domain.add_node(2, 0.0,  LY, 0.0);
    domain.add_node(3,  LX,  LY, 0.0);

    {
        PetscInt nid = 4;
        for (int jy = 0; jy < 4; ++jy)
            for (int jx = 0; jx < 4; ++jx)
                domain.add_node(nid++,
                                LX * jx / 3.0,
                                LY * jy / 3.0,
                                H);
    }

    const std::array<std::pair<PetscInt, PetscInt>, 4> col_pairs = {{
        {0, 4}, {1, 7}, {2, 16}, {3, 19}
    }};
    for (auto [base, top] : col_pairs) {
        PetscInt conn[2] = {base, top};
        auto& g = domain.make_element<LagrangeElement3D<2>>(
            GaussLegendreCellIntegrator<2>{}, tag++, conn);
        g.set_physical_group("Columns");
    }

    {
        PetscInt slab_conn[16];
        for (int i = 0; i < 16; ++i) slab_conn[i] = 4 + i;
        auto& g = domain.make_element<LagrangeElement3D<4, 4>>(
            GaussLegendreCellIntegrator<4, 4>{}, tag++, slab_conn);
        g.set_physical_group("Slabs");
    }

    domain.assemble_sieve();

    // ── RC fiber section + elastic shell ──────────────────────────
    const auto col_mat = make_rc_column_section({
        .b = COL_B, .h = COL_H, .cover = COL_CVR,
        .bar_diameter = COL_BAR, .tie_spacing = COL_TIE,
        .fpc = COL_FPC, .nu = NU_RC,
        .steel_E = STEEL_E, .steel_fy = STEEL_FY, .steel_b = STEEL_B,
        .tie_fy = TIE_FY,
    });

    MindlinShellMaterial slab_relation{SLAB_E, NU_RC, SLAB_T};
    const auto slab_mat = Material<MindlinReissnerShell3D>{
        slab_relation, ElasticUpdate{}};

    auto builder = StructuralModelBuilder<
        BeamElemT2, ShellElemT,
        TimoshenkoBeam3D, MindlinReissnerShell3D>{};
    builder.set_frame_material("Columns", col_mat);
    builder.set_shell_material("Slabs",   slab_mat);

    auto elements = builder.build_elements(domain);
    StructModel model{domain, std::move(elements)};

    model.fix_z(0.0);

    const std::array<std::size_t, 4> slab_corners = {4, 7, 16, 19};
    for (auto nid : slab_corners)
        model.constrain_dof(nid, 0, 0.0);   // u_x controlled

    model.setup();

    std::println("  Nodes: {}  Elements: {}", domain.num_nodes(),
                 model.elements().size());

    // ── Kinematics extractor ──────────────────────────────────────
    auto extract_beam_kinematics = [&model](std::size_t e_idx) -> ElementKinematics {
        const auto& se = model.elements()[e_idx];
        const auto* beam = se.as<BeamElemT2>();
        const auto u_e = se.extract_element_dofs(model.state_vector());

        auto kin_A = extract_section_kinematics(*beam, u_e, -1.0);
        auto kin_B = extract_section_kinematics(*beam, u_e, +1.0);
        kin_A.E = EC_COL;  kin_A.G = GC_COL;  kin_A.nu = NU_RC;
        kin_B.E = EC_COL;  kin_B.G = GC_COL;  kin_B.nu = NU_RC;

        ElementKinematics ek;
        ek.element_id   = e_idx;
        ek.kin_A        = kin_A;
        ek.kin_B        = kin_B;
        ek.endpoint_A   = beam->geometry().map_local_point(std::array{-1.0});
        ek.endpoint_B   = beam->geometry().map_local_point(std::array{+1.0});
        ek.up_direction = std::array<double,3>{1.0, 0.0, 0.0};
        return ek;
    };

    // ── Build sub-models for all 4 columns ────────────────────────
    MultiscaleCoordinator coordinator;
    for (std::size_t eid = 0; eid < 4; ++eid)
        coordinator.add_critical_element(extract_beam_kinematics(eid));

    const double cvr   = COL_CVR;
    const double bar_d = COL_BAR;
    const double bar_a = std::numbers::pi / 4.0 * bar_d * bar_d;
    const double y0    = -COL_B / 2.0 + cvr + bar_d / 2.0;
    const double y1    =  COL_B / 2.0 - cvr - bar_d / 2.0;
    const double z0    = -COL_H / 2.0 + cvr + bar_d / 2.0;
    const double z1    =  COL_H / 2.0 - cvr - bar_d / 2.0;

    std::vector<SubModelSpec::RebarBar> bars = {
        {y0, z0, bar_a, bar_d}, {y1, z0, bar_a, bar_d},
        {y0, z1, bar_a, bar_d}, {y1, z1, bar_a, bar_d},
        {0.0, z0, bar_a, bar_d}, {0.0, z1, bar_a, bar_d},
        {y0, 0.0, bar_a, bar_d}, {y1, 0.0, bar_a, bar_d},
    };

    coordinator.build_sub_models(SubModelSpec{
        .section_width  = COL_B,
        .section_height = COL_H,
        .nx = SUB_NX, .ny = SUB_NY, .nz = SUB_NZ,
        .hex_order = HexOrder::Quadratic,
        .rebar_bars = std::move(bars),
        .rebar_E  = STEEL_E,
        .rebar_fy = STEEL_FY,
        .rebar_b  = STEEL_B,
    });

    {
        const auto rpt = coordinator.report();
        std::println("  Sub-models: {}  ({} hex/sub, {} nodes/sub)",
                     rpt.num_elements, rpt.total_elements, rpt.total_nodes);
    }

    // ── Nonlinear sub-model evolvers ──────────────────────────────
    const std::string evol_dir = out_dir + "/sub_models";
    std::filesystem::create_directories(evol_dir);

    std::vector<NonlinearSubModelEvolver> nl_evolvers;
    for (auto& sub : coordinator.sub_models()) {
        nl_evolvers.emplace_back(sub, COL_FPC, evol_dir, 20);
        nl_evolvers.back().set_incremental_params(20, 4);
        nl_evolvers.back().set_penalty_alpha(EC_COL * 10.0);
        nl_evolvers.back().set_snes_params(100, 2.0, 1e-3);
        if (!two_way)
            nl_evolvers.back().set_arc_length_threshold(9999);
    }

    // ── MultiscaleModel + MultiscaleAnalysis ──────────────────────
    MultiscaleModel<NonlinearSubModelEvolver> ms_model;
    ms_model.set_kinematics_extractor(extract_beam_kinematics);
    ms_model.set_response_injector(
        [&model](std::size_t eid,
                 const Eigen::Matrix<double,6,6>& D_hom,
                 const Eigen::Vector<double,6>&   f_hom)
        {
            if (auto* beam = model.elements()[eid].as<BeamElemT2>()) {
                beam->set_homogenized_tangent(D_hom);
                auto eps_ref = beam->midpoint_strain(model.state_vector());
                beam->set_homogenized_forces(f_hom, eps_ref);
            }
        });

    for (auto& ev : nl_evolvers)
        ms_model.register_local_model(ev.parent_element_id(), std::move(ev));

    std::unique_ptr<ScaleBridgePolicy> bridge = two_way
        ? std::unique_ptr<ScaleBridgePolicy>(std::make_unique<TwoWayStaggered>(MAX_STAGGERED_ITER))
        : std::unique_ptr<ScaleBridgePolicy>(std::make_unique<OneWayCoupling>());

    MultiscaleAnalysis<NonlinearSubModelEvolver> analysis(
        std::move(ms_model),
        std::move(bridge),
        std::make_unique<FrobeniusConvergence>(STAGGERED_TOL),
        std::make_unique<ConstantRelaxation>(STAGGERED_RELAX));
    analysis.set_coupling_start_step(COUPLING_START_STEP);
    analysis.set_section_dimensions(COL_B, COL_H);

    // ── Skip trivial zero-displacement initial solve ─────────────
    // The first coupling step triggers first_solve() (incremental ramp
    // from zero to actual beam kinematics), avoiding the zero→non-zero
    // jump that causes SNES divergence in subsequent_solve().
    std::println("  Sub-models will ramp at first coupling step...");

    // ── Solver ────────────────────────────────────────────────────
    PetscOptionsSetValue(nullptr, "-snes_linesearch_type", "bt");
    PetscOptionsSetValue(nullptr, "-snes_max_it", "100");
    PetscOptionsSetValue(nullptr, "-ksp_type",  "preonly");
    PetscOptionsSetValue(nullptr, "-pc_type",   "lu");

    NonlinearAnalysis<TimoshenkoBeam3D, continuum::SmallStrain, NDOF,
                      StructPolicy> nl{&model};

    // ── Recording ─────────────────────────────────────────────────
    std::vector<StepRecord> records;
    records.reserve(FE2_STEPS + 1);
    records.push_back({0, 0.0, 0.0, 0.0});

    const std::vector<std::size_t> base_nodes = {0, 1, 2, 3};
    int fe2_step = 0;
    auto t0 = std::chrono::steady_clock::now();

    // ── PVD time-series setup ────────────────────────────────────
    std::filesystem::create_directories(out_dir + "/vtk");
    PVDWriter pvd_fe2{out_dir + "/vtk/table_fe2"};

    auto beam_profile  = fall_n::reconstruction::RectangularSectionProfile<2>{COL_B, COL_H};
    auto shell_profile = fall_n::reconstruction::ShellThicknessProfile<3>{};

    nl.set_step_callback(
        [&](int step, double p, const StructModel& /*m*/) {
            double d     = cyclic_displacement(p);
            double shear = extract_base_shear_x(model, base_nodes);
            records.push_back({step, p, d, shear});

            // FE² staggered coupling
            ++fe2_step;
            analysis.step(static_cast<double>(step), fe2_step);

            // Per-step VTM snapshot
            {
                fall_n::vtk::StructuralVTMExporter vtm{
                    model, beam_profile, shell_profile};
                vtm.set_displacement(model.state_vector());
                vtm.set_yield_strain(EPS_YIELD);
                auto vtm_file = std::format("{}/vtk/table_fe2_{:04d}.vtm",
                                            out_dir, step);
                vtm.write(vtm_file);
                pvd_fe2.add_timestep(p, vtm_file);
            }

            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - t0).count();
            if (step % 5 == 0 || step == FE2_STEPS) {
                std::println("    step={:3d}/{:3d}  p={:.4f}  d={:+.4e} m  "
                             "V={:+.4e} MN  stag={}  t={}s",
                             step, FE2_STEPS, p, d, shear,
                             analysis.last_staggered_iterations(),
                             elapsed);
                std::fflush(stdout);
            }
        });

    // ── Cyclic control scheme ─────────────────────────────────────
    auto scheme = make_control(
        [&slab_corners]
        (double p, Vec /*f_full*/, Vec f_ext, StructModel* m) {
            VecSet(f_ext, 0.0);
            double d = cyclic_displacement(p);
            for (auto nid : slab_corners)
                m->update_imposed_value(nid, 0, d);          // X
        });

    bool ok = nl.solve_incremental(FE2_STEPS, MAX_BISECT, scheme);

    std::println("  Result: {} ({} records)",
                 ok ? "COMPLETED" : "ABORTED", records.size());

    // ── Finalize sub-models ───────────────────────────────────────
    for (auto& ev : analysis.model().local_models())
        ev.finalize();

    // ── Finalize PVD time series ─────────────────────────────────
    pvd_fe2.write();

    write_csv(out_dir + "/hysteresis.csv", records);
    return records;
}


// =============================================================================
//  Helper: separator
// =============================================================================
static void sep(char c = '=', int n = 72) {
    std::cout << std::string(n, c) << '\n';
}


// =============================================================================
//  Main
// =============================================================================

int main(int argc, char* argv[])
{
    setvbuf(stdout, nullptr, _IONBF, 0);

    PetscInitialize(&argc, &argv, nullptr, nullptr);
    PetscOptionsSetValue(nullptr, "-snes_monitor_cancel", "");
    PetscOptionsSetValue(nullptr, "-ksp_monitor_cancel",  "");

    int exit_code = 0;
    try {
        exit_code = [&]() -> int {

    // ── Parse --case argument ─────────────────────────────────────
    std::string case_id = "all";
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--case" && i + 1 < argc)
            case_id = argv[++i];
    }

    sep('=');
    std::println("  fall_n — Cyclic Validation of FE² Pipeline");
    std::println("  Case: {}  |  Protocol: ±2.5/5/10/20 mm (geometric)",
                 case_id);
    std::println("  Steps: {}  |  Max bisection: {}", NUM_STEPS, MAX_BISECT);
    sep('=');

    std::filesystem::create_directories(OUT_ROOT);

    // ── Dispatch ──────────────────────────────────────────────────
    std::map<std::string, std::vector<StepRecord>> results;

    auto should_run = [&](const std::string& id) {
        return case_id == "all" || case_id == id;
    };

    // ── Case 0: Linear elastic reference ────────────────────────
    if (should_run("0")) {
        auto dir = OUT_ROOT + "case0";
        std::filesystem::create_directories(dir);
        results["0"] = run_case0(dir);
    }

    // ── Case 1: Timoshenko beams N=2..10 ─────────────────────────
    //  1a = N=2, 1b = N=3, … 1i = N=10
    auto run_beam = [&]<std::size_t N>() {
        std::string sub(1, static_cast<char>('a' + (N - 2)));
        std::string id = "1" + sub;
        if (should_run(id)) {
            auto dir = OUT_ROOT + "case" + id;
            std::filesystem::create_directories(dir);
            results[id] = run_case1<N>(dir);
        }
    };
    run_beam.template operator()<2>();
    run_beam.template operator()<3>();
    run_beam.template operator()<4>();
    run_beam.template operator()<5>();
    run_beam.template operator()<6>();
    run_beam.template operator()<7>();
    run_beam.template operator()<8>();
    run_beam.template operator()<9>();
    run_beam.template operator()<10>();

    // ── Case 2: Continuum columns ────────────────────────────────
    if (should_run("2a")) {
        auto dir = OUT_ROOT + "case2a";
        std::filesystem::create_directories(dir);
        results["2a"] = run_case2(HexOrder::Linear, dir);
    }
    if (should_run("2b")) {
        auto dir = OUT_ROOT + "case2b";
        std::filesystem::create_directories(dir);
        results["2b"] = run_case2(HexOrder::Serendipity, dir);
    }
    if (should_run("2c")) {
        auto dir = OUT_ROOT + "case2c";
        std::filesystem::create_directories(dir);
        results["2c"] = run_case2(HexOrder::Quadratic, dir);
    }
    if (should_run("3")) {
        auto dir = OUT_ROOT + "case3";
        std::filesystem::create_directories(dir);
        results["3"] = run_case3(dir);
    }
    if (should_run("4")) {
        auto dir = OUT_ROOT + "case4";
        std::filesystem::create_directories(dir);
        results["4"] = run_case_fe2(false, dir);
    }
    if (should_run("5")) {
        auto dir = OUT_ROOT + "case5";
        std::filesystem::create_directories(dir);
        results["5"] = run_case_fe2(true, dir);
    }

    // ── Summary table ─────────────────────────────────────────────
    sep('=');
    std::println("\n  SUMMARY");
    sep('-');
    std::println("  {:>6s}  {:>8s}  {:>12s}  {:>12s}",
                 "Case", "Records", "Max drift", "Max shear");
    sep('-');

    for (const auto& [id, recs] : results) {
        double max_d = 0.0, max_v = 0.0;
        for (const auto& r : recs) {
            max_d = std::max(max_d, std::abs(r.drift));
            max_v = std::max(max_v, std::abs(r.base_shear));
        }
        std::println("  {:>6s}  {:8d}  {:12.4e}  {:12.4e}",
                     id, static_cast<int>(recs.size()), max_d, max_v);
    }

    sep('=');
    std::println("  Output: {}", OUT_ROOT);

    return 0;

        }();
    } catch (const std::exception& e) {
        std::cerr << "\n*** EXCEPTION: " << e.what() << " ***\n";
        PetscFinalize();
        return 1;
    }

    PetscFinalize();
    return exit_code;
}
