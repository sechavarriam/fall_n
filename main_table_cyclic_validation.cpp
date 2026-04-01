// =============================================================================
//  main_table_cyclic_validation.cpp
//
//  Progressive incremental validation of the FE² pipeline using a
//  cyclic-loaded "4-legged table" geometry.  Five cases of increasing
//  complexity:
//
//    Case 1a/1b/1c — Single cantilever beam (N=2,3,4 node TimoshenkoBeamN)
//    Case 2a/2b/2c — Single continuum column (Hex8/Hex20/Hex27 + rebar)
//    Case 3        — Full table structural model (4 beams + MITC16 shell)
//    Case 4        — Table + FE² sub-models, 1 direction  (TODO)
//    Case 5        — Table + FE² sub-models, 2 directions (TODO)
//
//  All cases use displacement-controlled cyclic loading with increasing
//  amplitudes: ±1δy, ±2δy, ±4δy, ±8δy (triangular protocol).
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

#include <Eigen/Dense>

#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
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
static constexpr double DELTA_Y   = 0.01;                 // approx. yield drift [m]
static constexpr int    NUM_STEPS = 120;                   // 10 steps per segment × 12
static constexpr int    MAX_BISECT = 6;

// ── Continuum mesh (Cases 2, 4, 5) ──────────────────────────────────
static constexpr int C_NX = 2;
static constexpr int C_NY = 2;
static constexpr int C_NZ = 12;

// ── Type aliases ─────────────────────────────────────────────────────
static constexpr std::size_t NDOF = 6;
using StructPolicy = SingleElementPolicy<StructuralElement>;
using StructModel  = Model<TimoshenkoBeam3D, continuum::SmallStrain, NDOF, StructPolicy>;
using BeamElemT2   = BeamElement<TimoshenkoBeam3D, 3, beam::SmallRotation>;
using ShellElemT   = MITC16Shell<>;

} // anonymous namespace


// =============================================================================
//  Cyclic displacement protocol: p ∈ [0,1] → d(p) [m]
// =============================================================================
//
//  Triangular cyclic displacement with 4 amplitude levels:
//    Level 0: ±1δy    Level 1: ±2δy    Level 2: ±4δy    Level 3: ±8δy
//
//  Each level has 3 segments: 0→+A, +A→−A, −A→0.
//  Total: 12 segments mapped onto p ∈ [0, 1].

static double cyclic_displacement(double p, double delta_y = DELTA_Y)
{
    constexpr double amps[] = {1.0, 2.0, 4.0, 8.0};
    constexpr int N_SEG = 12;   // 3 segments × 4 levels

    const double t   = p * N_SEG;
    const int    seg = std::clamp(static_cast<int>(t), 0, N_SEG - 1);
    const double f   = t - static_cast<double>(seg);

    const int level = seg / 3;
    const int phase = seg % 3;
    const double A  = amps[level] * delta_y;

    switch (phase) {
    case 0:  return  f * A;                 // 0 → +A
    case 1:  return  A * (1.0 - 2.0 * f);  // +A → −A
    case 2:  return -A * (1.0 - f);         // −A → 0
    default: return 0.0;
    }
}


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
    const char sub = (N == 2) ? 'a' : (N == 3) ? 'b' : 'c';
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

static std::vector<StepRecord>
run_case2(HexOrder order, const std::string& out_dir)
{
    const char* label = (order == HexOrder::Linear)     ? "Hex8"  :
                        (order == HexOrder::Serendipity) ? "Hex20" : "Hex27";
    const char sub = (order == HexOrder::Linear)     ? 'a' :
                     (order == HexOrder::Serendipity) ? 'b' : 'c';

    std::println("\n  Case 2{}: {} + embedded rebar  ({}×{}×{} mesh)",
                 sub, label, C_NX, C_NY, C_NZ);

    // ── Prismatic domain ──────────────────────────────────────────
    PrismaticSpec spec{
        .width  = COL_B,
        .height = COL_H,
        .length = H,
        .nx = C_NX, .ny = C_NY, .nz = C_NZ,
        .hex_order = order,
    };

    // 8-bar rebar layout matching the fiber section
    const double cvr   = COL_CVR;
    const double bar_d = COL_BAR;
    const double bar_a = std::numbers::pi / 4.0 * bar_d * bar_d;
    const double y0    = -COL_B / 2.0 + cvr + bar_d / 2.0;
    const double y1    =  COL_B / 2.0 - cvr - bar_d / 2.0;
    const double z0    = -COL_H / 2.0 + cvr + bar_d / 2.0;
    const double z1    =  COL_H / 2.0 - cvr - bar_d / 2.0;

    RebarSpec rebar;
    rebar.bars = {
        {y0, z0, bar_a, bar_d}, {y1, z0, bar_a, bar_d},
        {y0, z1, bar_a, bar_d}, {y1, z1, bar_a, bar_d},
        {0.0, z0, bar_a, bar_d}, {0.0, z1, bar_a, bar_d},
        {y0, 0.0, bar_a, bar_d}, {y1, 0.0, bar_a, bar_d},
    };

    auto rd = make_reinforced_prismatic_domain(spec, rebar);
    auto& domain      = rd.domain;
    auto& grid        = rd.grid;
    auto& rebar_range = rd.rebar_range;

    // ── Build heterogeneous element vector (hex + truss) ──────────
    InelasticMaterial<KoBatheConcrete3D> concrete_inst{COL_FPC};
    Material<ThreeDimensionalMaterial> concrete_mat{concrete_inst, InelasticUpdate{}};

    InelasticMaterial<MenegottoPintoSteel> steel_inst{
        STEEL_E, STEEL_FY, STEEL_B, 20.0, 18.5, 0.15};
    Material<UniaxialMaterial> rebar_mat{steel_inst, InelasticUpdate{}};

    using HexElem = ContinuumElement<ThreeDimensionalMaterial, 3,
                                     continuum::SmallStrain>;

    std::vector<FEM_Element> elements;
    elements.reserve(domain.num_elements());

    for (std::size_t i = 0; i < rebar_range.first; ++i) {
        elements.emplace_back(HexElem{&domain.element(i), concrete_mat});
    }

    const auto nz_sz = static_cast<std::size_t>(C_NZ);
    for (std::size_t i = rebar_range.first; i < rebar_range.last; ++i) {
        std::size_t bar_idx = (i - rebar_range.first) / nz_sz;
        double area = rebar.bars[bar_idx].area;
        elements.emplace_back(
            TrussElement<3>{&domain.element(i), rebar_mat, area});
    }

    using ContModel = Model<ThreeDimensionalMaterial, continuum::SmallStrain,
                            3, MultiElementPolicy>;

    ContModel model{domain, std::move(elements)};

    // ── BCs ───────────────────────────────────────────────────────
    // Bottom face: fully clamped
    auto face_min = grid.nodes_on_face(PrismFace::MinZ);
    for (auto nid : face_min)
        model.constrain_node(static_cast<std::size_t>(nid), {0.0, 0.0, 0.0});

    // Top face: constrain u_x for cyclic control (u_y, u_z free)
    auto face_max = grid.nodes_on_face(PrismFace::MaxZ);
    for (auto nid : face_max)
        model.constrain_dof(static_cast<std::size_t>(nid), 0, 0.0);

    model.setup();

    std::println("  Elements: {} hex + {} truss = {} total  ({} nodes)",
                 rebar_range.first,
                 rebar_range.last - rebar_range.first,
                 domain.num_elements(),
                 domain.num_nodes());

    // ── Solver ────────────────────────────────────────────────────
    PetscOptionsSetValue(nullptr, "-snes_linesearch_type", "bt");
    PetscOptionsSetValue(nullptr, "-snes_max_it", "100");
    PetscOptionsSetValue(nullptr, "-ksp_type",  "preonly");
    PetscOptionsSetValue(nullptr, "-pc_type",   "lu");

    NonlinearAnalysis<ThreeDimensionalMaterial, continuum::SmallStrain, 3,
                      MultiElementPolicy> nl{&model};

    // ── Recording ─────────────────────────────────────────────────
    std::vector<StepRecord> records;
    records.reserve(NUM_STEPS + 1);
    records.push_back({0, 0.0, 0.0, 0.0});

    std::vector<std::size_t> base_nodes;
    base_nodes.reserve(face_min.size());
    for (auto nid : face_min)
        base_nodes.push_back(static_cast<std::size_t>(nid));

    nl.set_step_callback(
        [&](int step, double p, const ContModel& m) {
            double d     = cyclic_displacement(p);
            double shear = extract_base_shear_x(m, base_nodes);
            records.push_back({step, p, d, shear});

            if (step % 20 == 0 || step == NUM_STEPS) {
                std::println("    step={:3d}  p={:.4f}  d={:+.4e} m  V={:+.4e} MN",
                             step, p, d, shear);
            }
        });

    // ── Cyclic control scheme ─────────────────────────────────────
    std::vector<std::size_t> top_nodes;
    top_nodes.reserve(face_max.size());
    for (auto nid : face_max)
        top_nodes.push_back(static_cast<std::size_t>(nid));

    auto scheme = make_control(
        [&top_nodes](double p, Vec /*f_full*/, Vec f_ext, ContModel* m) {
            VecSet(f_ext, 0.0);
            double d = cyclic_displacement(p);
            for (auto nid : top_nodes)
                m->update_imposed_value(nid, 0, d);
        });

    bool ok = nl.solve_incremental(NUM_STEPS, MAX_BISECT, scheme);

    std::println("  Result: {} ({} records)",
                 ok ? "COMPLETED" : "ABORTED", records.size());

    // ── VTK ───────────────────────────────────────────────────────
    {
        std::filesystem::create_directories(out_dir + "/vtk");
        fall_n::vtk::VTKModelExporter exporter{model};
        exporter.set_displacement();
        exporter.compute_material_fields();
        exporter.write_mesh(out_dir + "/vtk/final_mesh.vtu");
        exporter.write_gauss_points(out_dir + "/vtk/final_gauss.vtu");
    }

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

    nl.set_step_callback(
        [&](int step, double p, const StructModel& m) {
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
        [&slab_corners](double p, Vec /*f_full*/, Vec f_ext, StructModel* m) {
            VecSet(f_ext, 0.0);
            double d = cyclic_displacement(p);
            for (auto nid : slab_corners)
                m->update_imposed_value(nid, 0, d);
        });

    bool ok = nl.solve_incremental(NUM_STEPS, MAX_BISECT, scheme);

    std::println("  Result: {} ({} records)",
                 ok ? "COMPLETED" : "ABORTED", records.size());

    // ── VTK ───────────────────────────────────────────────────────
    {
        std::filesystem::create_directories(out_dir + "/vtk");
        fall_n::vtk::StructuralVTMExporter vtm{
            model,
            fall_n::reconstruction::RectangularSectionProfile<2>{COL_B, COL_H},
            fall_n::reconstruction::ShellThicknessProfile<3>{}};
        vtm.set_displacement(model.state_vector());
        vtm.set_yield_strain(EPS_YIELD);
        vtm.write(out_dir + "/vtk/final.vtm");
    }

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
    std::println("  Case: {}  |  Protocol: ±1/2/4/8 δy  |  δy = {} m",
                 case_id, DELTA_Y);
    std::println("  Steps: {}  |  Max bisection: {}", NUM_STEPS, MAX_BISECT);
    sep('=');

    std::filesystem::create_directories(OUT_ROOT);

    // ── Dispatch ──────────────────────────────────────────────────
    std::map<std::string, std::vector<StepRecord>> results;

    auto should_run = [&](const std::string& id) {
        return case_id == "all" || case_id == id;
    };

    if (should_run("1a")) {
        auto dir = OUT_ROOT + "case1a";
        std::filesystem::create_directories(dir);
        results["1a"] = run_case1<2>(dir);
    }
    if (should_run("1b")) {
        auto dir = OUT_ROOT + "case1b";
        std::filesystem::create_directories(dir);
        results["1b"] = run_case1<3>(dir);
    }
    if (should_run("1c")) {
        auto dir = OUT_ROOT + "case1c";
        std::filesystem::create_directories(dir);
        results["1c"] = run_case1<4>(dir);
    }
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
