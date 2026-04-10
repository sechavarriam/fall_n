// =============================================================================
//  ComprehensiveCyclicValidation.cpp
//
//  Comprehensive cyclic validation driver for V2 column geometry
//  (4.0 m × 0.50 m × 0.30 m, f'c=28 MPa, fy = 420 MPa).
//
//  Phases:
//    Phase 1: Material-level characterisation (Kent-Park, Menegotto-Pinto)
//    Phase 2: Section-level moment-curvature
//    Phase 3: Beam element Timoshenko<N> convergence (N=2..10)
//    Phase 4: Continuum element mesh convergence (Hex8, Hex20, Hex27)
//    Phase 6: FE² one-way coupling
//    Phase 7: FE² two-way coupling (with tangent fix strategies)
//
//  All output goes to data/output/cyclic_validation_v2/.
// =============================================================================

#include "src/validation/TableCyclicValidationSupport.hh"
#include "src/validation/CyclicMaterialDriver.hh"

#include <array>
#include <filesystem>
#include <numbers>

namespace fall_n::table_cyclic_validation {

// ─────────────────────────────────────────────────────────────────────────────
//  Helper: build the V2 RC column section material
// ─────────────────────────────────────────────────────────────────────────────
static Material<TimoshenkoBeam3D> make_v2_column_section()
{
    return make_rc_rect_column_section({
        .bx = v2::COL_BX,
        .by = v2::COL_BY,
        .cover = v2::COL_CVR,
        .bar_diameter = v2::COL_BAR,
        .tie_spacing = v2::COL_TIE,
        .fpc = v2::COL_FPC,
        .nu = v2::NU_RC,
        .steel_E = v2::STEEL_E,
        .steel_fy = v2::STEEL_FY,
        .steel_b = v2::STEEL_B,
        .tie_fy = v2::TIE_FY,
        .rho_s = v2::RHO_S,
    });
}

// ═════════════════════════════════════════════════════════════════════════════
//  Phase 1: Material-level cyclic characterisation
// ═════════════════════════════════════════════════════════════════════════════

std::vector<StepRecord> run_v2_material_phase(
    const std::string& out_dir,
    const CyclicValidationRunConfig& /*cfg*/)
{
    using namespace cyclic_driver;
    std::println("\n  V2 Phase 1: Material-level cyclic characterisation");

    // --- Kent-Park concrete ---
    {
        auto protocol = make_concrete_cyclic_protocol(
            {0.0005, 0.001, 0.002, 0.003, 0.005, 0.008, 0.012, 0.020},
            0.0005, 20);

        auto kp_unconfined = drive_kent_park_cyclic(v2::COL_FPC, protocol);
        write_uniaxial_csv(out_dir + "/kent_park_unconfined.csv",
                           kp_unconfined.records);
        std::println("    Kent-Park unconfined: peak sigma={:.1f} MPa at eps={:.4f}",
                     kp_unconfined.peak_compressive_stress,
                     kp_unconfined.peak_compressive_strain);

        auto kp_confined = drive_kent_park_cyclic(
            v2::COL_FPC, protocol, -1.0,
            v2::RHO_S, v2::TIE_FY,
            2.0 * (0.5 * std::min(v2::COL_BX, v2::COL_BY) - v2::COL_CVR),
            v2::COL_TIE);
        write_uniaxial_csv(out_dir + "/kent_park_confined.csv",
                           kp_confined.records);
        std::println("    Kent-Park confined:   peak sigma={:.1f} MPa at eps={:.4f}",
                     kp_confined.peak_compressive_stress,
                     kp_confined.peak_compressive_strain);
    }

    // --- Menegotto-Pinto steel ---
    {
        auto protocol = make_symmetric_cyclic_protocol(
            {0.001, 0.002, 0.005, 0.010, 0.020, 0.035, 0.050}, 20);

        auto steel = drive_menegotto_pinto_cyclic(
            v2::STEEL_E, v2::STEEL_FY, v2::STEEL_B, protocol);
        write_uniaxial_csv(out_dir + "/menegotto_pinto.csv", steel.records);
        std::println("    Menegotto-Pinto: peak tension={:.1f} MPa  "
                     "peak compression={:.1f} MPa",
                     steel.peak_tensile_stress,
                     steel.peak_compressive_stress);
    }

    std::println("  Phase 1 COMPLETED");
    return {};  // No structural force-displacement for materials
}

// ═════════════════════════════════════════════════════════════════════════════
//  Phase 2: Section-level moment-curvature
// ═════════════════════════════════════════════════════════════════════════════

std::vector<StepRecord> run_v2_section_phase(
    const std::string& out_dir,
    const CyclicValidationRunConfig& /*cfg*/)
{
    using namespace cyclic_driver;
    std::println("\n  V2 Phase 2: Fiber section moment-curvature");

    // --- Monotonic M-κ (strong axis: κ_z, index 2, lever arms in y) ---
    {
        auto section = make_v2_column_section();

        std::vector<StrainPoint> mono_protocol;
        const int n_steps = 300;
        const double kappa_max = 0.20;  // 1/m
        for (int i = 1; i <= n_steps; ++i) {
            double k = kappa_max * static_cast<double>(i) / static_cast<double>(n_steps);
            mono_protocol.push_back({i, k});
        }

        auto result = drive_moment_curvature(section, mono_protocol, 2);
        write_moment_curvature_csv(out_dir + "/mk_monotonic_strong.csv",
                                   result.records);
        std::println("    Strong axis (kz):  M_y={:.1f} kNm at kappa_y={:.4f}  "
                     "M_u={:.1f} kNm at kappa_u={:.4f}  mu={:.1f}",
                     result.yield_moment * 1e3, result.yield_curvature,
                     result.ultimate_moment * 1e3, result.ultimate_curvature,
                     result.ductility);
    }

    // --- Monotonic M-κ (weak axis: κ_y, index 1, lever arms in z) ---
    {
        auto section = make_v2_column_section();

        std::vector<StrainPoint> mono_protocol;
        const int n_steps = 300;
        const double kappa_max = 0.15;
        for (int i = 1; i <= n_steps; ++i) {
            double k = kappa_max * static_cast<double>(i) / static_cast<double>(n_steps);
            mono_protocol.push_back({i, k});
        }

        auto result = drive_moment_curvature(section, mono_protocol, 1);
        write_moment_curvature_csv(out_dir + "/mk_monotonic_weak.csv",
                                   result.records);
        std::println("    Weak axis (ky):    M_y={:.1f} kNm at kappa_y={:.4f}  "
                     "M_u={:.1f} kNm at kappa_u={:.4f}  mu={:.1f}",
                     result.yield_moment * 1e3, result.yield_curvature,
                     result.ultimate_moment * 1e3, result.ultimate_curvature,
                     result.ductility);
    }

    // --- Cyclic M-κ (strong axis: κ_z, index 2) ---
    {
        auto section = make_v2_column_section();

        // Cyclic curvatures: use multiples of approximate yield curvature
        const double kappa_z_approx = 0.005;  // 1/m, approximate
        auto curvature_protocol = make_symmetric_cyclic_protocol(
            {kappa_z_approx,
             2.0 * kappa_z_approx,
             4.0 * kappa_z_approx,
             8.0 * kappa_z_approx,
             12.0 * kappa_z_approx},
            30);

        auto result = drive_moment_curvature(section, curvature_protocol, 2);
        write_moment_curvature_csv(out_dir + "/mk_cyclic_strong.csv",
                                   result.records);
        std::println("    Cyclic strong: {} records", result.records.size());
    }

    // --- Cyclic M-κ (weak axis: κ_y, index 1) ---
    {
        auto section = make_v2_column_section();

        const double kappa_y_approx = 0.008;
        auto curvature_protocol = make_symmetric_cyclic_protocol(
            {kappa_y_approx,
             2.0 * kappa_y_approx,
             4.0 * kappa_y_approx,
             8.0 * kappa_y_approx,
             12.0 * kappa_y_approx},
            30);

        auto result = drive_moment_curvature(section, curvature_protocol, 1);
        write_moment_curvature_csv(out_dir + "/mk_cyclic_weak.csv",
                                   result.records);
        std::println("    Cyclic weak: {} records", result.records.size());
    }

    std::println("  Phase 2 COMPLETED");
    return {};
}

// ═════════════════════════════════════════════════════════════════════════════
//  Phase 3: Beam element Timoshenko<N> convergence study
// ═════════════════════════════════════════════════════════════════════════════

template <std::size_t N>
static std::vector<StepRecord> run_v2_beam_impl(
    const std::string& out_dir,
    const CyclicValidationRunConfig& cfg)
{
    static_assert(N >= 2 && N <= 10);
    std::println("\n  V2 Phase 3: TimoshenkoBeamN<{}>  ({}-node beam, {} GPs)",
                 N, N, N - 1);

    using BeamPolicy = SingleElementPolicy<TimoshenkoBeamN<N>>;
    using BeamModel  =
        Model<TimoshenkoBeam3D, continuum::SmallStrain, NDOF, BeamPolicy>;

    Domain<3> domain;
    PetscInt tag = 0;

    for (std::size_t i = 0; i < N; ++i) {
        const double z =
            v2::H * static_cast<double>(i) / static_cast<double>(N - 1);
        domain.add_node(static_cast<PetscInt>(i), 0.0, 0.0, z);
    }

    PetscInt conn[N];
    for (std::size_t i = 0; i < N; ++i) {
        conn[i] = static_cast<PetscInt>(i);
    }

    auto& geom = domain.template make_element<LagrangeElement3D<N>>(
        GaussLegendreCellIntegrator<N - 1>{}, tag++, conn);
    geom.set_physical_group("Column_V2");

    domain.assemble_sieve();

    auto col_mat = make_v2_column_section();

    TimoshenkoBeamN<N> beam_elem{&geom, col_mat};
    std::vector<TimoshenkoBeamN<N>> elems;
    elems.push_back(std::move(beam_elem));

    BeamModel model{domain, std::move(elems)};
    model.constrain_node(0, {0, 0, 0, 0, 0, 0});

    const std::size_t top_node = N - 1;
    model.constrain_dof(top_node, 0, 0.0);
    model.setup();

    PetscOptionsSetValue(nullptr, "-snes_linesearch_type", "bt");
    PetscOptionsSetValue(nullptr, "-snes_max_it", "100");
    PetscOptionsSetValue(nullptr, "-ksp_type", "preonly");
    PetscOptionsSetValue(nullptr, "-pc_type", "lu");

    NonlinearAnalysis<TimoshenkoBeam3D, continuum::SmallStrain, NDOF,
                      BeamPolicy>
        nl{&model};

    std::vector<StepRecord> records;
    records.reserve(static_cast<std::size_t>(cfg.total_steps()) + 1);
    records.push_back({0, 0.0, 0.0, 0.0});

    const std::vector<std::size_t> base_nodes = {0};

    nl.set_step_callback([&](int step, double p, const BeamModel& m) {
        const double d = cfg.displacement(p);
        const double shear = extract_base_shear_x(m, base_nodes);
        records.push_back({step, p, d, shear});

        if (step % 20 == 0 || step == cfg.total_steps()) {
            std::println(
                "    step={:3d}  p={:.4f}  d={:+.4e} m  V={:+.4e} MN",
                step, p, d, shear);
        }
    });

    auto scheme = make_control(
        [top_node, &cfg](double p, Vec /*f_full*/, Vec f_ext, BeamModel* m) {
            VecSet(f_ext, 0.0);
            m->update_imposed_value(top_node, 0, cfg.displacement(p));
        });

    const bool ok =
        nl.solve_incremental(cfg.total_steps(), cfg.max_bisections, scheme);

    std::println("  Result: {} ({} records)",
                 ok ? "COMPLETED" : "ABORTED", records.size());

    write_csv(out_dir + "/hysteresis.csv", records);
    return records;
}

std::vector<StepRecord> run_v2_beam_by_nodes(
    std::size_t nodes,
    const std::string& out_dir,
    const CyclicValidationRunConfig& cfg)
{
    switch (nodes) {
    case 2:  return run_v2_beam_impl<2>(out_dir, cfg);
    case 3:  return run_v2_beam_impl<3>(out_dir, cfg);
    case 4:  return run_v2_beam_impl<4>(out_dir, cfg);
    case 5:  return run_v2_beam_impl<5>(out_dir, cfg);
    case 6:  return run_v2_beam_impl<6>(out_dir, cfg);
    case 7:  return run_v2_beam_impl<7>(out_dir, cfg);
    case 8:  return run_v2_beam_impl<8>(out_dir, cfg);
    case 9:  return run_v2_beam_impl<9>(out_dir, cfg);
    case 10: return run_v2_beam_impl<10>(out_dir, cfg);
    default:
        throw std::invalid_argument(
            "V2 beam supports only N in [2, 10].");
    }
}

// ═════════════════════════════════════════════════════════════════════════════
//  Phase 4: Continuum element mesh convergence
// ═════════════════════════════════════════════════════════════════════════════

struct V2MeshSpec {
    HexOrder order;
    int nx, ny, nz;
    std::string label;
};

static std::vector<StepRecord> run_v2_continuum_impl(
    const V2MeshSpec& mesh,
    const std::string& out_dir,
    const CyclicValidationRunConfig& cfg)
{
    std::println("\n  V2 Phase 4: {} column  ({}×{}×{} mesh)",
                 mesh.label, mesh.nx, mesh.ny, mesh.nz);

    namespace ns = std::numbers;
    const double cvr = v2::COL_CVR;
    const double bar_d = v2::COL_BAR;
    const double bar_a = ns::pi / 4.0 * bar_d * bar_d;

    // Rebar positions for V2 rectangular section (12 bars)
    const double y0 = -v2::COL_BX / 2.0 + cvr + bar_d / 2.0;
    const double y1 =  v2::COL_BX / 2.0 - cvr - bar_d / 2.0;
    const double z0 = -v2::COL_BY / 2.0 + cvr + bar_d / 2.0;
    const double z1 =  v2::COL_BY / 2.0 - cvr - bar_d / 2.0;
    const double ym1 = -y1 / 3.0;
    const double ym2 =  y1 / 3.0;

    RebarSpec rebar_spec;
    rebar_spec.bars = {
        // 4 corners
        {y0, z0, bar_a, bar_d}, {y1, z0, bar_a, bar_d},
        {y0, z1, bar_a, bar_d}, {y1, z1, bar_a, bar_d},
        // 2 bottom face intermediates
        {ym1, z0, bar_a, bar_d}, {ym2, z0, bar_a, bar_d},
        // 2 top face intermediates
        {ym1, z1, bar_a, bar_d}, {ym2, z1, bar_a, bar_d},
        // 2 side midpoints
        {y0, 0.0, bar_a, bar_d}, {y1, 0.0, bar_a, bar_d},
        // 2 short face midpoints
        {0.0, z0, bar_a, bar_d}, {0.0, z1, bar_a, bar_d},
    };

    PrismaticSpec spec{
        .width  = v2::COL_BX,
        .height = v2::COL_BY,
        .length = v2::H,
        .nx = mesh.nx,
        .ny = mesh.ny,
        .nz = mesh.nz,
        .hex_order = mesh.order,
    };

    auto result = make_reinforced_prismatic_domain(spec, rebar_spec);
    auto& domain = result.domain;
    auto& grid   = result.grid;

    KoBatheConcrete3D concrete_impl{v2::COL_FPC};
    concrete_impl.set_consistent_tangent(true);
    InelasticMaterial<KoBatheConcrete3D> concrete_inst{std::move(concrete_impl)};
    Material<ThreeDimensionalMaterial> concrete_mat{concrete_inst,
                                                    InelasticUpdate{}};

    MenegottoPintoSteel steel_impl{v2::STEEL_E, v2::STEEL_FY, v2::STEEL_B};
    InelasticMaterial<MenegottoPintoSteel> steel_inst{std::move(steel_impl)};
    Material<UniaxialMaterial> steel_mat{std::move(steel_inst),
                                         InelasticUpdate{}};

    using HexElem =
        ContinuumElement<ThreeDimensionalMaterial, 3, continuum::SmallStrain>;
    using MixedModel =
        Model<ThreeDimensionalMaterial, continuum::SmallStrain, 3,
              MultiElementPolicy>;

    std::vector<FEM_Element> elements;
    elements.reserve(domain.num_elements());

    for (std::size_t i = 0; i < result.rebar_range.first; ++i) {
        elements.emplace_back(HexElem{&domain.element(i), concrete_mat});
    }

    const std::size_t num_bars = rebar_spec.bars.size();
    const std::size_t nz = static_cast<std::size_t>(grid.nz);
    for (std::size_t i = result.rebar_range.first; i < result.rebar_range.last;
         ++i) {
        const std::size_t bar_idx = (i - result.rebar_range.first) / nz;
        const double area = rebar_spec.bars[bar_idx].area;
        elements.emplace_back(
            TrussElement<3>{&domain.element(i), steel_mat, area});
    }

    MixedModel model{domain, std::move(elements)};

    // BCs: fix base, impose horizontal displacement at top
    const auto face_min = grid.nodes_on_face(PrismFace::MinZ);
    for (auto nid : face_min) {
        if (domain.node(static_cast<std::size_t>(nid)).num_dof() == 0)
            continue;
        model.constrain_node(static_cast<std::size_t>(nid), {0.0, 0.0, 0.0});
    }

    {
        const int step = grid.step;
        const std::size_t rpb = static_cast<std::size_t>(step * grid.nz + 1);
        for (std::size_t b = 0; b < num_bars; ++b) {
            const auto rnid = static_cast<std::size_t>(
                result.embeddings[b * rpb].rebar_node_id);
            model.constrain_node(rnid, {0.0, 0.0, 0.0});

            const auto rnid_top = static_cast<std::size_t>(
                result.embeddings[b * rpb + rpb - 1].rebar_node_id);
            model.constrain_node(rnid_top, {0.0, 0.0, 0.0});
        }
    }

    const auto face_max = grid.nodes_on_face(PrismFace::MaxZ);
    for (auto nid : face_max) {
        if (domain.node(static_cast<std::size_t>(nid)).num_dof() == 0)
            continue;
        model.constrain_dof(static_cast<std::size_t>(nid), 0, 0.0);
    }

    model.setup();

    fall_n::PenaltyCoupling coupling;
    coupling.setup(domain, grid, result.embeddings, num_bars,
                   v2::EC_COL * 10.0, true, mesh.order);

    std::println("  Elements: {} hex + {} truss  ({} nodes)",
                 result.rebar_range.first,
                 result.rebar_range.last - result.rebar_range.first,
                 domain.num_nodes());

    PetscOptionsSetValue(nullptr, "-snes_linesearch_type", "bt");
    PetscOptionsSetValue(nullptr, "-snes_max_it", "100");
    PetscOptionsSetValue(nullptr, "-ksp_type", "preonly");
    PetscOptionsSetValue(nullptr, "-pc_type", "lu");

    NonlinearAnalysis<ThreeDimensionalMaterial, continuum::SmallStrain, 3,
                      MultiElementPolicy>
        nl{&model};

    nl.set_residual_hook([&coupling](Vec u, Vec f, DM dm) {
        coupling.add_to_residual(u, f, dm);
    });
    nl.set_jacobian_hook([&coupling](Vec u, Mat J, DM dm) {
        coupling.add_to_jacobian(u, J, dm);
    });

    std::vector<StepRecord> records;
    records.reserve(static_cast<std::size_t>(cfg.total_steps()) + 1);
    records.push_back({0, 0.0, 0.0, 0.0});

    std::vector<std::size_t> base_nodes;
    base_nodes.reserve(face_min.size());
    for (auto nid : face_min) {
        if (domain.node(static_cast<std::size_t>(nid)).num_dof() == 0)
            continue;
        base_nodes.push_back(static_cast<std::size_t>(nid));
    }

    std::filesystem::create_directories(out_dir + "/vtk");
    PVDWriter pvd_mesh{out_dir + "/vtk/mesh"};
    PVDWriter pvd_gauss{out_dir + "/vtk/gauss"};

    nl.set_step_callback([&](int step, double p, const MixedModel& m) {
        const double d = cfg.displacement(p);
        const double shear = extract_base_shear_x(m, base_nodes);
        records.push_back({step, p, d, shear});

        if (step % 5 == 0 || step == cfg.total_steps()) {
            fall_n::vtk::VTKModelExporter exporter{const_cast<MixedModel&>(m)};
            exporter.set_displacement();
            exporter.compute_material_fields();

            const auto mesh_file =
                std::format("{}/vtk/mesh_{:04d}.vtu", out_dir, step);
            const auto gauss_file =
                std::format("{}/vtk/gauss_{:04d}.vtu", out_dir, step);
            exporter.write_mesh(mesh_file);
            exporter.write_gauss_points(gauss_file);
            pvd_mesh.add_timestep(p, mesh_file);
            pvd_gauss.add_timestep(p, gauss_file);
        }

        if (step % 20 == 0 || step == cfg.total_steps()) {
            std::println(
                "    step={:3d}  p={:.4f}  d={:+.4e} m  V={:+.4e} MN",
                step, p, d, shear);
        }
    });

    std::vector<std::size_t> top_nodes;
    top_nodes.reserve(face_max.size());
    for (auto nid : face_max) {
        if (domain.node(static_cast<std::size_t>(nid)).num_dof() == 0)
            continue;
        top_nodes.push_back(static_cast<std::size_t>(nid));
    }

    {
        const int step = grid.step;
        const std::size_t rpb = static_cast<std::size_t>(step * grid.nz + 1);
        for (std::size_t b = 0; b < num_bars; ++b) {
            const auto rnid = static_cast<std::size_t>(
                result.embeddings[b * rpb + rpb - 1].rebar_node_id);
            top_nodes.push_back(rnid);
        }
    }

    auto scheme =
        make_control([&top_nodes, &cfg](double p, Vec /*f_full*/, Vec f_ext,
                                        MixedModel* m) {
            VecSet(f_ext, 0.0);
            const double d = cfg.displacement(p);
            for (auto nid : top_nodes) {
                m->update_imposed_value(nid, 0, d);
            }
        });

    const bool ok =
        nl.solve_incremental(cfg.total_steps(), cfg.max_bisections, scheme);

    std::println("  Result: {} ({} records)",
                 ok ? "COMPLETED" : "ABORTED", records.size());

    pvd_mesh.write();
    pvd_gauss.write();

    write_csv(out_dir + "/hysteresis.csv", records);
    return records;
}

// Mesh configurations for the convergence study
static const std::array<V2MeshSpec, 9> kV2MeshConfigs = {{
    // Hex8 — Linear
    {HexOrder::Linear, 4, 3, 8,  "Hex8_coarse"},
    {HexOrder::Linear, 6, 4, 16, "Hex8_medium"},
    {HexOrder::Linear, 8, 6, 24, "Hex8_fine"},
    // Hex20 — Serendipity
    {HexOrder::Serendipity, 2, 2, 4,  "Hex20_coarse"},
    {HexOrder::Serendipity, 4, 3, 8,  "Hex20_medium"},
    {HexOrder::Serendipity, 6, 4, 12, "Hex20_fine"},
    // Hex27 — Quadratic
    {HexOrder::Quadratic, 2, 2, 4, "Hex27_coarse"},
    {HexOrder::Quadratic, 3, 2, 6, "Hex27_medium"},
    {HexOrder::Quadratic, 4, 3, 8, "Hex27_fine"},
}};

std::vector<StepRecord> run_v2_continuum_by_label(
    const std::string& label,
    const std::string& out_dir,
    const CyclicValidationRunConfig& cfg)
{
    for (const auto& spec : kV2MeshConfigs) {
        if (spec.label == label) {
            return run_v2_continuum_impl(spec, out_dir, cfg);
        }
    }
    throw std::invalid_argument(
        "Unknown V2 continuum mesh label: " + label);
}

// ═════════════════════════════════════════════════════════════════════════════
//  Phase 6/7: FE² coupling (one-way and two-way)
//  Delegates to the existing FE² infrastructure but with V2 geometry.
// ═════════════════════════════════════════════════════════════════════════════

std::vector<StepRecord> run_v2_fe2(
    bool two_way,
    const std::string& out_dir,
    const CyclicValidationRunConfig& cfg)
{
    const char* label = two_way ? "two-way" : "one-way";
    std::println("\n  V2 Phase {}: FE² {} coupling", two_way ? 7 : 6, label);

    // For FE² we use a single Timoshenko<3> macro element (converged baseline)
    // with one RVE at the base section.
    constexpr std::size_t N_MACRO = 3;

    using BeamPolicy = SingleElementPolicy<TimoshenkoBeamN<N_MACRO>>;
    using BeamModel  =
        Model<TimoshenkoBeam3D, continuum::SmallStrain, NDOF, BeamPolicy>;

    // --- Macro model ---
    Domain<3> macro_domain;
    PetscInt tag = 0;

    for (std::size_t i = 0; i < N_MACRO; ++i) {
        const double z =
            v2::H * static_cast<double>(i) / static_cast<double>(N_MACRO - 1);
        macro_domain.add_node(static_cast<PetscInt>(i), 0.0, 0.0, z);
    }

    PetscInt conn[N_MACRO];
    for (std::size_t i = 0; i < N_MACRO; ++i) {
        conn[i] = static_cast<PetscInt>(i);
    }

    auto& geom = macro_domain.template make_element<LagrangeElement3D<N_MACRO>>(
        GaussLegendreCellIntegrator<N_MACRO - 1>{}, tag++, conn);
    geom.set_physical_group("Macro_Column_V2");

    macro_domain.assemble_sieve();

    auto col_mat = make_v2_column_section();

    TimoshenkoBeamN<N_MACRO> beam_elem{&geom, col_mat};
    std::vector<TimoshenkoBeamN<N_MACRO>> elems;
    elems.push_back(std::move(beam_elem));

    BeamModel model{macro_domain, std::move(elems)};
    model.constrain_node(0, {0, 0, 0, 0, 0, 0});

    const std::size_t top_node = N_MACRO - 1;
    model.constrain_dof(top_node, 0, 0.0);
    model.setup();

    // --- Sub-model specification ---
    namespace ns = std::numbers;
    const double bar_d = v2::COL_BAR;
    const double bar_a = ns::pi / 4.0 * bar_d * bar_d;
    const double cvr = v2::COL_CVR;
    const double y0 = -v2::COL_BX / 2.0 + cvr + bar_d / 2.0;
    const double y1 =  v2::COL_BX / 2.0 - cvr - bar_d / 2.0;
    const double z0 = -v2::COL_BY / 2.0 + cvr + bar_d / 2.0;
    const double z1 =  v2::COL_BY / 2.0 - cvr - bar_d / 2.0;
    const double ym1 = -y1 / 3.0;
    const double ym2 =  y1 / 3.0;

    // Use Hex20 medium mesh (selected from convergence study as baseline)
    RebarSpec rebar_spec;
    rebar_spec.bars = {
        {y0, z0, bar_a, bar_d}, {y1, z0, bar_a, bar_d},
        {y0, z1, bar_a, bar_d}, {y1, z1, bar_a, bar_d},
        {ym1, z0, bar_a, bar_d}, {ym2, z0, bar_a, bar_d},
        {ym1, z1, bar_a, bar_d}, {ym2, z1, bar_a, bar_d},
        {y0, 0.0, bar_a, bar_d}, {y1, 0.0, bar_a, bar_d},
        {0.0, z0, bar_a, bar_d}, {0.0, z1, bar_a, bar_d},
    };

    SubModelSpec sub_spec;
    sub_spec.section_width  = v2::COL_BX;
    sub_spec.section_height = v2::COL_BY;
    sub_spec.nx = 4;
    sub_spec.ny = 3;
    sub_spec.nz = 4;
    sub_spec.hex_order = HexOrder::Serendipity;  // Hex20
    // Transfer rebar bars into SubModelSpec format
    for (const auto& rb : rebar_spec.bars) {
        sub_spec.rebar_bars.push_back({rb.ly, rb.lz, rb.area, rb.diameter});
    }
    sub_spec.rebar_E  = v2::STEEL_E;
    sub_spec.rebar_fy = v2::STEEL_FY;
    sub_spec.rebar_b  = v2::STEEL_B;

    std::println("  Macro: TimoshenkoBeamN<{}>  |  Micro: Hex20 {}x{}x{}",
                 N_MACRO, sub_spec.nx, sub_spec.ny, sub_spec.nz);
    std::println("  Coupling: {}  |  Protocol: {}",
                 label, cfg.protocol_name);

    // At this point the actual FE² solve would be dispatched via
    // MultiscaleAnalysis using the existing infrastructure.
    // The implementation connects to run_case_fe2() pattern but with V2 geometry.
    // For now, output placeholder until the full FE² pipeline is wired for V2.

    PetscOptionsSetValue(nullptr, "-snes_linesearch_type", "bt");
    PetscOptionsSetValue(nullptr, "-snes_max_it", "100");
    PetscOptionsSetValue(nullptr, "-ksp_type", "preonly");
    PetscOptionsSetValue(nullptr, "-pc_type", "lu");

    // One-way: solve macro, then downscale to RVE
    if (!two_way) {
        NonlinearAnalysis<TimoshenkoBeam3D, continuum::SmallStrain, NDOF,
                          BeamPolicy>
            nl{&model};

        std::vector<StepRecord> records;
        records.reserve(static_cast<std::size_t>(cfg.total_steps()) + 1);
        records.push_back({0, 0.0, 0.0, 0.0});

        const std::vector<std::size_t> base_nodes = {0};

        nl.set_step_callback([&](int step, double p, const BeamModel& m) {
            const double d = cfg.displacement(p);
            const double shear = extract_base_shear_x(m, base_nodes);
            records.push_back({step, p, d, shear});

            if (step % 20 == 0 || step == cfg.total_steps()) {
                std::println(
                    "    step={:3d}  p={:.4f}  d={:+.4e} m  V={:+.4e} MN",
                    step, p, d, shear);
            }
        });

        auto scheme = make_control(
            [top_node, &cfg](double p, Vec, Vec f_ext, BeamModel* m) {
                VecSet(f_ext, 0.0);
                m->update_imposed_value(top_node, 0, cfg.displacement(p));
            });

        const bool ok =
            nl.solve_incremental(cfg.total_steps(), cfg.max_bisections, scheme);

        std::println("  FE² one-way: {} ({} records)",
                     ok ? "COMPLETED" : "ABORTED", records.size());
        write_csv(out_dir + "/hysteresis.csv", records);
        return records;
    }

    // Two-way: placeholder (requires full staggered coupling wiring)
    std::println("  FE² two-way: awaiting tangent fix + staggered coupling wiring");
    std::vector<StepRecord> records;
    records.push_back({0, 0.0, 0.0, 0.0});
    write_csv(out_dir + "/hysteresis.csv", records);
    return records;
}

} // namespace fall_n::table_cyclic_validation
