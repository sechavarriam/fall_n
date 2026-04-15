#include "src/validation/TableCyclicValidationSupport.hh"
#include "src/validation/ReducedRCColumnStructuralBaseline.hh"

#include <array>
#include <filesystem>
#include <numbers>

namespace fall_n::table_cyclic_validation {

std::vector<StepRecord> run_case0(const std::string& out_dir,
                                  const CyclicValidationRunConfig& cfg)
{
    std::println("\n  Case 0: Linear elastic reference (N=3 beam)");

    constexpr std::size_t N = 3;
    using BeamPolicy = SingleElementPolicy<TimoshenkoBeamN<N>>;
    using BeamModel  =
        Model<TimoshenkoBeam3D, continuum::SmallStrain, NDOF, BeamPolicy>;

    Domain<3> domain;
    PetscInt tag = 0;

    for (std::size_t i = 0; i < N; ++i) {
        const double z =
            H * static_cast<double>(i) / static_cast<double>(N - 1);
        domain.add_node(static_cast<PetscInt>(i), 0.0, 0.0, z);
    }

    PetscInt conn[N];
    for (std::size_t i = 0; i < N; ++i) {
        conn[i] = static_cast<PetscInt>(i);
    }

    auto& geom = domain.template make_element<LagrangeElement3D<N>>(
        GaussLegendreCellIntegrator<N - 1>{}, tag++, conn);
    geom.set_physical_group("Column");

    domain.assemble_sieve();

    const double Ec = 4700.0 * std::sqrt(COL_FPC);
    const double Gc = Ec / (2.0 * (1.0 + NU_RC));
    const double A  = COL_B * COL_H;
    const double Iy = COL_B * std::pow(COL_H, 3) / 12.0;
    const double Iz = COL_H * std::pow(COL_B, 3) / 12.0;
    const double J  = 0.1406 * COL_B * std::pow(COL_H, 3);
    const double k  = 5.0 / 6.0;

    TimoshenkoBeamMaterial3D rel{Ec, Gc, A, Iy, Iz, J, k, k};
    auto col_mat = Material<TimoshenkoBeam3D>{rel, ElasticUpdate{}};

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

template <std::size_t N>
static std::vector<StepRecord> run_case1_impl(
    const std::string& out_dir,
    const CyclicValidationRunConfig& cfg)
{
    static_assert(N >= 2 && N <= 10,
                  "TimoshenkoBeamN supports N=2..10");
    const char sub = static_cast<char>('a' + (N - 2));
    std::println("\n  Case 1{}: TimoshenkoBeamN<{}>  ({}-node beam, {} GPs)",
                 sub, N, N, N - 1);

    using BeamPolicy = SingleElementPolicy<TimoshenkoBeamN<N>>;
    using BeamModel  =
        Model<TimoshenkoBeam3D, continuum::SmallStrain, NDOF, BeamPolicy>;

    Domain<3> domain;
    PetscInt tag = 0;

    for (std::size_t i = 0; i < N; ++i) {
        const double z =
            H * static_cast<double>(i) / static_cast<double>(N - 1);
        domain.add_node(static_cast<PetscInt>(i), 0.0, 0.0, z);
    }

    PetscInt conn[N];
    for (std::size_t i = 0; i < N; ++i) {
        conn[i] = static_cast<PetscInt>(i);
    }

    auto& geom = domain.template make_element<LagrangeElement3D<N>>(
        GaussLegendreCellIntegrator<N - 1>{}, tag++, conn);
    geom.set_physical_group("Column");

    domain.assemble_sieve();

    const auto col_mat = make_rc_column_section({
        .b = COL_B,
        .h = COL_H,
        .cover = COL_CVR,
        .bar_diameter = COL_BAR,
        .tie_spacing = COL_TIE,
        .fpc = COL_FPC,
        .nu = NU_RC,
        .steel_E = STEEL_E,
        .steel_fy = STEEL_FY,
        .steel_b = STEEL_B,
        .tie_fy = TIE_FY,
    });

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

std::vector<StepRecord> run_case1_by_nodes(
    std::size_t nodes,
    const std::string& out_dir,
    const CyclicValidationRunConfig& cfg)
{
    return validation_reboot::run_reduced_rc_column_small_strain_beam_case(
        {
            .beam_nodes = nodes,
            .beam_axis_quadrature_family = BeamAxisQuadratureFamily::GaussLegendre,
            .axial_compression_force_mn = 0.0,
            .write_hysteresis_csv = true,
            .print_progress = true,
        },
        out_dir,
        cfg);
}

static std::vector<StepRecord> run_case2_impl(
    HexOrder order,
    const std::string& out_dir,
    const CyclicValidationRunConfig& cfg)
{
    const char* label = (order == HexOrder::Linear)
                      ? "Hex8"
                      : (order == HexOrder::Serendipity ? "Hex20" : "Hex27");
    const char sub = (order == HexOrder::Linear)
                   ? 'a'
                   : (order == HexOrder::Serendipity ? 'b' : 'c');

    std::println("\n  Case 2{}: {} reinforced column  ({}×{}×{} mesh)",
                 sub, label, C_NX, C_NY, C_NZ);

    const double cvr = COL_CVR;
    const double bar_d = COL_BAR;
    const double bar_a = std::numbers::pi / 4.0 * bar_d * bar_d;
    const double y0 = -COL_B / 2.0 + cvr + bar_d / 2.0;
    const double y1 = COL_B / 2.0 - cvr - bar_d / 2.0;
    const double z0 = -COL_H / 2.0 + cvr + bar_d / 2.0;
    const double z1 = COL_H / 2.0 - cvr - bar_d / 2.0;

    RebarSpec rebar_spec;
    rebar_spec.bars = {
        {y0, z0, bar_a, bar_d}, {y1, z0, bar_a, bar_d},
        {y0, z1, bar_a, bar_d}, {y1, z1, bar_a, bar_d},
        {0.0, z0, bar_a, bar_d}, {0.0, z1, bar_a, bar_d},
        {y0, 0.0, bar_a, bar_d}, {y1, 0.0, bar_a, bar_d},
    };

    PrismaticSpec spec{
        .width = COL_B,
        .height = COL_H,
        .length = H,
        .nx = C_NX,
        .ny = C_NY,
        .nz = C_NZ,
        .hex_order = order,
    };

    auto result = make_reinforced_prismatic_domain(spec, rebar_spec);
    auto& domain = result.domain;
    auto& grid = result.grid;

    KoBatheConcrete3D concrete_impl{COL_FPC};
    concrete_impl.set_consistent_tangent(true);
    InelasticMaterial<KoBatheConcrete3D> concrete_inst{std::move(concrete_impl)};
    Material<ThreeDimensionalMaterial> concrete_mat{concrete_inst,
                                                    InelasticUpdate{}};

    MenegottoPintoSteel steel_impl{STEEL_E, STEEL_FY, STEEL_B};
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

    const auto face_min = grid.nodes_on_face(PrismFace::MinZ);
    for (auto nid : face_min) {
        if (domain.node(static_cast<std::size_t>(nid)).num_dof() == 0) {
            continue;
        }
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
        if (domain.node(static_cast<std::size_t>(nid)).num_dof() == 0) {
            continue;
        }
        model.constrain_dof(static_cast<std::size_t>(nid), 0, 0.0);
    }

    model.setup();

    fall_n::PenaltyCoupling coupling;
    coupling.setup(domain, grid, result.embeddings, num_bars, EC_COL * 10.0,
                   true, order);

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
        if (domain.node(static_cast<std::size_t>(nid)).num_dof() == 0) {
            continue;
        }
        base_nodes.push_back(static_cast<std::size_t>(nid));
    }

    std::filesystem::create_directories(out_dir + "/vtk");
    PVDWriter pvd_mesh{out_dir + "/vtk/mesh"};
    PVDWriter pvd_gauss{out_dir + "/vtk/gauss"};

    nl.set_step_callback([&](int step, double p, const MixedModel& m) {
        const double d = cfg.displacement(p);
        const double shear = extract_base_shear_x(m, base_nodes);
        records.push_back({step, p, d, shear});

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

        if (step % 20 == 0 || step == cfg.total_steps()) {
            std::println(
                "    step={:3d}  p={:.4f}  d={:+.4e} m  V={:+.4e} MN",
                step, p, d, shear);
        }
    });

    std::vector<std::size_t> top_nodes;
    top_nodes.reserve(face_max.size());
    for (auto nid : face_max) {
        if (domain.node(static_cast<std::size_t>(nid)).num_dof() == 0) {
            continue;
        }
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

std::vector<StepRecord> run_case2_variant(
    char variant,
    const std::string& out_dir,
    const CyclicValidationRunConfig& cfg)
{
    switch (variant) {
    case 'a': return run_case2_impl(HexOrder::Linear, out_dir, cfg);
    case 'b': return run_case2_impl(HexOrder::Serendipity, out_dir, cfg);
    case 'c': return run_case2_impl(HexOrder::Quadratic, out_dir, cfg);
    default:
        throw std::invalid_argument(
            "Case 2 supports only variants a (Hex8), b (Hex20), c (Hex27).");
    }
}

std::vector<StepRecord> run_case3(const std::string& out_dir,
                                  const CyclicValidationRunConfig& cfg)
{
    std::println("\n  Case 3: Full table (4 × BeamElement + 1 × MITC16)");

    Domain<3> domain;
    PetscInt tag = 0;

    domain.add_node(0, 0.0, 0.0, 0.0);
    domain.add_node(1, LX, 0.0, 0.0);
    domain.add_node(2, 0.0, LY, 0.0);
    domain.add_node(3, LX, LY, 0.0);

    {
        PetscInt nid = 4;
        for (int jy = 0; jy < 4; ++jy) {
            for (int jx = 0; jx < 4; ++jx) {
                domain.add_node(nid++, LX * jx / 3.0, LY * jy / 3.0, H);
            }
        }
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
        for (int i = 0; i < 16; ++i) {
            slab_conn[i] = 4 + i;
        }
        auto& g = domain.make_element<LagrangeElement3D<4, 4>>(
            GaussLegendreCellIntegrator<4, 4>{}, tag++, slab_conn);
        g.set_physical_group("Slabs");
    }

    domain.assemble_sieve();

    const auto col_mat = make_rc_column_section({
        .b = COL_B,
        .h = COL_H,
        .cover = COL_CVR,
        .bar_diameter = COL_BAR,
        .tie_spacing = COL_TIE,
        .fpc = COL_FPC,
        .nu = NU_RC,
        .steel_E = STEEL_E,
        .steel_fy = STEEL_FY,
        .steel_b = STEEL_B,
        .tie_fy = TIE_FY,
    });

    MindlinShellMaterial slab_relation{SLAB_E, NU_RC, SLAB_T};
    const auto slab_mat =
        Material<MindlinReissnerShell3D>{slab_relation, ElasticUpdate{}};

    auto builder =
        StructuralModelBuilder<BeamElemT2, ShellElemT, TimoshenkoBeam3D,
                               MindlinReissnerShell3D>{};
    builder.set_frame_material("Columns", col_mat);
    builder.set_shell_material("Slabs", slab_mat);

    auto elements = builder.build_elements(domain);

    StructModel model{domain, std::move(elements)};
    model.fix_z(0.0);

    const std::array<std::size_t, 4> slab_corners = {4, 7, 16, 19};
    for (auto nid : slab_corners) {
        model.constrain_dof(nid, 0, 0.0);
    }

    model.setup();

    std::println("  Nodes: {}  Elements: {}  (4 beams + 1 shell)",
                 domain.num_nodes(), model.elements().size());

    PetscOptionsSetValue(nullptr, "-snes_linesearch_type", "bt");
    PetscOptionsSetValue(nullptr, "-snes_max_it", "100");
    PetscOptionsSetValue(nullptr, "-ksp_type", "preonly");
    PetscOptionsSetValue(nullptr, "-pc_type", "lu");

    NonlinearAnalysis<TimoshenkoBeam3D, continuum::SmallStrain, NDOF,
                      StructPolicy>
        nl{&model};
    std::filesystem::create_directories(out_dir + "/recorders");
    MaxStrainDamageCriterion damage_crit{EPS_YIELD};
    FiberHysteresisRecorder<StructModel> hysteresis_rec{
        damage_crit, classify_table_fiber, {}, 5, 1};
    nl.set_observer(hysteresis_rec);
    std::ofstream global_csv(out_dir + "/recorders/global_history.csv");
    global_csv << "step,p,drift_m,base_shear_MN,peak_damage\n";

    std::vector<StepRecord> records;
    records.reserve(static_cast<std::size_t>(cfg.total_steps()) + 1);
    records.push_back({0, 0.0, 0.0, 0.0});

    const std::vector<std::size_t> base_nodes = {0, 1, 2, 3};

    std::filesystem::create_directories(out_dir + "/vtk");
    PVDWriter pvd_table{out_dir + "/vtk/table"};

    auto beam_profile =
        fall_n::reconstruction::RectangularSectionProfile<2>{COL_B, COL_H};
    auto shell_profile =
        fall_n::reconstruction::ShellThicknessProfile<3>{};

    nl.set_step_callback([&](int step, double p, const StructModel& m) {
        const double d = cfg.displacement(p);
        const double shear = extract_base_shear_x(m, base_nodes);
        const double peak_damage = peak_structural_damage(m, damage_crit);
        records.push_back({step, p, d, shear});
        global_csv << step << "," << p << "," << d << "," << shear << ","
                   << peak_damage << "\n";

        fall_n::vtk::StructuralVTMExporter vtm{const_cast<StructModel&>(m),
                                               beam_profile, shell_profile};
        vtm.set_displacement(m.state_vector());
        vtm.set_yield_strain(EPS_YIELD);
        const auto vtm_file =
            std::format("{}/vtk/table_{:04d}.vtm", out_dir, step);
        vtm.write(vtm_file);
        pvd_table.add_timestep(p, vtm_file);

        if (step % 20 == 0 || step == cfg.total_steps()) {
            std::println(
                "    step={:3d}  p={:.4f}  d={:+.4e} m  V={:+.4e} MN",
                step, p, d, shear);
        }
    });

    auto scheme = make_control(
        [&slab_corners, &cfg](double p, Vec /*f_full*/, Vec f_ext,
                              StructModel* m) {
            VecSet(f_ext, 0.0);
            const double d = cfg.displacement(p);
            for (auto nid : slab_corners) {
                m->update_imposed_value(nid, 0, d);
            }
        });

    const bool ok =
        nl.solve_incremental(cfg.total_steps(), cfg.max_bisections, scheme);

    std::println("  Result: {} ({} records)",
                 ok ? "COMPLETED" : "ABORTED", records.size());

    pvd_table.write();
    hysteresis_rec.write_hysteresis_csv(out_dir + "/recorders/fiber_hysteresis");

    write_csv(out_dir + "/hysteresis.csv", records);
    return records;
}

} // namespace fall_n::table_cyclic_validation
