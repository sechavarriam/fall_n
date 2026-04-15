#include "src/validation/ReducedRCColumnStructuralBaseline.hh"

#include "src/elements/TimoshenkoBeamN.hh"
#include "src/materials/RCSectionBuilder.hh"
#include "src/validation/TableCyclicValidationSupport.hh"

#include <algorithm>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <print>
#include <stdexcept>

namespace fall_n::validation_reboot {

namespace {

using table_cyclic_validation::COL_B;
using table_cyclic_validation::COL_BAR;
using table_cyclic_validation::COL_CVR;
using table_cyclic_validation::COL_FPC;
using table_cyclic_validation::COL_H;
using table_cyclic_validation::COL_TIE;
using table_cyclic_validation::NDOF;
using table_cyclic_validation::NU_RC;
using table_cyclic_validation::STEEL_B;
using table_cyclic_validation::STEEL_E;
using table_cyclic_validation::STEEL_FY;
using table_cyclic_validation::StepRecord;
using table_cyclic_validation::StructModel;
using table_cyclic_validation::TIE_FY;
using table_cyclic_validation::extract_base_shear_x;
using table_cyclic_validation::write_csv;

void write_section_response_csv(
    const std::string& path,
    const std::vector<ReducedRCColumnSectionResponseRecord>& records)
{
    std::ofstream ofs(path);
    ofs << "step,p,drift_m,section_gp,xi,axial_strain,curvature_y,curvature_z,"
           "axial_force_MN,moment_y_MNm,moment_z_MNm,tangent_ea,"
           "tangent_eiy,tangent_eiz\n";
    ofs << std::scientific << std::setprecision(8);
    for (const auto& r : records) {
        ofs << r.step << ","
            << r.p << ","
            << r.drift << ","
            << r.section_gp << ","
            << r.xi << ","
            << r.axial_strain << ","
            << r.curvature_y << ","
            << r.curvature_z << ","
            << r.axial_force << ","
            << r.moment_y << ","
            << r.moment_z << ","
            << r.tangent_ea << ","
            << r.tangent_eiy << ","
            << r.tangent_eiz << "\n";
    }
    std::println("  CSV: {} ({} records)", path, records.size());
}

void write_base_side_moment_curvature_csv(
    const std::string& path,
    const std::vector<ReducedRCColumnSectionResponseRecord>& records)
{
    if (records.empty()) {
        return;
    }

    std::size_t controlling_gp = records.front().section_gp;
    double min_xi = records.front().xi;
    for (const auto& r : records) {
        if (r.xi < min_xi || (r.xi == min_xi && r.section_gp < controlling_gp)) {
            min_xi = r.xi;
            controlling_gp = r.section_gp;
        }
    }

    std::ofstream ofs(path);
    ofs << "step,p,drift_m,section_gp,xi,curvature_y,moment_y_MNm,"
           "axial_force_MN,tangent_eiy\n";
    ofs << std::scientific << std::setprecision(8);
    std::size_t written = 0;
    for (const auto& r : records) {
        if (r.section_gp != controlling_gp) {
            continue;
        }
        ofs << r.step << ","
            << r.p << ","
            << r.drift << ","
            << r.section_gp << ","
            << r.xi << ","
            << r.curvature_y << ","
            << r.moment_y << ","
            << r.axial_force << ","
            << r.tangent_eiy << "\n";
        ++written;
    }
    std::println(
        "  CSV: {} ({} records, base-side section_gp={}, xi={:+.6f})",
        path,
        written,
        controlling_gp,
        min_xi);
}

template <typename BeamModelT>
std::vector<ReducedRCColumnSectionResponseRecord>
extract_section_response_records(
    const BeamModelT& model,
    int step,
    double p,
    double drift)
{
    const auto& beam = model.elements().front();
    const auto u_loc = beam.local_state_vector(model.state_vector());

    std::vector<ReducedRCColumnSectionResponseRecord> records;
    records.reserve(beam.num_integration_points());

    for (std::size_t gp = 0; gp < beam.num_integration_points(); ++gp) {
        const auto xi_view = beam.geometry().reference_integration_point(gp);
        const auto strain = beam.sample_generalized_strain_at_gp(gp, u_loc);
        const auto resultant = beam.sample_resultants_at_gp(gp, u_loc);
        const auto tangent = beam.sections()[gp].tangent(strain);

        records.push_back({
            .step = step,
            .p = p,
            .drift = drift,
            .section_gp = gp,
            .xi = xi_view[0],
            .axial_strain = strain.axial_strain(),
            .curvature_y = strain.curvature_y(),
            .curvature_z = strain.curvature_z(),
            .axial_force = resultant.axial_force(),
            .moment_y = resultant.moment_y(),
            .moment_z = resultant.moment_z(),
            .tangent_ea = tangent(0, 0),
            .tangent_eiy = tangent(1, 1),
            .tangent_eiz = tangent(2, 2),
        });
    }

    return records;
}

template <typename T>
void append_records(std::vector<T>& into, const std::vector<T>& extra)
{
    into.insert(into.end(), extra.begin(), extra.end());
}

template <std::size_t N, BeamAxisQuadratureFamily QuadratureFamily>
[[nodiscard]] ReducedRCColumnStructuralRunResult
run_reduced_rc_column_small_strain_beam_case_impl(
    const ReducedRCColumnStructuralRunSpec& spec,
    const std::string& out_dir,
    const table_cyclic_validation::CyclicValidationRunConfig& cfg)
{
    using QuadratureT = BeamAxisQuadratureT<QuadratureFamily, N - 1>;
    using BeamElemT = TimoshenkoBeamN<N>;
    using BeamPolicy = SingleElementPolicy<BeamElemT>;
    using BeamModel =
        Model<TimoshenkoBeam3D, continuum::SmallStrain, NDOF, BeamPolicy>;

    Domain<3> domain;
    PetscInt tag = 0;

    for (std::size_t i = 0; i < N; ++i) {
        const double z =
            table_cyclic_validation::H *
            static_cast<double>(i) / static_cast<double>(N - 1);
        domain.add_node(static_cast<PetscInt>(i), 0.0, 0.0, z);
    }

    PetscInt conn[N];
    for (std::size_t i = 0; i < N; ++i) {
        conn[i] = static_cast<PetscInt>(i);
    }

    auto& geom = domain.template make_element<LagrangeElement3D<N>>(
        QuadratureT{}, tag++, conn);
    geom.set_physical_group("ReducedRCColumn");

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

    std::vector<BeamElemT> elements;
    elements.emplace_back(&geom, col_mat);

    BeamModel model{domain, std::move(elements)};
    model.constrain_node(0, {0, 0, 0, 0, 0, 0});

    const std::size_t top_node = N - 1;
    model.constrain_dof(top_node, 0, 0.0);
    model.setup();

    if (spec.axial_compression_force_mn != 0.0) {
        model.apply_node_force(
            top_node, 0.0, 0.0, -spec.axial_compression_force_mn, 0.0, 0.0, 0.0);
    }

    PetscOptionsSetValue(nullptr, "-snes_linesearch_type", "bt");
    PetscOptionsSetValue(nullptr, "-snes_max_it", "100");
    PetscOptionsSetValue(nullptr, "-ksp_type", "preonly");
    PetscOptionsSetValue(nullptr, "-pc_type", "lu");

    NonlinearAnalysis<TimoshenkoBeam3D, continuum::SmallStrain, NDOF, BeamPolicy>
        nl{&model};

    ReducedRCColumnStructuralRunResult result;
    result.hysteresis_records.reserve(
        static_cast<std::size_t>(cfg.total_steps()) + 1);
    result.hysteresis_records.push_back({0, 0.0, 0.0, 0.0});
    append_records(
        result.section_response_records,
        extract_section_response_records(model, 0, 0.0, 0.0));

    const std::vector<std::size_t> base_nodes = {0};

    nl.set_step_callback([&](int step, double p, const BeamModel& m) {
        const double d = cfg.displacement(p);
        const double shear = extract_base_shear_x(m, base_nodes);
        result.hysteresis_records.push_back({step, p, d, shear});
        append_records(
            result.section_response_records,
            extract_section_response_records(m, step, p, d));

        if (spec.print_progress &&
            (step % 20 == 0 || step == cfg.total_steps())) {
            const auto step_section_records =
                extract_section_response_records(m, step, p, d);
            const auto controlling = std::min_element(
                step_section_records.begin(),
                step_section_records.end(),
                [](const auto& a, const auto& b) {
                    if (a.xi == b.xi) {
                        return a.section_gp < b.section_gp;
                    }
                    return a.xi < b.xi;
                });
            std::println(
                "    reduced-column step={:3d}  p={:.4f}  d={:+.4e} m"
                "  V={:+.4e} MN  M_y={:+.4e} MNm  kappa_y={:+.4e}  quad={}",
                step,
                p,
                d,
                shear,
                controlling != step_section_records.end() ? controlling->moment_y : 0.0,
                controlling != step_section_records.end() ? controlling->curvature_y : 0.0,
                beam_axis_quadrature_family_name<QuadratureFamily>());
        }
    });

    auto scheme = make_control(
        [top_node, &cfg](double p, Vec /*f_full*/, Vec f_ext, BeamModel* m) {
            VecSet(f_ext, 0.0);
            m->update_imposed_value(top_node, 0, cfg.displacement(p));
        });

    const bool ok =
        nl.solve_incremental(cfg.total_steps(), cfg.max_bisections, scheme);

    if (spec.print_progress) {
        std::println(
            "  Reduced-column baseline result: {} ({} records, N={}, quadrature={})",
            ok ? "COMPLETED" : "ABORTED",
            result.hysteresis_records.size(),
            N,
            beam_axis_quadrature_family_name<QuadratureFamily>());
    }

    if (spec.write_hysteresis_csv) {
        std::filesystem::create_directories(out_dir);
        write_csv(out_dir + "/hysteresis.csv", result.hysteresis_records);
    }

    if (spec.write_section_response_csv) {
        std::filesystem::create_directories(out_dir);
        write_section_response_csv(
            out_dir + "/section_response.csv",
            result.section_response_records);
        write_base_side_moment_curvature_csv(
            out_dir + "/moment_curvature_base.csv",
            result.section_response_records);
    }

    return result;
}

template <std::size_t N>
[[nodiscard]] ReducedRCColumnStructuralRunResult
dispatch_small_strain_quadrature(
    BeamAxisQuadratureFamily quadrature_family,
    const ReducedRCColumnStructuralRunSpec& spec,
    const std::string& out_dir,
    const table_cyclic_validation::CyclicValidationRunConfig& cfg)
{
    switch (quadrature_family) {
        case BeamAxisQuadratureFamily::GaussLegendre:
            return run_reduced_rc_column_small_strain_beam_case_impl<
                N, BeamAxisQuadratureFamily::GaussLegendre>(spec, out_dir, cfg);
        case BeamAxisQuadratureFamily::GaussLobatto:
            return run_reduced_rc_column_small_strain_beam_case_impl<
                N, BeamAxisQuadratureFamily::GaussLobatto>(spec, out_dir, cfg);
        case BeamAxisQuadratureFamily::GaussRadauLeft:
            return run_reduced_rc_column_small_strain_beam_case_impl<
                N, BeamAxisQuadratureFamily::GaussRadauLeft>(spec, out_dir, cfg);
        case BeamAxisQuadratureFamily::GaussRadauRight:
            return run_reduced_rc_column_small_strain_beam_case_impl<
                N, BeamAxisQuadratureFamily::GaussRadauRight>(spec, out_dir, cfg);
    }

    throw std::invalid_argument("Unsupported beam-axis quadrature family.");
}

} // namespace

ReducedRCColumnStructuralRunResult
run_reduced_rc_column_small_strain_beam_case_result(
    const ReducedRCColumnStructuralRunSpec& spec,
    const std::string& out_dir,
    const table_cyclic_validation::CyclicValidationRunConfig& cfg)
{
    switch (spec.beam_nodes) {
        case 2:
            return dispatch_small_strain_quadrature<2>(
                spec.beam_axis_quadrature_family, spec, out_dir, cfg);
        case 3:
            return dispatch_small_strain_quadrature<3>(
                spec.beam_axis_quadrature_family, spec, out_dir, cfg);
        case 4:
            return dispatch_small_strain_quadrature<4>(
                spec.beam_axis_quadrature_family, spec, out_dir, cfg);
        case 5:
            return dispatch_small_strain_quadrature<5>(
                spec.beam_axis_quadrature_family, spec, out_dir, cfg);
        case 6:
            return dispatch_small_strain_quadrature<6>(
                spec.beam_axis_quadrature_family, spec, out_dir, cfg);
        case 7:
            return dispatch_small_strain_quadrature<7>(
                spec.beam_axis_quadrature_family, spec, out_dir, cfg);
        case 8:
            return dispatch_small_strain_quadrature<8>(
                spec.beam_axis_quadrature_family, spec, out_dir, cfg);
        case 9:
            return dispatch_small_strain_quadrature<9>(
                spec.beam_axis_quadrature_family, spec, out_dir, cfg);
        case 10:
            return dispatch_small_strain_quadrature<10>(
                spec.beam_axis_quadrature_family, spec, out_dir, cfg);
        default:
            throw std::invalid_argument(
                "ReducedRCColumnStructuralRunSpec supports TimoshenkoBeamN with"
                " N in [2, 10].");
    }
}

std::vector<table_cyclic_validation::StepRecord>
run_reduced_rc_column_small_strain_beam_case(
    const ReducedRCColumnStructuralRunSpec& spec,
    const std::string& out_dir,
    const table_cyclic_validation::CyclicValidationRunConfig& cfg)
{
    return run_reduced_rc_column_small_strain_beam_case_result(
               spec,
               out_dir,
               cfg)
        .hysteresis_records;
}

} // namespace fall_n::validation_reboot
