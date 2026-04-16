#include "src/validation/ReducedRCColumnStructuralBaseline.hh"

#include "src/analysis/IncrementalControl.hh"
#include "src/analysis/NLAnalysis.hh"
#include "src/domain/Domain.hh"
#include "src/elements/ElementPolicy.hh"
#include "src/elements/TimoshenkoBeamN.hh"
#include "src/elements/element_geometry/LagrangeElement.hh"
#include "src/model/Model.hh"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <print>
#include <stdexcept>

namespace fall_n::validation_reboot {

namespace {

using table_cyclic_validation::StepRecord;

inline constexpr std::size_t kReducedRCColumnNDoF = 6;

struct ReducedRCColumnControlPath {
    int lateral_steps{0};
    int axial_preload_steps{0};

    [[nodiscard]] int total_runtime_steps() const noexcept
    {
        return lateral_steps + axial_preload_steps;
    }

    [[nodiscard]] bool has_preload_stage() const noexcept
    {
        return axial_preload_steps > 0;
    }

    [[nodiscard]] double preload_completion_runtime_p() const noexcept
    {
        if (!has_preload_stage()) {
            return 0.0;
        }
        return static_cast<double>(axial_preload_steps) /
               static_cast<double>(total_runtime_steps());
    }

    [[nodiscard]] bool is_preload_runtime_step(int runtime_step) const noexcept
    {
        return has_preload_stage() && runtime_step <= axial_preload_steps;
    }

    [[nodiscard]] bool is_preload_completion_runtime_step(
        int runtime_step) const noexcept
    {
        return has_preload_stage() && runtime_step == axial_preload_steps;
    }

    [[nodiscard]] int logical_lateral_step(int runtime_step) const noexcept
    {
        if (!has_preload_stage()) {
            return runtime_step;
        }
        return std::max(runtime_step - axial_preload_steps, 0);
    }

    [[nodiscard]] double preload_progress(double runtime_p) const noexcept
    {
        if (!has_preload_stage()) {
            return 1.0;
        }
        return std::clamp(
            runtime_p / preload_completion_runtime_p(),
            0.0,
            1.0);
    }

    [[nodiscard]] double lateral_progress(double runtime_p) const noexcept
    {
        if (!has_preload_stage()) {
            return std::clamp(runtime_p, 0.0, 1.0);
        }

        if (runtime_p <= preload_completion_runtime_p()) {
            return 0.0;
        }

        const double denom = 1.0 - preload_completion_runtime_p();
        return std::clamp(
            (runtime_p - preload_completion_runtime_p()) / denom,
            0.0,
            1.0);
    }
};

void write_hysteresis_csv(
    const std::string& path,
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

template <typename ModelT>
[[nodiscard]] double extract_base_shear_x(
    const ModelT& model,
    const std::vector<std::size_t>& base_nodes)
{
    Vec f_int;
    VecDuplicate(model.state_vector(), &f_int);
    VecSet(f_int, 0.0);

    auto& mut_model = const_cast<ModelT&>(model);
    for (auto& elem : mut_model.elements()) {
        elem.compute_internal_forces(model.state_vector(), f_int);
    }
    VecAssemblyBegin(f_int);
    VecAssemblyEnd(f_int);

    double shear = 0.0;
    for (auto nid : base_nodes) {
        PetscScalar val{};
        PetscInt idx = static_cast<PetscInt>(
            model.get_domain().node(nid).dof_index()[0]);
        VecGetValues(f_int, 1, &idx, &val);
        shear += val;
    }

    VecDestroy(&f_int);
    return shear;
}

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

template <typename BeamModelT>
void append_runtime_observables(
    ReducedRCColumnStructuralRunResult& result,
    const BeamModelT& model,
    const std::vector<std::size_t>& base_nodes,
    int logical_step,
    double logical_p,
    double drift)
{
    const double shear = extract_base_shear_x(model, base_nodes);
    result.hysteresis_records.push_back({logical_step, logical_p, drift, shear});
    append_records(
        result.section_response_records,
        extract_section_response_records(model, logical_step, logical_p, drift));
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
        Model<TimoshenkoBeam3D, continuum::SmallStrain, kReducedRCColumnNDoF, BeamPolicy>;

    const auto& reference_spec = spec.reference_spec;
    const auto section_spec = to_rc_column_section_spec(reference_spec);

    Domain<3> domain;
    PetscInt tag = 0;

    for (std::size_t i = 0; i < N; ++i) {
        const double z =
            reference_spec.column_height_m *
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

    const auto col_mat = make_rc_column_section(section_spec);

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

    NonlinearAnalysis<TimoshenkoBeam3D, continuum::SmallStrain, kReducedRCColumnNDoF, BeamPolicy>
        nl{&model};

    const ReducedRCColumnControlPath control_path{
        .lateral_steps = cfg.total_steps(),
        .axial_preload_steps =
            spec.uses_equilibrated_axial_preload_stage() ? spec.axial_preload_steps : 0};

    if (control_path.total_runtime_steps() <= 0) {
        throw std::invalid_argument(
            "Reduced structural column baseline requires a strictly positive "
            "number of lateral or preload runtime steps.");
    }

    ReducedRCColumnStructuralRunResult result;
    result.hysteresis_records.reserve(
        static_cast<std::size_t>(cfg.total_steps()) + 1);
    if (!control_path.has_preload_stage()) {
        result.hysteresis_records.push_back({0, 0.0, 0.0, 0.0});
        append_records(
            result.section_response_records,
            extract_section_response_records(model, 0, 0.0, 0.0));
    }

    const std::vector<std::size_t> base_nodes = {0};

    nl.set_step_callback([&](
                             int runtime_step,
                             double runtime_p,
                             const BeamModel& m) {
        if (control_path.is_preload_runtime_step(runtime_step) &&
            !control_path.is_preload_completion_runtime_step(runtime_step)) {
            return;
        }

        const double logical_p = control_path.lateral_progress(runtime_p);
        const double drift = cfg.displacement(logical_p);
        const int logical_step = control_path.logical_lateral_step(runtime_step);

        append_runtime_observables(
            result,
            m,
            base_nodes,
            logical_step,
            logical_p,
            drift);

        const bool report_step =
            logical_step == 0 ||
            (logical_step % 20 == 0) ||
            (logical_step == cfg.total_steps());

        if (spec.print_progress && report_step) {
            const auto step_section_records =
                extract_section_response_records(m, logical_step, logical_p, drift);
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
                "  V={:+.4e} MN  M_y={:+.4e} MNm  kappa_y={:+.4e}"
                "  quad={}  stage={}",
                logical_step,
                logical_p,
                drift,
                result.hysteresis_records.back().base_shear,
                controlling != step_section_records.end() ? controlling->moment_y : 0.0,
                controlling != step_section_records.end() ? controlling->curvature_y : 0.0,
                beam_axis_quadrature_family_name<QuadratureFamily>(),
                control_path.is_preload_completion_runtime_step(runtime_step)
                    ? "preload_equilibrated"
                    : "lateral_branch");
        }
    });

    auto scheme = make_control(
        [top_node, &cfg, control_path](
            double runtime_p, Vec f_full, Vec f_ext, BeamModel* m) {
            const double lateral_p = control_path.lateral_progress(runtime_p);

            if (control_path.has_preload_stage() &&
                runtime_p <= control_path.preload_completion_runtime_p()) {
                VecCopy(f_full, f_ext);
                VecScale(f_ext, control_path.preload_progress(runtime_p));
                m->update_imposed_value(top_node, 0, 0.0);
                return;
            }

            VecCopy(f_full, f_ext);
            m->update_imposed_value(top_node, 0, cfg.displacement(lateral_p));
        });

    const bool ok =
        nl.solve_incremental(
            control_path.total_runtime_steps(),
            cfg.max_bisections,
            scheme);

    if (spec.print_progress) {
        std::println(
            "  Reduced-column baseline result: {} ({} logical records,"
            " N={}, quadrature={}, preload_stage={})",
            ok ? "COMPLETED" : "ABORTED",
            result.hysteresis_records.size(),
            N,
            beam_axis_quadrature_family_name<QuadratureFamily>(),
            control_path.has_preload_stage() ? "enabled" : "disabled");
    }

    if (spec.write_hysteresis_csv) {
        std::filesystem::create_directories(out_dir);
        write_hysteresis_csv(
            out_dir + "/hysteresis.csv", result.hysteresis_records);
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
