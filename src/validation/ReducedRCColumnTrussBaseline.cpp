#include "src/validation/ReducedRCColumnTrussBaseline.hh"

#include "src/domain/Domain.hh"
#include "src/elements/TrussElement.hh"
#include "src/materials/FiberSectionFactory.hh"
#include "src/numerics/numerical_integration/GaussLegendreCellIntegrator.hh"
#include "src/utils/Benchmark.hh"
#include "src/validation/CyclicMaterialDriver.hh"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numbers>
#include <print>
#include <ranges>
#include <stdexcept>

namespace fall_n::validation_reboot {

namespace {

using ProtocolPoint = ReducedRCColumnMaterialBaselineRunSpec::ProtocolPoint;
using StrainPoint = fall_n::cyclic_driver::StrainPoint;

[[nodiscard]] double default_length_m(
    const ReducedRCColumnTrussBaselineRunSpec& spec) noexcept
{
    return resolve_reduced_rc_column_truss_length_m(spec);
}

[[nodiscard]] double default_area_m2(
    const ReducedRCColumnTrussBaselineRunSpec& spec) noexcept
{
    return resolve_reduced_rc_column_truss_area_m2(spec);
}

[[nodiscard]] double default_monotonic_target_strain(
    const ReducedRCColumnTrussBaselineRunSpec& spec) noexcept
{
    if (std::abs(spec.monotonic_target_strain) > 0.0) {
        return -std::abs(spec.monotonic_target_strain);
    }

    const double ey =
        spec.reference_spec.steel_fy_mpa / spec.reference_spec.steel_E_mpa;
    return -2.0 * ey;
}

[[nodiscard]] std::vector<double> default_compression_amplitude_levels(
    const ReducedRCColumnTrussBaselineRunSpec& spec)
{
    if (!spec.compression_amplitude_levels.empty()) {
        return spec.compression_amplitude_levels;
    }

    const double ey =
        spec.reference_spec.steel_fy_mpa / spec.reference_spec.steel_E_mpa;
    return {0.25 * ey, 0.5 * ey, 1.0 * ey, 2.0 * ey};
}

[[nodiscard]] std::vector<StrainPoint> make_monotonic_compression_protocol(
    double target_strain,
    int steps_per_branch)
{
    std::vector<StrainPoint> protocol;
    protocol.reserve(static_cast<std::size_t>(steps_per_branch));
    for (int step = 1; step <= steps_per_branch; ++step) {
        const double t =
            static_cast<double>(step) / static_cast<double>(steps_per_branch);
        protocol.push_back(StrainPoint{
            .step = step,
            .strain = t * target_strain,
        });
    }
    return protocol;
}

[[nodiscard]] std::vector<StrainPoint> to_driver_protocol(
    const std::vector<ProtocolPoint>& custom_protocol)
{
    std::vector<StrainPoint> protocol;
    protocol.reserve(custom_protocol.size());
    for (const auto& point : custom_protocol) {
        protocol.push_back(StrainPoint{
            .step = point.step,
            .strain = point.strain,
        });
    }
    return protocol;
}

void validate_custom_protocol(const std::vector<ProtocolPoint>& custom_protocol)
{
    int previous_step = -1;
    for (const auto& point : custom_protocol) {
        if (point.step <= previous_step) {
            throw std::invalid_argument(
                "Reduced RC truss baseline custom protocol requires strictly increasing step ids.");
        }
        if (!std::isfinite(point.strain)) {
            throw std::invalid_argument(
                "Reduced RC truss baseline custom protocol requires finite strains.");
        }
        previous_step = point.step;
    }
}

[[nodiscard]] std::vector<StrainPoint> make_protocol(
    const ReducedRCColumnTrussBaselineRunSpec& spec)
{
    if (!spec.custom_protocol.empty()) {
        validate_custom_protocol(spec.custom_protocol);
        return to_driver_protocol(spec.custom_protocol);
    }

    switch (spec.protocol_kind) {
        case ReducedRCColumnTrussProtocolKind::monotonic_compression:
            return make_monotonic_compression_protocol(
                default_monotonic_target_strain(spec),
                spec.steps_per_branch);
        case ReducedRCColumnTrussProtocolKind::cyclic_compression_return:
            return fall_n::cyclic_driver::make_compression_return_protocol(
                default_compression_amplitude_levels(spec),
                spec.steps_per_branch);
    }
    throw std::invalid_argument(
        "Reduced RC truss baseline received an unknown protocol kind.");
}

struct QuadraticTrussFixture {
    Domain<3> domain;
    TrussElement<3, 3> element;

    QuadraticTrussFixture(
        double length_m,
        double area_m2,
        Material<UniaxialMaterial> material)
        : element{
              [&]() -> ElementGeometry<3>* {
                  domain.add_node(0, 0.0, 0.0, 0.0);
                  domain.add_node(1, 0.5 * length_m, 0.0, 0.0);
                  domain.add_node(2, length_m, 0.0, 0.0);
                  PetscInt conn[3] = {0, 1, 2};
                  domain.make_element<LagrangeElement3D<3>>(
                      GaussLegendreCellIntegrator<3>{},
                      0,
                      conn);
                  domain.assemble_sieve();
                  return &domain.element(0);
              }(),
              std::move(material),
              area_m2}
    {}
};

[[nodiscard]] Eigen::VectorXd make_uniform_axial_local_state(
    double length_m,
    double axial_strain)
{
    Eigen::VectorXd u_e = Eigen::VectorXd::Zero(9);
    const double end_disp = axial_strain * length_m;
    u_e[3] = 0.5 * end_disp;
    u_e[6] = end_disp;
    return u_e;
}

[[nodiscard]] Eigen::VectorXd make_uniform_axial_control_direction()
{
    Eigen::VectorXd control = Eigen::VectorXd::Zero(9);
    control[3] = 0.5;
    control[6] = 1.0;
    return control;
}

[[nodiscard]] double compute_rms(const std::vector<double>& values)
{
    if (values.empty()) {
        return 0.0;
    }
    double sum_sq = 0.0;
    for (const double value : values) {
        sum_sq += value * value;
    }
    return std::sqrt(sum_sq / static_cast<double>(values.size()));
}

void write_protocol_csv(
    const std::string& path,
    const std::vector<StrainPoint>& protocol)
{
    std::ofstream out(path);
    out << "step,strain\n";
    out << std::scientific << std::setprecision(8);
    for (const auto& point : protocol) {
        out << point.step << "," << point.strain << "\n";
    }
}

void write_material_response_csv(
    const std::string& path,
    const std::vector<ReducedRCColumnMaterialBaselineRecord>& records)
{
    std::ofstream out(path);
    out << "step,strain,stress_MPa,tangent_MPa,energy_density_MPa\n";
    out << std::scientific << std::setprecision(8);
    for (const auto& record : records) {
        out << record.step << ","
            << record.strain << ","
            << record.stress_mpa << ","
            << record.tangent_mpa << ","
            << record.energy_density_mpa << "\n";
    }
}

void write_truss_response_csv(
    const std::string& path,
    const std::vector<ReducedRCColumnTrussBaselineRecord>& records)
{
    std::ofstream out(path);
    out << "step,axial_strain,axial_stress_MPa,tangent_MPa,"
           "tangent_from_element_MPa,end_displacement_m,axial_force_MN,"
           "middle_node_force_MN,energy_density_MPa,truss_work_MN_m,"
           "gp_strain_spread,gp_stress_spread_MPa,gp_tangent_spread_MPa\n";
    out << std::scientific << std::setprecision(8);
    for (const auto& record : records) {
        out << record.step << ","
            << record.axial_strain << ","
            << record.axial_stress_mpa << ","
            << record.tangent_mpa << ","
            << record.tangent_from_element_mpa << ","
            << record.end_displacement_m << ","
            << record.axial_force_mn << ","
            << record.middle_node_force_mn << ","
            << record.energy_density_mpa << ","
            << record.truss_work_mn_m << ","
            << record.gp_strain_spread << ","
            << record.gp_stress_spread_mpa << ","
            << record.gp_tangent_spread_mpa << "\n";
    }
}

void write_truss_gauss_response_csv(
    const std::string& path,
    const std::vector<ReducedRCColumnTrussGaussRecord>& records)
{
    std::ofstream out(path);
    out << "step,gauss_point,xi,axial_strain,axial_stress_MPa,tangent_MPa\n";
    out << std::scientific << std::setprecision(8);
    for (const auto& record : records) {
        out << record.step << ","
            << record.gauss_point << ","
            << record.xi << ","
            << record.axial_strain << ","
            << record.axial_stress_mpa << ","
            << record.tangent_mpa << "\n";
    }
}

void print_progress(
    const ReducedRCColumnTrussBaselineRunSpec& spec,
    const std::vector<ReducedRCColumnTrussBaselineRecord>& records)
{
    if (!spec.print_progress || records.empty()) {
        return;
    }

    const auto every_n = std::max(
        1,
        static_cast<int>(records.size() / 5));
    for (std::size_t i = 0; i < records.size(); ++i) {
        const auto& record = records[i];
        if (i == 0 || i + 1 == records.size() ||
            static_cast<int>(i) % every_n == 0) {
            std::println(
                "    reduced-truss {:>24s}  step={:4d}  eps={:+.4e}  sig={:+.4e} MPa  "
                "Et={:+.4e} MPa  gp_spread={:.3e}",
                to_string(spec.protocol_kind),
                record.step,
                record.axial_strain,
                record.axial_stress_mpa,
                record.tangent_mpa,
                record.gp_stress_spread_mpa);
        }
    }
}

} // namespace

ReducedRCColumnTrussBaselineResult run_reduced_rc_column_truss_baseline(
    const ReducedRCColumnTrussBaselineRunSpec& spec,
    const std::string& out_dir)
{
    if (spec.custom_protocol.empty() && spec.steps_per_branch <= 0) {
        throw std::invalid_argument(
            "Reduced RC truss baseline requires steps_per_branch > 0.");
    }

    const auto protocol = make_protocol(spec);
    const double length_m = default_length_m(spec);
    const double area_m2 = default_area_m2(spec);
    std::vector<ProtocolPoint> material_protocol;
    material_protocol.reserve(protocol.size());
    for (const auto& point : protocol) {
        material_protocol.push_back(
            ProtocolPoint{
                .step = point.step,
                .strain = point.strain,
            });
    }

    StopWatch total_timer;
    total_timer.start();
    StopWatch analysis_timer;
    analysis_timer.start();

    const auto material_result = run_reduced_rc_column_material_baseline(
        ReducedRCColumnMaterialBaselineRunSpec{
            .reference_spec = spec.reference_spec,
            .material_kind = ReducedRCColumnMaterialReferenceKind::steel_rebar,
            .protocol_kind =
                spec.protocol_kind ==
                        ReducedRCColumnTrussProtocolKind::monotonic_compression
                    ? ReducedRCColumnMaterialProtocolKind::monotonic
                    : ReducedRCColumnMaterialProtocolKind::cyclic,
            .monotonic_target_strain = default_monotonic_target_strain(spec),
            .amplitude_levels = {},
            .custom_protocol = material_protocol,
            .steps_per_branch = spec.steps_per_branch,
            .write_csv = false,
            .print_progress = false,
        },
        out_dir);

    QuadraticTrussFixture fixture{
        length_m,
        area_m2,
        make_steel_fiber_material(
            spec.reference_spec.steel_E_mpa,
            spec.reference_spec.steel_fy_mpa,
            spec.reference_spec.steel_b)};

    std::vector<ReducedRCColumnTrussBaselineRecord> truss_records;
    std::vector<ReducedRCColumnTrussGaussRecord> gauss_records;
    truss_records.reserve(protocol.size() + 1);
    gauss_records.reserve(
        (protocol.size() + 1) *
        fixture.element.num_integration_points());

    {
        const Eigen::VectorXd u0 = Eigen::VectorXd::Zero(9);
        const auto fields = fixture.element.collect_gauss_fields(u0);
        for (std::size_t gp = 0; gp < fields.size(); ++gp) {
            gauss_records.push_back(
                ReducedRCColumnTrussGaussRecord{
                    .step = 0,
                    .gauss_point = static_cast<int>(gp),
                    .xi =
                        fixture.element.geometry().reference_integration_point(gp)[0],
                    .axial_strain = fields[gp].strain[0],
                    .axial_stress_mpa = fields[gp].stress[0],
                    .tangent_mpa = fixture.element.materials()[gp]
                                       .tangent(Strain<1>{fields[gp].strain[0]})(0, 0),
                });
        }
        truss_records.push_back(
            ReducedRCColumnTrussBaselineRecord{
                .step = 0,
                .axial_strain = 0.0,
                .axial_stress_mpa = 0.0,
                .tangent_mpa = spec.reference_spec.steel_E_mpa,
                .tangent_from_element_mpa = spec.reference_spec.steel_E_mpa,
                .end_displacement_m = 0.0,
                .axial_force_mn = 0.0,
                .middle_node_force_mn = 0.0,
                .energy_density_mpa = 0.0,
                .truss_work_mn_m = 0.0,
                .gp_strain_spread = 0.0,
                .gp_stress_spread_mpa = 0.0,
                .gp_tangent_spread_mpa = 0.0,
            });
    }

    double previous_end_displacement_m = 0.0;
    double previous_axial_force_mn = 0.0;
    double truss_work_mn_m = 0.0;
    const Eigen::VectorXd control_direction =
        make_uniform_axial_control_direction();

    for (const auto& point : protocol) {
        const Eigen::VectorXd u_e =
            make_uniform_axial_local_state(length_m, point.strain);
        const auto internal_force =
            fixture.element.compute_internal_force_vector(u_e);
        const auto tangent =
            fixture.element.compute_tangent_stiffness_matrix(u_e);
        const auto fields = fixture.element.collect_gauss_fields(u_e);

        std::vector<double> gp_strains;
        std::vector<double> gp_stresses;
        std::vector<double> gp_tangents;
        gp_strains.reserve(fields.size());
        gp_stresses.reserve(fields.size());
        gp_tangents.reserve(fields.size());

        for (std::size_t gp = 0; gp < fields.size(); ++gp) {
            const double strain = fields[gp].strain[0];
            const double stress_mpa = fields[gp].stress[0];
            const double tangent_mpa =
                fixture.element.materials()[gp]
                    .tangent(Strain<1>{strain})(0, 0);
            gp_strains.push_back(strain);
            gp_stresses.push_back(stress_mpa);
            gp_tangents.push_back(tangent_mpa);

            gauss_records.push_back(
                ReducedRCColumnTrussGaussRecord{
                    .step = point.step,
                    .gauss_point = static_cast<int>(gp),
                    .xi =
                        fixture.element.geometry().reference_integration_point(gp)[0],
                    .axial_strain = strain,
                    .axial_stress_mpa = stress_mpa,
                    .tangent_mpa = tangent_mpa,
                });
        }

        const auto [min_strain_it, max_strain_it] =
            std::minmax_element(gp_strains.begin(), gp_strains.end());
        const auto [min_stress_it, max_stress_it] =
            std::minmax_element(gp_stresses.begin(), gp_stresses.end());
        const auto [min_tangent_it, max_tangent_it] =
            std::minmax_element(gp_tangents.begin(), gp_tangents.end());

        const double gp_strain_spread =
            gp_strains.empty() ? 0.0 : (*max_strain_it - *min_strain_it);
        const double gp_stress_spread_mpa =
            gp_stresses.empty() ? 0.0 : (*max_stress_it - *min_stress_it);
        const double gp_tangent_spread_mpa =
            gp_tangents.empty() ? 0.0 : (*max_tangent_it - *min_tangent_it);

        const double end_displacement_m = u_e[6];
        const double axial_force_mn = internal_force[6];
        const double middle_node_force_mn = internal_force[3];
        // Project the quadratic element tangent onto the affine control path
        // u = delta * [0, 0, 0, 0.5, 0, 0, 1, 0, 0]^T.
        // For a constant-strain quadratic bar, dF_end / ddelta projected along
        // that path must recover A * Et / L. Multiplying by L/A maps the
        // projected element stiffness back to the equivalent uniaxial tangent.
        const double tangent_from_element_mpa =
            tangent.row(6).dot(control_direction) * length_m / area_m2;

        truss_work_mn_m +=
            0.5 * (axial_force_mn + previous_axial_force_mn) *
            (end_displacement_m - previous_end_displacement_m);
        const double truss_energy_density_mpa =
            truss_work_mn_m / (area_m2 * length_m);

        truss_records.push_back(
            ReducedRCColumnTrussBaselineRecord{
                .step = point.step,
                .axial_strain = point.strain,
                .axial_stress_mpa = gp_stresses.empty() ? 0.0 : gp_stresses.front(),
                .tangent_mpa = gp_tangents.empty() ? 0.0 : gp_tangents.front(),
                .tangent_from_element_mpa = tangent_from_element_mpa,
                .end_displacement_m = end_displacement_m,
                .axial_force_mn = axial_force_mn,
                .middle_node_force_mn = middle_node_force_mn,
                .energy_density_mpa = truss_energy_density_mpa,
                .truss_work_mn_m = truss_work_mn_m,
                .gp_strain_spread = gp_strain_spread,
                .gp_stress_spread_mpa = gp_stress_spread_mpa,
                .gp_tangent_spread_mpa = gp_tangent_spread_mpa,
            });

        previous_end_displacement_m = end_displacement_m;
        previous_axial_force_mn = axial_force_mn;
        fixture.element.commit_material_state(u_e);
    }

    const double analysis_wall_seconds = analysis_timer.stop();

    if (material_result.records.size() != truss_records.size()) {
        throw std::runtime_error(
            "Reduced RC truss baseline expected the direct material path and the truss path to produce the same record count.");
    }

    std::vector<double> stress_errors;
    std::vector<double> tangent_errors;
    std::vector<double> element_tangent_errors;
    std::vector<double> energy_errors;
    stress_errors.reserve(truss_records.size());
    tangent_errors.reserve(truss_records.size());
    element_tangent_errors.reserve(truss_records.size());
    energy_errors.reserve(truss_records.size());

    ReducedRCColumnTrussBaselineComparisonSummary comparison{};
    for (std::size_t i = 0; i < truss_records.size(); ++i) {
        const double stress_error = std::abs(
            truss_records[i].axial_stress_mpa - material_result.records[i].stress_mpa);
        const double tangent_error = std::abs(
            truss_records[i].tangent_mpa - material_result.records[i].tangent_mpa);
        const double element_tangent_error = std::abs(
            truss_records[i].tangent_from_element_mpa -
            material_result.records[i].tangent_mpa);
        const double energy_error = std::abs(
            truss_records[i].energy_density_mpa -
            material_result.records[i].energy_density_mpa);
        const double axial_force_closure = std::abs(
            truss_records[i].axial_force_mn -
            truss_records[i].axial_stress_mpa * area_m2);

        stress_errors.push_back(stress_error);
        tangent_errors.push_back(tangent_error);
        element_tangent_errors.push_back(element_tangent_error);
        energy_errors.push_back(energy_error);

        comparison.max_abs_stress_error_mpa =
            std::max(comparison.max_abs_stress_error_mpa, stress_error);
        comparison.max_abs_tangent_error_mpa =
            std::max(comparison.max_abs_tangent_error_mpa, tangent_error);
        comparison.max_abs_element_tangent_error_mpa =
            std::max(
                comparison.max_abs_element_tangent_error_mpa,
                element_tangent_error);
        comparison.max_abs_energy_density_error_mpa =
            std::max(comparison.max_abs_energy_density_error_mpa, energy_error);
        comparison.max_abs_axial_force_closure_mn =
            std::max(
                comparison.max_abs_axial_force_closure_mn,
                axial_force_closure);
        comparison.max_abs_gp_strain_spread =
            std::max(comparison.max_abs_gp_strain_spread,
                     truss_records[i].gp_strain_spread);
        comparison.max_abs_gp_stress_spread_mpa =
            std::max(comparison.max_abs_gp_stress_spread_mpa,
                     truss_records[i].gp_stress_spread_mpa);
        comparison.max_abs_gp_tangent_spread_mpa =
            std::max(comparison.max_abs_gp_tangent_spread_mpa,
                     truss_records[i].gp_tangent_spread_mpa);
        comparison.max_abs_middle_node_force_mn =
            std::max(comparison.max_abs_middle_node_force_mn,
                     std::abs(truss_records[i].middle_node_force_mn));
    }
    comparison.rms_abs_stress_error_mpa = compute_rms(stress_errors);
    comparison.rms_abs_tangent_error_mpa = compute_rms(tangent_errors);
    comparison.rms_abs_element_tangent_error_mpa =
        compute_rms(element_tangent_errors);
    comparison.rms_abs_energy_density_error_mpa = compute_rms(energy_errors);

    print_progress(spec, truss_records);

    StopWatch output_timer;
    output_timer.start();
    if (spec.write_csv) {
        std::filesystem::create_directories(out_dir);
        write_protocol_csv(out_dir + "/protocol.csv", protocol);
        write_material_response_csv(
            out_dir + "/material_response.csv",
            material_result.records);
        write_truss_response_csv(
            out_dir + "/truss_response.csv",
            truss_records);
        write_truss_gauss_response_csv(
            out_dir + "/truss_gauss_response.csv",
            gauss_records);
    }

    const double output_write_wall_seconds = output_timer.stop();
    const double total_wall_seconds = total_timer.stop();

    return ReducedRCColumnTrussBaselineResult{
        .material_records = material_result.records,
        .truss_records = std::move(truss_records),
        .gauss_records = std::move(gauss_records),
        .comparison = comparison,
        .timing =
            ReducedRCColumnTrussBaselineTimingSummary{
                .total_wall_seconds = total_wall_seconds,
                .analysis_wall_seconds = analysis_wall_seconds,
                .output_write_wall_seconds = output_write_wall_seconds,
            },
        .completed_successfully = true,
    };
}

} // namespace fall_n::validation_reboot
