#include "src/validation/ReducedRCColumnSectionBaseline.hh"

#include "src/materials/beam/BeamGeneralizedStrain.hh"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <print>
#include <stdexcept>

namespace fall_n::validation_reboot {

namespace {

using BeamSectionStrain = BeamGeneralizedStrain<6, 3>;

[[nodiscard]] double signed_target_axial_force_mn(
    double target_axial_compression_force_mn) noexcept
{
    return -std::abs(target_axial_compression_force_mn);
}

void write_section_baseline_csv(
    const std::string& path,
    const std::vector<ReducedRCColumnSectionBaselineRecord>& records)
{
    std::ofstream ofs(path);
    ofs << "step,load_factor,target_axial_force_MN,solved_axial_strain,"
           "curvature_y,curvature_z,axial_force_MN,moment_y_MNm,moment_z_MNm,"
           "tangent_ea,tangent_eiy,tangent_eiz,newton_iterations,"
           "final_axial_force_residual_MN\n";
    ofs << std::scientific << std::setprecision(8);

    for (const auto& r : records) {
        ofs << r.step << ","
            << r.load_factor << ","
            << r.target_axial_force << ","
            << r.solved_axial_strain << ","
            << r.curvature_y << ","
            << r.curvature_z << ","
            << r.axial_force << ","
            << r.moment_y << ","
            << r.moment_z << ","
            << r.tangent_ea << ","
            << r.tangent_eiy << ","
            << r.tangent_eiz << ","
            << r.newton_iterations << ","
            << r.final_axial_force_residual << "\n";
    }

    std::println("  CSV: {} ({} records)", path, records.size());
}

[[nodiscard]] ReducedRCColumnSectionBaselineRecord solve_section_step(
    Material<TimoshenkoBeam3D>& section_material,
    const ReducedRCColumnSectionBaselineRunSpec& spec,
    int step,
    double load_factor,
    double starting_axial_strain)
{
    const double target_axial_force =
        signed_target_axial_force_mn(spec.target_axial_compression_force_mn);
    const double curvature_y = load_factor * spec.max_curvature_y;

    BeamSectionStrain strain{};
    strain[1] = curvature_y;
    strain[2] = 0.0;

    double axial_strain = starting_axial_strain;
    int iterations = 0;
    double residual = 0.0;
    BeamSectionForces<6> response{};
    Eigen::Matrix<double, 6, 6> tangent = Eigen::Matrix<double, 6, 6>::Zero();

    for (iterations = 1; iterations <= spec.axial_force_newton_max_iterations;
         ++iterations) {
        strain[0] = axial_strain;
        section_material.update_state(strain);
        response = section_material.compute_response(strain);
        tangent = section_material.tangent(strain);

        residual = response.axial_force() - target_axial_force;
        if (std::abs(residual) <= spec.axial_force_newton_tolerance_mn) {
            break;
        }

        const double tangent_ea = tangent(0, 0);
        if (!std::isfinite(tangent_ea) || std::abs(tangent_ea) < 1.0e-12) {
            throw std::runtime_error(
                "Reduced RC section baseline encountered a singular axial tangent "
                "while solving section-level axial equilibrium.");
        }

        axial_strain -= residual / tangent_ea;
    }

    if (std::abs(residual) > spec.axial_force_newton_tolerance_mn) {
        throw std::runtime_error(
            "Reduced RC section baseline did not converge the axial-force "
            "equilibrium solve within the configured Newton budget.");
    }

    section_material.commit(strain);

    return ReducedRCColumnSectionBaselineRecord{
        .step = step,
        .load_factor = load_factor,
        .target_axial_force = target_axial_force,
        .solved_axial_strain = axial_strain,
        .curvature_y = strain[1],
        .curvature_z = strain[2],
        .axial_force = response.axial_force(),
        .moment_y = response.moment_y(),
        .moment_z = response.moment_z(),
        .tangent_ea = tangent(0, 0),
        .tangent_eiy = tangent(1, 1),
        .tangent_eiz = tangent(2, 2),
        .newton_iterations = iterations,
        .final_axial_force_residual = residual,
    };
}

} // namespace

ReducedRCColumnSectionBaselineResult
run_reduced_rc_column_section_moment_curvature_baseline(
    const ReducedRCColumnSectionBaselineRunSpec& spec,
    const std::string& out_dir)
{
    if (spec.steps <= 0) {
        throw std::invalid_argument(
            "Reduced RC section baseline requires a strictly positive number of "
            "steps.");
    }

    auto section_material =
        make_rc_column_section(to_rc_column_section_spec(spec.reference_spec));

    ReducedRCColumnSectionBaselineResult result;
    result.records.reserve(static_cast<std::size_t>(spec.steps) + 1);

    const auto initial_record = solve_section_step(
        section_material,
        spec,
        0,
        0.0,
        0.0);
    result.records.push_back(initial_record);

    double starting_axial_strain = initial_record.solved_axial_strain;
    for (int step = 1; step <= spec.steps; ++step) {
        const double load_factor =
            static_cast<double>(step) / static_cast<double>(spec.steps);

        const auto record = solve_section_step(
            section_material,
            spec,
            step,
            load_factor,
            starting_axial_strain);

        starting_axial_strain = record.solved_axial_strain;
        result.records.push_back(record);

        if (spec.print_progress &&
            (step % 20 == 0 || step == spec.steps)) {
            std::println(
                "    reduced-section step={:3d}  kappa_y={:+.4e}  M_y={:+.4e} MNm"
                "  N={:+.4e} MN  eps0={:+.4e}  newton={}  residual={:+.4e}",
                record.step,
                record.curvature_y,
                record.moment_y,
                record.axial_force,
                record.solved_axial_strain,
                record.newton_iterations,
                record.final_axial_force_residual);
        }
    }

    if (spec.write_csv) {
        std::filesystem::create_directories(out_dir);
        write_section_baseline_csv(
            out_dir + "/section_moment_curvature_baseline.csv",
            result.records);
    }

    return result;
}

} // namespace fall_n::validation_reboot
