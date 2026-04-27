#include "src/validation/ReducedRCColumnMaterialBaseline.hh"

#include "src/utils/Benchmark.hh"
#include "src/validation/CyclicMaterialDriver.hh"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <print>
#include <ranges>
#include <stdexcept>

namespace fall_n::validation_reboot {

namespace {

using MaterialKind = ReducedRCColumnMaterialReferenceKind;
using ProtocolKind = ReducedRCColumnMaterialProtocolKind;
using StrainPoint = fall_n::cyclic_driver::StrainPoint;
using UniaxialRecord = fall_n::cyclic_driver::UniaxialRecord;

[[nodiscard]] KentParkConcreteTensionConfig
to_concrete_tension_config(const ReducedRCColumnReferenceSpec& spec) noexcept
{
    return KentParkConcreteTensionConfig{
        .tensile_strength = spec.concrete_ft_ratio * spec.concrete_fpc_mpa,
        .softening_multiplier = spec.concrete_tension_softening_multiplier,
        .residual_tangent_ratio = spec.concrete_tension_residual_tangent_ratio,
        .crack_transition_multiplier =
            spec.concrete_tension_transition_multiplier,
    };
}

[[nodiscard]] double default_monotonic_target_strain(
    const ReducedRCColumnMaterialBaselineRunSpec& spec) noexcept
{
    if (std::abs(spec.monotonic_target_strain) > 0.0) {
        return spec.monotonic_target_strain;
    }

    return spec.material_kind == MaterialKind::steel_rebar ? 0.03 : -0.006;
}

[[nodiscard]] std::vector<double> default_amplitude_levels(
    const ReducedRCColumnMaterialBaselineRunSpec& spec)
{
    if (!spec.amplitude_levels.empty()) {
        return spec.amplitude_levels;
    }

    if (spec.material_kind == MaterialKind::steel_rebar) {
        const double ey =
            spec.reference_spec.steel_fy_mpa / spec.reference_spec.steel_E_mpa;
        return {0.5 * ey, 1.0 * ey};
    }

    return {0.001, 0.002, 0.003, 0.004};
}

[[nodiscard]] std::vector<StrainPoint> make_monotonic_protocol(
    double target_strain,
    int steps)
{
    std::vector<StrainPoint> protocol;
    protocol.reserve(static_cast<std::size_t>(steps));
    for (int step = 1; step <= steps; ++step) {
        const double load_factor =
            static_cast<double>(step) / static_cast<double>(steps);
        protocol.push_back(StrainPoint{
            .step = step,
            .strain = load_factor * target_strain,
        });
    }
    return protocol;
}

[[nodiscard]] std::vector<StrainPoint> to_driver_protocol(
    const std::vector<ReducedRCColumnMaterialBaselineRunSpec::ProtocolPoint>&
        custom_protocol)
{
    std::vector<StrainPoint> protocol;
    protocol.reserve(custom_protocol.size());
    std::ranges::transform(
        custom_protocol,
        std::back_inserter(protocol),
        [](const auto& point) {
            return StrainPoint{
                .step = point.step,
                .strain = point.strain,
            };
        });
    return protocol;
}

void validate_custom_protocol(
    const std::vector<ReducedRCColumnMaterialBaselineRunSpec::ProtocolPoint>&
        custom_protocol)
{
    if (custom_protocol.empty()) {
        return;
    }

    int previous_step = -1;
    for (const auto& point : custom_protocol) {
        if (point.step <= previous_step) {
            throw std::invalid_argument(
                "Reduced RC material baseline custom protocol requires strictly increasing step ids.");
        }
        if (!std::isfinite(point.strain)) {
            throw std::invalid_argument(
                "Reduced RC material baseline custom protocol requires finite strains.");
        }
        previous_step = point.step;
    }
}

[[nodiscard]] std::vector<ReducedRCColumnMaterialBaselineRecord>
to_baseline_records(const std::vector<UniaxialRecord>& records)
{
    std::vector<ReducedRCColumnMaterialBaselineRecord> baseline;
    baseline.reserve(records.size());
    std::ranges::transform(
        records,
        std::back_inserter(baseline),
        [](const UniaxialRecord& record) {
            return ReducedRCColumnMaterialBaselineRecord{
                .step = record.step,
                .strain = record.strain,
                .stress_mpa = record.stress,
                .tangent_mpa = record.tangent,
                .energy_density_mpa = record.energy_density,
            };
        });
    return baseline;
}

void write_material_baseline_csv(
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

[[nodiscard]] std::vector<ReducedRCColumnMaterialBaselineRecord>
run_steel_history(const ReducedRCColumnMaterialBaselineRunSpec& spec)
{
    const auto protocol = !spec.custom_protocol.empty()
                              ? to_driver_protocol(spec.custom_protocol)
                              : (spec.protocol_kind == ProtocolKind::monotonic
                                     ? make_monotonic_protocol(
                                           std::abs(
                                               default_monotonic_target_strain(
                                                   spec)),
                                           spec.steps_per_branch)
                                     : fall_n::cyclic_driver::
                                           make_symmetric_cyclic_protocol(
                                               default_amplitude_levels(spec),
                                               spec.steps_per_branch));

    const auto result = fall_n::cyclic_driver::drive_menegotto_pinto_cyclic(
        spec.reference_spec.steel_E_mpa,
        spec.reference_spec.steel_fy_mpa,
        spec.reference_spec.steel_b,
        protocol);

    return to_baseline_records(result.records);
}

[[nodiscard]] std::vector<ReducedRCColumnMaterialBaselineRecord>
run_concrete_history(const ReducedRCColumnMaterialBaselineRunSpec& spec)
{
    const auto protocol = !spec.custom_protocol.empty()
                              ? to_driver_protocol(spec.custom_protocol)
                              : (spec.protocol_kind == ProtocolKind::monotonic
                                     ? make_monotonic_protocol(
                                           -std::abs(
                                               default_monotonic_target_strain(
                                                   spec)),
                                           spec.steps_per_branch)
                                     : fall_n::cyclic_driver::
                                           make_concrete_cyclic_protocol(
                                               default_amplitude_levels(spec),
                                               2.0e-4,
                                               spec.steps_per_branch));

    const auto result = fall_n::cyclic_driver::drive_kent_park_cyclic(
        spec.reference_spec.concrete_fpc_mpa,
        to_concrete_tension_config(spec.reference_spec),
        protocol);

    return to_baseline_records(result.records);
}

void print_progress(
    const ReducedRCColumnMaterialBaselineRunSpec& spec,
    const std::vector<ReducedRCColumnMaterialBaselineRecord>& records)
{
    if (!spec.print_progress || records.empty()) {
        return;
    }

    const auto every_n = std::max(
        1,
        static_cast<int>(records.size() / 5));
    for (std::size_t i = 0; i < records.size(); ++i) {
        const auto& record = records[i];
        if (i == 0 || i + 1 == records.size() || static_cast<int>(i) % every_n == 0) {
            std::println(
                "    reduced-material {:>20s} {:>9s}  step={:4d}  eps={:+.4e}  sig={:+.4e} MPa  Et={:+.4e} MPa",
                to_string(spec.material_kind),
                to_string(spec.protocol_kind),
                record.step,
                record.strain,
                record.stress_mpa,
                record.tangent_mpa);
        }
    }
}

} // namespace

ReducedRCColumnMaterialBaselineResult
run_reduced_rc_column_material_baseline(
    const ReducedRCColumnMaterialBaselineRunSpec& spec,
    const std::string& out_dir)
{
    if (spec.custom_protocol.empty() && spec.steps_per_branch <= 0) {
        throw std::invalid_argument(
            "Reduced RC material baseline requires steps_per_branch > 0.");
    }
    validate_custom_protocol(spec.custom_protocol);

    StopWatch total_timer;
    total_timer.start();
    StopWatch solve_timer;
    solve_timer.start();

    auto records = [&]() {
        switch (spec.material_kind) {
            case MaterialKind::steel_rebar:
                return run_steel_history(spec);
            case MaterialKind::concrete_unconfined:
                return run_concrete_history(spec);
        }
        throw std::invalid_argument(
            "Reduced RC material baseline received an unknown material kind.");
    }();

    const double solve_wall_seconds = solve_timer.stop();
    print_progress(spec, records);

    StopWatch output_timer;
    output_timer.start();
    if (spec.write_csv) {
        std::filesystem::create_directories(out_dir);
        write_material_baseline_csv(out_dir + "/uniaxial_response.csv", records);
    }

    return ReducedRCColumnMaterialBaselineResult{
        .records = std::move(records),
        .timing =
            ReducedRCColumnMaterialBaselineTimingSummary{
                .total_wall_seconds = total_timer.stop(),
                .solve_wall_seconds = solve_wall_seconds,
                .output_write_wall_seconds = output_timer.stop(),
            },
        .completed_successfully = true,
    };
}

} // namespace fall_n::validation_reboot
