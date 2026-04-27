#include "src/validation/ReducedRCColumnTrussBaseline.hh"
#include "src/validation/ReducedRCColumnBenchmarkSurface.hh"
#include "src/validation/ReducedRCColumnBenchmarkManifestSupport.hh"

#include <petsc.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <print>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using fall_n::validation_reboot::json_escape;
using fall_n::validation_reboot::ReducedRCColumnBenchmarkAnalysisKind;
using fall_n::validation_reboot::ReducedRCColumnTrussBaselineResult;
using fall_n::validation_reboot::ReducedRCColumnTrussBaselineRunSpec;
using fall_n::validation_reboot::ReducedRCColumnTrussProtocolKind;

struct CliOptions {
    std::string output_dir{};
    std::string protocol{"cyclic_compression_return"};
    std::string protocol_csv{};
    std::string amplitudes_csv{};
    double monotonic_target_strain{0.0};
    double element_length_m{0.0};
    double area_m2{0.0};
    int steps_per_branch{40};
    bool print_progress{false};
};

[[nodiscard]] std::vector<std::string_view> argv_span(int argc, char** argv)
{
    std::vector<std::string_view> args;
    args.reserve(static_cast<std::size_t>(std::max(argc - 1, 0)));
    for (int i = 1; i < argc; ++i) {
        args.emplace_back(argv[i]);
    }
    return args;
}

[[nodiscard]] std::vector<double> parse_csv_doubles(std::string_view raw)
{
    std::vector<double> values;
    std::size_t start = 0;
    while (start < raw.size()) {
        const auto end = raw.find(',', start);
        const auto token = raw.substr(
            start,
            end == std::string_view::npos ? raw.size() - start : end - start);
        if (!token.empty()) {
            values.push_back(std::stod(std::string{token}));
        }
        if (end == std::string_view::npos) {
            break;
        }
        start = end + 1;
    }
    return values;
}

[[nodiscard]] std::vector<
    fall_n::validation_reboot::ReducedRCColumnMaterialBaselineRunSpec::ProtocolPoint>
parse_protocol_csv(const std::filesystem::path& path)
{
    std::ifstream in(path);
    if (!in) {
        throw std::invalid_argument(
            std::string{"Unable to open custom protocol CSV: "} +
            path.string());
    }

    std::string header;
    if (!std::getline(in, header)) {
        return {};
    }

    std::vector<
        fall_n::validation_reboot::ReducedRCColumnMaterialBaselineRunSpec::ProtocolPoint>
        protocol;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }

        const auto first_comma = line.find(',');
        if (first_comma == std::string::npos) {
            continue;
        }

        const auto second_comma = line.find(',', first_comma + 1);
        const auto step_token = line.substr(0, first_comma);
        const auto strain_token = line.substr(
            first_comma + 1,
            second_comma == std::string::npos
                ? std::string::npos
                : second_comma - first_comma - 1);
        if (step_token.empty() || strain_token.empty()) {
            continue;
        }

        protocol.push_back(
            fall_n::validation_reboot::ReducedRCColumnMaterialBaselineRunSpec::
                ProtocolPoint{
                    .step = std::stoi(step_token),
                    .strain = std::stod(strain_token),
                });
    }

    return protocol;
}

[[nodiscard]] CliOptions parse_args(int argc, char** argv)
{
    const auto args = argv_span(argc, argv);
    const auto value_of = [&](std::string_view flag) -> std::string {
        for (std::size_t i = 0; i + 1 < args.size(); ++i) {
            if (args[i] == flag) {
                return std::string{args[i + 1]};
            }
        }
        return {};
    };
    const auto has_flag = [&](std::string_view flag) {
        return std::ranges::find(args, flag) != args.end();
    };

    if (has_flag("--help") || has_flag("-h")) {
        std::println(
            "Usage: fall_n_reduced_rc_truss_reference_benchmark "
            "--output-dir <dir> [--protocol monotonic_compression|cyclic_compression_return] "
            "[--protocol-csv path] [--amplitudes csv] [--monotonic-target-strain value] "
            "[--element-length-m value] [--area-m2 value] "
            "[--steps-per-branch N] [--print-progress]");
        std::exit(0);
    }

    CliOptions opts;
    opts.output_dir = value_of("--output-dir");
    if (opts.output_dir.empty()) {
        throw std::invalid_argument("--output-dir is required.");
    }
    if (const auto value = value_of("--protocol"); !value.empty()) {
        opts.protocol = value;
    }
    if (const auto value = value_of("--protocol-csv"); !value.empty()) {
        opts.protocol_csv = value;
    }
    if (const auto value = value_of("--amplitudes"); !value.empty()) {
        opts.amplitudes_csv = value;
    }
    if (const auto value = value_of("--monotonic-target-strain"); !value.empty()) {
        opts.monotonic_target_strain = std::stod(value);
    }
    if (const auto value = value_of("--element-length-m"); !value.empty()) {
        opts.element_length_m = std::stod(value);
    }
    if (const auto value = value_of("--area-m2"); !value.empty()) {
        opts.area_m2 = std::stod(value);
    }
    if (const auto value = value_of("--steps-per-branch"); !value.empty()) {
        opts.steps_per_branch = std::stoi(value);
    }
    if (has_flag("--print-progress")) {
        opts.print_progress = true;
    }
    return opts;
}

[[nodiscard]] ReducedRCColumnTrussProtocolKind protocol_kind_of(
    std::string_view raw)
{
    if (raw == "monotonic_compression") {
        return ReducedRCColumnTrussProtocolKind::monotonic_compression;
    }
    if (raw == "cyclic_compression_return") {
        return ReducedRCColumnTrussProtocolKind::cyclic_compression_return;
    }
    throw std::invalid_argument(
        "--protocol must be either 'monotonic_compression' or 'cyclic_compression_return'.");
}

void write_manifest(
    const std::filesystem::path& path,
    const CliOptions& options,
    const ReducedRCColumnTrussBaselineRunSpec& resolved_spec,
    const ReducedRCColumnTrussBaselineResult& result)
{
    const auto& reference_spec = resolved_spec.reference_spec;
    const double resolved_length_m =
        resolved_spec.element_length_m > 0.0
            ? resolved_spec.element_length_m
            : reference_spec.column_height_m;
    const double resolved_area_m2 =
        resolved_spec.area_m2 > 0.0
            ? resolved_spec.area_m2
            : 0.25 * 3.14159265358979323846 *
                  reference_spec.longitudinal_bar_diameter_m *
                  reference_spec.longitudinal_bar_diameter_m;
    const auto analysis_kind =
        protocol_kind_of(options.protocol) ==
                ReducedRCColumnTrussProtocolKind::cyclic_compression_return
            ? ReducedRCColumnBenchmarkAnalysisKind::cyclic
            : ReducedRCColumnBenchmarkAnalysisKind::monotonic;
    const auto input_surface =
        fall_n::validation_reboot::make_truss_benchmark_input_surface(
            analysis_kind);
    const auto local_model_taxonomy =
        fall_n::validation_reboot::describe_reduced_rc_column_truss_local_model(
            ReducedRCColumnTrussBaselineRunSpec{});

    std::ofstream out(path);
    out << "{\n";
    fall_n::validation_reboot::write_manifest_preamble(
        out,
        {
            .tool = "fall_n",
            .status =
                result.completed_successfully ? "completed" : "failed",
            .input_surface = input_surface,
            .local_model_taxonomy = local_model_taxonomy,
        },
        "  ");
    out << ",\n"
        << "  \"benchmark_scope\": \"reduced_rc_column_quadratic_truss_internal_reference\",\n"
        << "  \"protocol\": \"" << json_escape(options.protocol) << "\",\n"
        << "  \"protocol_source\": \""
        << (options.protocol_csv.empty() ? "generated" : "csv") << "\",\n"
        << "  \"protocol_csv\": \"" << json_escape(options.protocol_csv)
        << "\",\n"
        << "  \"amplitudes_csv\": \"" << json_escape(options.amplitudes_csv)
        << "\",\n"
        << "  \"monotonic_target_strain\": " << options.monotonic_target_strain << ",\n"
        << "  \"element_length_m\": " << resolved_length_m << ",\n"
        << "  \"area_m2\": " << resolved_area_m2 << ",\n"
        << "  \"steps_per_branch\": " << options.steps_per_branch << ",\n"
        << "  \"material_record_count\": " << result.material_records.size() << ",\n"
        << "  \"truss_record_count\": " << result.truss_records.size() << ",\n"
        << "  \"gauss_record_count\": " << result.gauss_records.size() << ",\n"
        << "  \"comparison\": {\n"
        << "    \"max_abs_stress_error_mpa\": "
        << result.comparison.max_abs_stress_error_mpa << ",\n"
        << "    \"rms_abs_stress_error_mpa\": "
        << result.comparison.rms_abs_stress_error_mpa << ",\n"
        << "    \"max_abs_tangent_error_mpa\": "
        << result.comparison.max_abs_tangent_error_mpa << ",\n"
        << "    \"rms_abs_tangent_error_mpa\": "
        << result.comparison.rms_abs_tangent_error_mpa << ",\n"
        << "    \"max_abs_element_tangent_error_mpa\": "
        << result.comparison.max_abs_element_tangent_error_mpa << ",\n"
        << "    \"rms_abs_element_tangent_error_mpa\": "
        << result.comparison.rms_abs_element_tangent_error_mpa << ",\n"
        << "    \"max_abs_energy_density_error_mpa\": "
        << result.comparison.max_abs_energy_density_error_mpa << ",\n"
        << "    \"rms_abs_energy_density_error_mpa\": "
        << result.comparison.rms_abs_energy_density_error_mpa << ",\n"
        << "    \"max_abs_axial_force_closure_mn\": "
        << result.comparison.max_abs_axial_force_closure_mn << ",\n"
        << "    \"max_abs_gp_strain_spread\": "
        << result.comparison.max_abs_gp_strain_spread << ",\n"
        << "    \"max_abs_gp_stress_spread_mpa\": "
        << result.comparison.max_abs_gp_stress_spread_mpa << ",\n"
        << "    \"max_abs_gp_tangent_spread_mpa\": "
        << result.comparison.max_abs_gp_tangent_spread_mpa << ",\n"
        << "    \"max_abs_middle_node_force_mn\": "
        << result.comparison.max_abs_middle_node_force_mn << "\n"
        << "  },\n"
        << "  \"timing\": {\n"
        << "    \"total_wall_seconds\": " << result.timing.total_wall_seconds << ",\n"
        << "    \"analysis_wall_seconds\": " << result.timing.analysis_wall_seconds << ",\n"
        << "    \"output_write_wall_seconds\": " << result.timing.output_write_wall_seconds << "\n"
        << "  },\n"
        << "  \"benchmark_note\": \"Standalone quadratic truss audit using the same Menegotto-Pinto steel factory as the reduced-column structural section and continuum rebar carriers.\"\n"
        << "}\n";
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const auto options = parse_args(argc, argv);
        PetscInitializeNoArguments();
        std::filesystem::create_directories(options.output_dir);

        const auto result =
            [&]() {
                const auto run_spec = ReducedRCColumnTrussBaselineRunSpec{
                    .protocol_kind = protocol_kind_of(options.protocol),
                    .element_length_m = options.element_length_m,
                    .area_m2 = options.area_m2,
                    .monotonic_target_strain = options.monotonic_target_strain,
                    .compression_amplitude_levels =
                        parse_csv_doubles(options.amplitudes_csv),
                    .custom_protocol =
                        options.protocol_csv.empty()
                            ? std::vector<
                                  fall_n::validation_reboot::
                                      ReducedRCColumnMaterialBaselineRunSpec::
                                          ProtocolPoint>{}
                            : parse_protocol_csv(options.protocol_csv),
                    .steps_per_branch = std::max(options.steps_per_branch, 1),
                    .write_csv = true,
                    .print_progress = options.print_progress,
                };
                const auto result =
                    fall_n::validation_reboot::run_reduced_rc_column_truss_baseline(
                        run_spec,
                        options.output_dir);
                return std::pair{run_spec, result};
            }();

        write_manifest(
            std::filesystem::path{options.output_dir} / "runtime_manifest.json",
            options,
            result.first,
            result.second);

        std::println(
            "Truss benchmark completed: total={:.6f}s, analysis={:.6f}s, "
            "write={:.6f}s, records={}, gp_records={}",
            result.second.timing.total_wall_seconds,
            result.second.timing.analysis_wall_seconds,
            result.second.timing.output_write_wall_seconds,
            result.second.truss_records.size(),
            result.second.gauss_records.size());
        PetscFinalize();
        return 0;
    } catch (const std::exception& ex) {
        PetscBool petsc_initialized = PETSC_FALSE;
        PetscInitialized(&petsc_initialized);
        if (petsc_initialized) {
            PetscFinalize();
        }
        std::println(stderr, "ERROR: {}", ex.what());
        return 1;
    }
}
