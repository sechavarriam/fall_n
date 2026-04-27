#include "src/validation/ReducedRCColumnSectionBaseline.hh"
#include "src/validation/ReducedRCColumnBenchmarkSurface.hh"
#include "src/validation/ReducedRCColumnBenchmarkManifestSupport.hh"
#include "src/validation/ReducedRCColumnReferenceSpec.hh"

#include <algorithm>
#include <cctype>
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

using fall_n::validation_reboot::ReducedRCColumnSectionBaselineResult;
using fall_n::validation_reboot::ReducedRCColumnSectionBaselineRunSpec;

struct CliOptions {
    std::string output_dir{};
    std::string analysis{"monotonic"};
    std::string material_mode{"nonlinear"};
    double axial_compression_mn{0.02};
    double max_curvature_y{0.03};
    std::vector<double> amplitudes_curvature_y{};
    int steps{120};
    int steps_per_segment{4};
    int axial_force_newton_max_iterations{40};
    double axial_force_newton_tolerance_mn{1.0e-8};
    bool print_progress{false};
};

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

[[nodiscard]] std::vector<std::string_view> argv_span(int argc, char** argv)
{
    std::vector<std::string_view> args;
    args.reserve(static_cast<std::size_t>(std::max(argc - 1, 0)));
    for (int i = 1; i < argc; ++i) {
        args.emplace_back(argv[i]);
    }
    return args;
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
            "Usage: fall_n_reduced_rc_column_section_reference_benchmark "
            "--output-dir <dir> [--analysis monotonic|cyclic] "
            "[--axial-compression-mn value] "
            "[--material-mode nonlinear|elasticized] "
            "[--max-curvature-y value] [--steps N] "
            "[--amplitudes-curvature-y comma,separated] "
            "[--steps-per-segment N] "
            "[--axial-force-newton-max-iterations N] "
            "[--axial-force-newton-tolerance-mn value] [--print-progress]");
        std::exit(0);
    }

    CliOptions opts;
    opts.output_dir = value_of("--output-dir");
    if (opts.output_dir.empty()) {
        throw std::invalid_argument("--output-dir is required.");
    }
    if (const auto value = value_of("--analysis"); !value.empty()) {
        opts.analysis = value;
    }
    if (const auto value = value_of("--material-mode"); !value.empty()) {
        opts.material_mode = value;
    }
    if (const auto value = value_of("--axial-compression-mn"); !value.empty()) {
        opts.axial_compression_mn = std::stod(value);
    }
    if (const auto value = value_of("--max-curvature-y"); !value.empty()) {
        opts.max_curvature_y = std::stod(value);
    }
    if (const auto value = value_of("--steps"); !value.empty()) {
        opts.steps = std::stoi(value);
    }
    if (const auto value = value_of("--amplitudes-curvature-y"); !value.empty()) {
        opts.amplitudes_curvature_y = parse_csv_doubles(value);
    }
    if (const auto value = value_of("--steps-per-segment"); !value.empty()) {
        opts.steps_per_segment = std::stoi(value);
    }
    if (const auto value = value_of("--axial-force-newton-max-iterations");
        !value.empty()) {
        opts.axial_force_newton_max_iterations = std::stoi(value);
    }
    if (const auto value = value_of("--axial-force-newton-tolerance-mn");
        !value.empty()) {
        opts.axial_force_newton_tolerance_mn = std::stod(value);
    }
    if (has_flag("--print-progress")) {
        opts.print_progress = true;
    }

    return opts;
}

void write_manifest(
    const std::filesystem::path& path,
    const CliOptions& options,
    const ReducedRCColumnSectionBaselineResult& result)
{
    const auto analysis_kind =
        options.analysis == "cyclic"
            ? fall_n::validation_reboot::ReducedRCColumnBenchmarkAnalysisKind::cyclic
            : fall_n::validation_reboot::ReducedRCColumnBenchmarkAnalysisKind::monotonic;
    const auto input_surface =
        fall_n::validation_reboot::make_section_benchmark_input_surface(
            analysis_kind);
    const auto local_model_taxonomy =
        fall_n::validation_reboot::describe_reduced_rc_column_section_local_model(
            ReducedRCColumnSectionBaselineRunSpec{});
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
        << "  \"benchmark_scope\": "
           "\"reduced_rc_column_section_internal_reference\",\n"
        << "  \"analysis\": \"" << json_escape(options.analysis) << "\",\n"
        << "  \"material_mode\": \"" << json_escape(options.material_mode)
        << "\",\n"
        << "  \"axial_compression_mn\": " << options.axial_compression_mn << ",\n"
        << "  \"max_curvature_y\": " << options.max_curvature_y << ",\n"
        << "  \"steps\": " << options.steps << ",\n"
        << "  \"steps_per_segment\": " << options.steps_per_segment << ",\n"
        << "  \"cyclic_curvature_levels_y\": [";
    for (std::size_t i = 0; i < options.amplitudes_curvature_y.size(); ++i) {
        out << options.amplitudes_curvature_y[i];
        if (i + 1 < options.amplitudes_curvature_y.size()) {
            out << ", ";
        }
    }
    out << "],\n"
        << "  \"record_count\": " << result.records.size() << ",\n"
        << "  \"timing\": {\n"
        << "    \"total_wall_seconds\": " << result.timing.total_wall_seconds
        << ",\n"
        << "    \"analysis_wall_seconds\": " << result.timing.solve_wall_seconds
        << ",\n"
        << "    \"output_write_wall_seconds\": "
        << result.timing.output_write_wall_seconds << "\n"
        << "  },\n"
        << "  \"benchmark_note\": \""
        << json_escape(
               "Internal section-level moment-curvature baseline solved by "
               "axial-force closure over the audited reduced RC fiber section.")
        << "\"\n"
        << "}\n";
}

void write_section_layout_csv(const std::filesystem::path& path)
{
    const auto layout = fall_n::build_rc_column_fiber_layout(
        fall_n::validation_reboot::to_rc_column_section_spec(
            fall_n::validation_reboot::default_reduced_rc_column_reference_spec_v));

    std::ofstream out(path);
    out << "fiber_index,y,z,area,zone,material_role\n";
    out << std::scientific << std::setprecision(8);
    for (const auto& fiber : layout) {
        out << fiber.fiber_index << ","
            << fiber.y << ","
            << fiber.z << ","
            << fiber.area << ","
            << std::string{fall_n::to_string(fiber.zone)} << ","
            << std::string{fall_n::to_string(fiber.material_role)} << "\n";
    }
}

void write_station_layout_csv(const std::filesystem::path& path)
{
    std::ofstream out(path);
    out << "section_gp,xi\n";
    out << std::scientific << std::setprecision(8);
    out << 0 << "," << 0.0 << "\n";
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const auto options = parse_args(argc, argv);
        std::filesystem::create_directories(options.output_dir);
        if (options.material_mode != "nonlinear" &&
            options.material_mode != "elasticized") {
            throw std::invalid_argument(
                "--material-mode must be either 'nonlinear' or 'elasticized'.");
        }
        const auto material_mode =
            options.material_mode == "elasticized"
                ? fall_n::validation_reboot::ReducedRCColumnSectionMaterialMode::elasticized
                : fall_n::validation_reboot::ReducedRCColumnSectionMaterialMode::nonlinear;
        const auto protocol_kind =
            options.analysis == "cyclic"
                ? fall_n::validation_reboot::ReducedRCColumnSectionProtocolKind::cyclic
                : fall_n::validation_reboot::ReducedRCColumnSectionProtocolKind::monotonic;

        const auto result =
            fall_n::validation_reboot::
                run_reduced_rc_column_section_moment_curvature_baseline(
                    ReducedRCColumnSectionBaselineRunSpec{
                        .material_mode = material_mode,
                        .protocol_kind = protocol_kind,
                        .target_axial_compression_force_mn =
                            options.axial_compression_mn,
                        .max_curvature_y = options.max_curvature_y,
                        .cyclic_curvature_levels_y =
                            options.amplitudes_curvature_y,
                        .steps = std::max(options.steps, 1),
                        .steps_per_segment =
                            std::max(options.steps_per_segment, 1),
                        .axial_force_newton_max_iterations =
                            std::max(options.axial_force_newton_max_iterations, 1),
                        .axial_force_newton_tolerance_mn =
                            options.axial_force_newton_tolerance_mn,
                        .write_csv = true,
                        .print_progress = options.print_progress,
                    },
                    options.output_dir);

        write_manifest(
            std::filesystem::path{options.output_dir} / "runtime_manifest.json",
            options,
            result);
        write_section_layout_csv(
            std::filesystem::path{options.output_dir} / "section_layout.csv");
        write_station_layout_csv(
            std::filesystem::path{options.output_dir} /
            "section_station_layout.csv");

        std::println(
            "Section benchmark completed: total={:.6f}s, analysis={:.6f}s, "
            "write={:.6f}s, records={}",
            result.timing.total_wall_seconds,
            result.timing.solve_wall_seconds,
            result.timing.output_write_wall_seconds,
            result.records.size());
        return 0;
    } catch (const std::exception& ex) {
        std::println(stderr, "ERROR: {}", ex.what());
        return 1;
    }
}
