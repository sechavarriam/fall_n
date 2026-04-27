#include "src/validation/ReducedRCColumnMaterialBaseline.hh"
#include "src/validation/ReducedRCColumnBenchmarkSurface.hh"
#include "src/validation/ReducedRCColumnBenchmarkManifestSupport.hh"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <print>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using fall_n::validation_reboot::json_escape;

using fall_n::validation_reboot::ReducedRCColumnMaterialBaselineResult;
using fall_n::validation_reboot::ReducedRCColumnMaterialBaselineRunSpec;
using fall_n::validation_reboot::ReducedRCColumnMaterialProtocolKind;
using fall_n::validation_reboot::ReducedRCColumnMaterialReferenceKind;

struct CliOptions {
    std::string output_dir{};
    std::string material{"steel"};
    std::string protocol{"cyclic"};
    std::string protocol_csv{};
    double monotonic_target_strain{0.0};
    std::string levels_csv{};
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

[[nodiscard]] std::vector<double> parse_levels_csv(std::string_view raw)
{
    std::vector<double> levels;
    std::size_t offset = 0;
    while (offset < raw.size()) {
        const auto comma = raw.find(',', offset);
        const auto token = raw.substr(
            offset,
            comma == std::string_view::npos ? raw.size() - offset
                                            : comma - offset);
        if (!token.empty()) {
            levels.push_back(std::stod(std::string{token}));
        }
        if (comma == std::string_view::npos) {
            break;
        }
        offset = comma + 1;
    }
    return levels;
}

[[nodiscard]] std::vector<
    fall_n::validation_reboot::ReducedRCColumnMaterialBaselineRunSpec::
        ProtocolPoint>
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

    std::vector<fall_n::validation_reboot::ReducedRCColumnMaterialBaselineRunSpec::ProtocolPoint>
        protocol;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        std::stringstream row(line);
        std::string step_token;
        std::string strain_token;
        std::getline(row, step_token, ',');
        std::getline(row, strain_token, ',');
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
            "Usage: fall_n_reduced_rc_material_reference_benchmark "
            "--output-dir <dir> [--material steel|concrete] "
            "[--protocol monotonic|cyclic] [--monotonic-target-strain value] "
            "[--levels csv] [--protocol-csv path] "
            "[--steps-per-branch N] [--print-progress]");
        std::exit(0);
    }

    CliOptions opts;
    opts.output_dir = value_of("--output-dir");
    if (opts.output_dir.empty()) {
        throw std::invalid_argument("--output-dir is required.");
    }
    if (const auto value = value_of("--material"); !value.empty()) {
        opts.material = value;
    }
    if (const auto value = value_of("--protocol"); !value.empty()) {
        opts.protocol = value;
    }
    if (const auto value = value_of("--protocol-csv"); !value.empty()) {
        opts.protocol_csv = value;
    }
    if (const auto value = value_of("--monotonic-target-strain"); !value.empty()) {
        opts.monotonic_target_strain = std::stod(value);
    }
    if (const auto value = value_of("--levels"); !value.empty()) {
        opts.levels_csv = value;
    }
    if (const auto value = value_of("--steps-per-branch"); !value.empty()) {
        opts.steps_per_branch = std::stoi(value);
    }
    if (has_flag("--print-progress")) {
        opts.print_progress = true;
    }

    return opts;
}

[[nodiscard]] ReducedRCColumnMaterialReferenceKind material_kind_of(
    std::string_view raw)
{
    if (raw == "steel") {
        return ReducedRCColumnMaterialReferenceKind::steel_rebar;
    }
    if (raw == "concrete") {
        return ReducedRCColumnMaterialReferenceKind::concrete_unconfined;
    }
    throw std::invalid_argument("--material must be either 'steel' or 'concrete'.");
}

[[nodiscard]] ReducedRCColumnMaterialProtocolKind protocol_kind_of(
    std::string_view raw)
{
    if (raw == "monotonic") {
        return ReducedRCColumnMaterialProtocolKind::monotonic;
    }
    if (raw == "cyclic") {
        return ReducedRCColumnMaterialProtocolKind::cyclic;
    }
    throw std::invalid_argument("--protocol must be either 'monotonic' or 'cyclic'.");
}

void write_manifest(
    const std::filesystem::path& path,
    const CliOptions& options,
    const ReducedRCColumnMaterialBaselineResult& result)
{
    const auto analysis_kind =
        options.protocol == "cyclic"
            ? fall_n::validation_reboot::ReducedRCColumnBenchmarkAnalysisKind::cyclic
            : fall_n::validation_reboot::ReducedRCColumnBenchmarkAnalysisKind::monotonic;
    const auto input_surface =
        fall_n::validation_reboot::make_material_benchmark_input_surface(
            analysis_kind);
    const auto local_model_taxonomy =
        fall_n::validation_reboot::describe_reduced_rc_column_material_local_model(
            ReducedRCColumnMaterialBaselineRunSpec{
                .material_kind = material_kind_of(options.material),
                .protocol_kind = protocol_kind_of(options.protocol),
            });
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
        << "  \"benchmark_scope\": \"reduced_rc_column_uniaxial_material_internal_reference\",\n"
        << "  \"material\": \"" << json_escape(options.material) << "\",\n"
        << "  \"protocol\": \"" << json_escape(options.protocol) << "\",\n"
        << "  \"protocol_source\": \""
        << (options.protocol_csv.empty() ? "generated" : "csv") << "\",\n"
        << "  \"protocol_csv\": \"" << json_escape(options.protocol_csv)
        << "\",\n"
        << "  \"monotonic_target_strain\": " << options.monotonic_target_strain << ",\n"
        << "  \"levels_csv\": \"" << json_escape(options.levels_csv) << "\",\n"
        << "  \"steps_per_branch\": " << options.steps_per_branch << ",\n"
        << "  \"record_count\": " << result.records.size() << ",\n"
        << "  \"timing\": {\n"
        << "    \"total_wall_seconds\": " << result.timing.total_wall_seconds << ",\n"
        << "    \"analysis_wall_seconds\": " << result.timing.solve_wall_seconds << ",\n"
        << "    \"output_write_wall_seconds\": " << result.timing.output_write_wall_seconds << "\n"
        << "  },\n"
        << "  \"benchmark_note\": \"Internal uniaxial reduced-column material baseline over the audited Menegotto-Pinto or Kent-Park ingredient.\"\n"
        << "}\n";
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const auto options = parse_args(argc, argv);
        std::filesystem::create_directories(options.output_dir);

        const auto result = fall_n::validation_reboot::run_reduced_rc_column_material_baseline(
            ReducedRCColumnMaterialBaselineRunSpec{
                .material_kind = material_kind_of(options.material),
                .protocol_kind = protocol_kind_of(options.protocol),
                .monotonic_target_strain = options.monotonic_target_strain,
                .amplitude_levels = parse_levels_csv(options.levels_csv),
                .custom_protocol = options.protocol_csv.empty()
                                       ? std::vector<ReducedRCColumnMaterialBaselineRunSpec::ProtocolPoint>{}
                                       : parse_protocol_csv(options.protocol_csv),
                .steps_per_branch = std::max(options.steps_per_branch, 1),
                .write_csv = true,
                .print_progress = options.print_progress,
            },
            options.output_dir);

        write_manifest(
            std::filesystem::path{options.output_dir} / "runtime_manifest.json",
            options,
            result);

        std::println(
            "Material benchmark completed: total={:.6f}s, analysis={:.6f}s, write={:.6f}s, records={}",
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
