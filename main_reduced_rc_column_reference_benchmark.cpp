#include "src/validation/ReducedRCColumnStructuralBaseline.hh"
#include "src/validation/ReducedRCColumnBenchmarkSurface.hh"
#include "src/validation/ReducedRCColumnBenchmarkManifestSupport.hh"
#include "src/validation/ReducedRCColumnReferenceSpec.hh"

#include <petsc.h>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <print>
#include <ranges>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using fall_n::table_cyclic_validation::CyclicValidationRunConfig;
using fall_n::validation_reboot::ReducedRCColumnContinuationKind;
using fall_n::validation_reboot::ReducedRCColumnSectionResponseRecord;
using fall_n::validation_reboot::ReducedRCColumnStructuralRunResult;
using fall_n::validation_reboot::ReducedRCColumnStructuralRunSpec;
using fall_n::validation_reboot::json_escape;

struct CliOptions {
    std::string analysis{"cyclic"};
    std::string output_dir{};
    std::string material_mode{"nonlinear"};
    std::string solver_policy{"canonical-cascade"};
    std::size_t beam_nodes{4};
    std::size_t structural_element_count{1};
    ::BeamAxisQuadratureFamily beam_integration{
        BeamAxisQuadratureFamily::GaussLegendre};
    bool clamp_top_bending_rotation{false};
    bool prescribe_top_bending_rotation_from_drift{false};
    double top_bending_rotation_drift_ratio{0.0};
    double axial_compression_mn{0.02};
    bool use_equilibrated_axial_preload_stage{true};
    int axial_preload_steps{4};
    ReducedRCColumnContinuationKind continuation_kind{
        ReducedRCColumnContinuationKind::
            reversal_guarded_incremental_displacement_control};
    int continuation_segment_substep_factor{2};
    double monotonic_tip_mm{2.5};
    int monotonic_steps{8};
    std::vector<double> amplitudes_mm{1.25, 2.50};
    int steps_per_segment{2};
    int max_bisections{8};
    std::string section_fiber_profile{"canonical"};
    bool write_element_tangent_audit_csv{false};
    bool print_progress{false};
};

struct ProtocolPoint {
    int step{0};
    double p{0.0};
    double target_drift_m{0.0};
};

[[nodiscard]] int effective_runtime_lateral_steps(
    const CliOptions& options,
    const CyclicValidationRunConfig& cfg) noexcept
{
    int steps = cfg.total_steps();
    if (options.continuation_kind ==
        ReducedRCColumnContinuationKind::
            reversal_guarded_incremental_displacement_control) {
        steps *= std::max(options.continuation_segment_substep_factor, 1);
    }
    return std::max(steps, 0);
}

[[nodiscard]] std::string to_lower_copy(std::string value)
{
    std::ranges::transform(value, value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
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

[[nodiscard]] fall_n::validation_reboot::ReducedRCColumnBenchmarkAnalysisKind
parse_benchmark_analysis_kind(std::string value)
{
    value = to_lower_copy(std::move(value));
    if (value == "monotonic") {
        return fall_n::validation_reboot::ReducedRCColumnBenchmarkAnalysisKind::
            monotonic;
    }
    if (value == "cyclic") {
        return fall_n::validation_reboot::ReducedRCColumnBenchmarkAnalysisKind::
            cyclic;
    }
    throw std::invalid_argument("Unsupported benchmark analysis kind.");
}

[[nodiscard]] ::BeamAxisQuadratureFamily
parse_beam_integration(std::string value)
{
    value = to_lower_copy(std::move(value));
    if (value == "legendre" || value == "gausslegendre" || value == "gauss-legendre") {
        return BeamAxisQuadratureFamily::GaussLegendre;
    }
    if (value == "lobatto" || value == "gausslobatto" || value == "gauss-lobatto") {
        return BeamAxisQuadratureFamily::GaussLobatto;
    }
    if (value == "radau-left" || value == "gaussradauleft" || value == "gauss-radau-left") {
        return BeamAxisQuadratureFamily::GaussRadauLeft;
    }
    if (value == "radau-right" || value == "gaussradauright" || value == "gauss-radau-right") {
        return BeamAxisQuadratureFamily::GaussRadauRight;
    }
    throw std::invalid_argument("Unsupported --beam-integration value.");
}

[[nodiscard]] fall_n::RCColumnSectionMeshSpec
parse_section_fiber_profile(std::string value)
{
    value = to_lower_copy(std::move(value));
    if (value == "coarse") {
        return fall_n::coarse_rc_column_section_mesh();
    }
    if (value == "canonical" || value == "default") {
        return fall_n::canonical_rc_column_section_mesh();
    }
    if (value == "fine") {
        return fall_n::fine_rc_column_section_mesh();
    }
    if (value == "ultra" || value == "ultra-fine" || value == "ultrafine") {
        return fall_n::ultra_fine_rc_column_section_mesh();
    }
    throw std::invalid_argument("Unsupported --section-fiber-profile value.");
}

[[nodiscard]] ReducedRCColumnContinuationKind
parse_continuation_kind(std::string value)
{
    value = to_lower_copy(std::move(value));
    if (value == "monolithic" || value == "monolithic_incremental_displacement_control") {
        return ReducedRCColumnContinuationKind::
            monolithic_incremental_displacement_control;
    }
    if (value == "segmented" || value == "segmented_incremental_displacement_control") {
        return ReducedRCColumnContinuationKind::
            segmented_incremental_displacement_control;
    }
    if (value == "reversal-guarded" ||
        value == "reversal_guarded_incremental_displacement_control") {
        return ReducedRCColumnContinuationKind::
            reversal_guarded_incremental_displacement_control;
    }
    if (value == "arc-length" || value == "arc_length_continuation_candidate") {
        return ReducedRCColumnContinuationKind::arc_length_continuation_candidate;
    }
    throw std::invalid_argument("Unsupported --continuation value.");
}

[[nodiscard]] fall_n::validation_reboot::ReducedRCColumnSolverPolicyKind
parse_solver_policy_kind(std::string value)
{
    value = to_lower_copy(std::move(value));
    if (value == "canonical-cascade" ||
        value == "canonical_newton_profile_cascade")
    {
        return fall_n::validation_reboot::ReducedRCColumnSolverPolicyKind::
            canonical_newton_profile_cascade;
    }
    if (value == "newton-backtracking-only" ||
        value == "newton_backtracking_only")
    {
        return fall_n::validation_reboot::ReducedRCColumnSolverPolicyKind::
            newton_backtracking_only;
    }
    if (value == "newton-l2-only" ||
        value == "newton_l2_only")
    {
        return fall_n::validation_reboot::ReducedRCColumnSolverPolicyKind::
            newton_l2_only;
    }
    if (value == "newton-l2-lu-symbolic-reuse-only" ||
        value == "newton_l2_lu_symbolic_reuse_only" ||
        value == "newton-l2-reuse-only")
    {
        return fall_n::validation_reboot::ReducedRCColumnSolverPolicyKind::
            newton_l2_lu_symbolic_reuse_only;
    }
    if (value == "newton-l2-gmres-ilu1-only" ||
        value == "newton_l2_gmres_ilu1_only" ||
        value == "newton-l2-gmres-ilu-only")
    {
        return fall_n::validation_reboot::ReducedRCColumnSolverPolicyKind::
            newton_l2_gmres_ilu1_only;
    }
    if (value == "newton-trust-region-only" ||
        value == "newton_trust_region_only")
    {
        return fall_n::validation_reboot::ReducedRCColumnSolverPolicyKind::
            newton_trust_region_only;
    }
    if (value == "newton-trust-region-dogleg-only" ||
        value == "newton_trust_region_dogleg_only" ||
        value == "newtontrdc-only")
    {
        return fall_n::validation_reboot::ReducedRCColumnSolverPolicyKind::
            newton_trust_region_dogleg_only;
    }
    if (value == "quasi-newton-only" ||
        value == "quasi_newton_only")
    {
        return fall_n::validation_reboot::ReducedRCColumnSolverPolicyKind::
            quasi_newton_only;
    }
    if (value == "nonlinear-gmres-only" ||
        value == "nonlinear_gmres_only" ||
        value == "ngmres-only")
    {
        return fall_n::validation_reboot::ReducedRCColumnSolverPolicyKind::
            nonlinear_gmres_only;
    }
    if (value == "nonlinear-cg-only" ||
        value == "nonlinear_cg_only" ||
        value == "ncg-only")
    {
        return fall_n::validation_reboot::ReducedRCColumnSolverPolicyKind::
            nonlinear_conjugate_gradient_only;
    }
    if (value == "anderson-only" ||
        value == "anderson_acceleration_only" ||
        value == "anderson-acceleration-only")
    {
        return fall_n::validation_reboot::ReducedRCColumnSolverPolicyKind::
            anderson_acceleration_only;
    }
    if (value == "nonlinear-richardson-only" ||
        value == "nonlinear_richardson_only" ||
        value == "nrichardson-only")
    {
        return fall_n::validation_reboot::ReducedRCColumnSolverPolicyKind::
            nonlinear_richardson_only;
    }
    throw std::invalid_argument("Unsupported --solver-policy value.");
}

[[nodiscard]] CliOptions parse_args(int argc, char** argv)
{
    auto opts = CliOptions{};
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
            "Usage: fall_n_reduced_rc_column_reference_benchmark "
            "--output-dir <dir> [--analysis monotonic|cyclic] "
            "[--material-mode nonlinear|elasticized] "
            "[--solver-policy canonical-cascade|newton-backtracking-only|newton-l2-only|newton-l2-lu-symbolic-reuse-only|newton-trust-region-only|newton-trust-region-dogleg-only|quasi-newton-only|nonlinear-gmres-only|nonlinear-cg-only|anderson-only|nonlinear-richardson-only] "
            "[--beam-nodes N] [--beam-integration legendre|lobatto|radau-left|radau-right] "
            "[--structural-element-count N] "
            "[--clamp-top-bending-rotation] "
            "[--top-bending-rotation-drift-ratio value] "
            "[--axial-compression-mn value] [--axial-preload-steps N] "
            "[--continuation kind] [--continuation-segment-substep-factor N] "
            "[--monotonic-tip-mm value] [--monotonic-steps N] "
            "[--amplitudes-mm comma,separated] [--steps-per-segment N] "
            "[--max-bisections N] "
            "[--section-fiber-profile coarse|canonical|fine|ultra] "
            "[--write-element-tangent-audit-csv] "
            "[--print-progress]");
        std::exit(0);
    }

    opts.output_dir = value_of("--output-dir");
    if (opts.output_dir.empty()) {
        throw std::invalid_argument("--output-dir is required.");
    }

    if (const auto value = value_of("--analysis"); !value.empty()) {
        opts.analysis = to_lower_copy(value);
    }
    if (const auto value = value_of("--material-mode"); !value.empty()) {
        opts.material_mode = to_lower_copy(value);
    }
    if (const auto value = value_of("--solver-policy"); !value.empty()) {
        opts.solver_policy = to_lower_copy(value);
    }
    if (const auto value = value_of("--beam-nodes"); !value.empty()) {
        opts.beam_nodes = static_cast<std::size_t>(std::stoul(value));
    }
    if (const auto value = value_of("--structural-element-count"); !value.empty()) {
        opts.structural_element_count =
            std::max<std::size_t>(
                static_cast<std::size_t>(std::stoul(value)),
                1u);
    }
    if (const auto value = value_of("--beam-integration"); !value.empty()) {
        opts.beam_integration = parse_beam_integration(value);
    }
    if (has_flag("--clamp-top-bending-rotation")) {
        opts.clamp_top_bending_rotation = true;
    }
    if (const auto value = value_of("--top-bending-rotation-drift-ratio");
        !value.empty()) {
        opts.prescribe_top_bending_rotation_from_drift = true;
        opts.top_bending_rotation_drift_ratio = std::stod(value);
    }
    if (opts.clamp_top_bending_rotation &&
        opts.prescribe_top_bending_rotation_from_drift) {
        throw std::invalid_argument(
            "--clamp-top-bending-rotation and "
            "--top-bending-rotation-drift-ratio are mutually exclusive.");
    }
    if (const auto value = value_of("--axial-compression-mn"); !value.empty()) {
        opts.axial_compression_mn = std::stod(value);
    }
    if (const auto value = value_of("--axial-preload-steps"); !value.empty()) {
        opts.axial_preload_steps = std::stoi(value);
    }
    if (const auto value = value_of("--continuation"); !value.empty()) {
        opts.continuation_kind = parse_continuation_kind(value);
    } else if (opts.analysis == "monotonic") {
        opts.continuation_kind =
            ReducedRCColumnContinuationKind::
                monolithic_incremental_displacement_control;
        opts.continuation_segment_substep_factor = 1;
    }
    if (const auto value =
            value_of("--continuation-segment-substep-factor");
        !value.empty()) {
        opts.continuation_segment_substep_factor = std::stoi(value);
    }
    if (const auto value = value_of("--monotonic-tip-mm"); !value.empty()) {
        opts.monotonic_tip_mm = std::stod(value);
    }
    if (const auto value = value_of("--monotonic-steps"); !value.empty()) {
        opts.monotonic_steps = std::stoi(value);
    }
    if (const auto value = value_of("--amplitudes-mm"); !value.empty()) {
        opts.amplitudes_mm = parse_csv_doubles(value);
    }
    if (const auto value = value_of("--steps-per-segment"); !value.empty()) {
        opts.steps_per_segment = std::stoi(value);
    }
    if (const auto value = value_of("--max-bisections"); !value.empty()) {
        opts.max_bisections = std::stoi(value);
    }
    if (const auto value = value_of("--section-fiber-profile"); !value.empty()) {
        (void)parse_section_fiber_profile(value);
        opts.section_fiber_profile = to_lower_copy(value);
    }
    if (has_flag("--print-progress")) {
        opts.print_progress = true;
    }
    if (has_flag("--write-element-tangent-audit-csv")) {
        opts.write_element_tangent_audit_csv = true;
    }
    if (has_flag("--disable-equilibrated-axial-preload-stage")) {
        opts.use_equilibrated_axial_preload_stage = false;
    }

    return opts;
}

[[nodiscard]] std::string beam_integration_key(
    ::BeamAxisQuadratureFamily family) noexcept
{
    switch (family) {
        case BeamAxisQuadratureFamily::GaussLegendre:
            return "legendre";
        case BeamAxisQuadratureFamily::GaussLobatto:
            return "lobatto";
        case BeamAxisQuadratureFamily::GaussRadauLeft:
            return "radau-left";
        case BeamAxisQuadratureFamily::GaussRadauRight:
            return "radau-right";
    }
    return "unknown";
}

[[nodiscard]] std::string section_fiber_profile_key(
    const fall_n::RCColumnSectionMeshSpec& mesh) noexcept
{
    const auto same = [](const auto& a, const auto& b) {
        return a.cover_top_bottom_ny == b.cover_top_bottom_ny &&
               a.cover_top_bottom_nz == b.cover_top_bottom_nz &&
               a.cover_side_ny == b.cover_side_ny &&
               a.cover_side_nz == b.cover_side_nz &&
               a.core_ny == b.core_ny &&
               a.core_nz == b.core_nz;
    };
    if (same(mesh, fall_n::coarse_rc_column_section_mesh())) {
        return "coarse";
    }
    if (same(mesh, fall_n::canonical_rc_column_section_mesh())) {
        return "canonical";
    }
    if (same(mesh, fall_n::fine_rc_column_section_mesh())) {
        return "fine";
    }
    if (same(mesh, fall_n::ultra_fine_rc_column_section_mesh())) {
        return "ultra";
    }
    return "custom";
}

[[nodiscard]] CyclicValidationRunConfig
make_protocol(const CliOptions& options)
{
    if (options.analysis == "monotonic") {
        return {
            .protocol_name = "monotonic",
            .execution_profile_name = "benchmark_reference",
            .amplitudes_m = {1.0e-3 * options.monotonic_tip_mm},
            .steps_per_segment = std::max(options.monotonic_steps, 1),
            .max_steps = 0,
            .max_bisections = std::max(options.max_bisections, 0),
        };
    }

    std::vector<double> amplitudes_m;
    amplitudes_m.reserve(options.amplitudes_mm.size());
    std::ranges::transform(
        options.amplitudes_mm,
        std::back_inserter(amplitudes_m),
        [](double mm) { return 1.0e-3 * mm; });

    return {
        .protocol_name = "reduced_rc_column_reference_cyclic",
        .execution_profile_name = "benchmark_reference",
        .amplitudes_m = std::move(amplitudes_m),
        .steps_per_segment = std::max(options.steps_per_segment, 1),
        .max_steps = 0,
        .max_bisections = std::max(options.max_bisections, 0),
    };
}

[[nodiscard]] std::vector<ProtocolPoint>
build_comparison_protocol(const CliOptions& options, const CyclicValidationRunConfig& cfg)
{
    std::vector<ProtocolPoint> protocol;
    protocol.push_back({.step = 0, .p = 0.0, .target_drift_m = 0.0});

    if (options.analysis == "monotonic") {
        const auto steps = std::max(options.monotonic_steps, 1);
        const auto target = 1.0e-3 * options.monotonic_tip_mm;
        for (int step = 1; step <= steps; ++step) {
            const auto p = static_cast<double>(step) / static_cast<double>(steps);
            protocol.push_back({.step = step, .p = p, .target_drift_m = target * p});
        }
        return protocol;
    }

    const auto lateral_runtime_steps = effective_runtime_lateral_steps(options, cfg);
    for (int step = 1; step <= lateral_runtime_steps; ++step) {
        const auto p =
            static_cast<double>(step) / static_cast<double>(lateral_runtime_steps);
        protocol.push_back({.step = step, .p = p, .target_drift_m = cfg.displacement(p)});
    }
    return protocol;
}

[[nodiscard]] std::vector<fall_n::table_cyclic_validation::StepRecord>
select_comparison_hysteresis(
    const CliOptions& options,
    std::span<const fall_n::table_cyclic_validation::StepRecord> rows)
{
    if (options.analysis != "monotonic" || rows.empty()) {
        return {rows.begin(), rows.end()};
    }
    return {rows.begin(), rows.end()};
}

[[nodiscard]] std::vector<ReducedRCColumnSectionResponseRecord>
select_comparison_section_rows(
    std::span<const ReducedRCColumnSectionResponseRecord> rows,
    std::span<const fall_n::table_cyclic_validation::StepRecord> comparison_hysteresis)
{
    const auto keep_steps = [&] {
        std::set<int> steps;
        for (const auto& row : comparison_hysteresis) {
            steps.insert(row.step);
        }
        return steps;
    }();

    std::vector<ReducedRCColumnSectionResponseRecord> filtered;
    filtered.reserve(rows.size());
    for (const auto& row : rows) {
        if (keep_steps.contains(row.step)) {
            filtered.push_back(row);
        }
    }
    return filtered;
}

[[nodiscard]] std::vector<fall_n::validation_reboot::ReducedRCColumnStructuralControlStateRecord>
select_comparison_control_states(
    std::span<const fall_n::validation_reboot::ReducedRCColumnStructuralControlStateRecord> rows,
    std::span<const fall_n::table_cyclic_validation::StepRecord> comparison_hysteresis)
{
    const auto keep_steps = [&] {
        std::set<int> steps;
        for (const auto& row : comparison_hysteresis) {
            steps.insert(row.step);
        }
        return steps;
    }();

    std::vector<fall_n::validation_reboot::ReducedRCColumnStructuralControlStateRecord>
        filtered;
    filtered.reserve(rows.size());
    for (const auto& row : rows) {
        if (keep_steps.contains(row.step)) {
            filtered.push_back(row);
        }
    }
    return filtered;
}

[[nodiscard]] std::vector<
    fall_n::validation_reboot::ReducedRCColumnStructuralSectionFiberRecord>
select_comparison_fiber_states(
    std::span<
        const fall_n::validation_reboot::ReducedRCColumnStructuralSectionFiberRecord>
        rows,
    std::span<const fall_n::table_cyclic_validation::StepRecord> comparison_hysteresis)
{
    const auto keep_steps = [&] {
        std::set<int> steps;
        for (const auto& row : comparison_hysteresis) {
            steps.insert(row.step);
        }
        return steps;
    }();

    std::vector<
        fall_n::validation_reboot::ReducedRCColumnStructuralSectionFiberRecord>
        filtered;
    filtered.reserve(rows.size());
    for (const auto& row : rows) {
        if (keep_steps.contains(row.step)) {
            filtered.push_back(row);
        }
    }
    return filtered;
}

void write_protocol_csv(const std::filesystem::path& path,
                        std::span<const ProtocolPoint> protocol)
{
    std::ofstream out(path);
    out << "step,p,target_drift_m\n";
    out << std::scientific << std::setprecision(8);
    for (const auto& row : protocol) {
        out << row.step << "," << row.p << "," << row.target_drift_m << "\n";
    }
}

void write_comparison_hysteresis_csv(
    const std::filesystem::path& path,
    std::span<const fall_n::table_cyclic_validation::StepRecord> rows)
{
    std::ofstream out(path);
    out << "step,p,drift_m,base_shear_MN\n";
    out << std::scientific << std::setprecision(8);
    for (const auto& row : rows) {
        out << row.step << "," << row.p << "," << row.drift << ","
            << row.base_shear << "\n";
    }
}

void write_section_response_records_csv(
    const std::filesystem::path& path,
    std::span<const ReducedRCColumnSectionResponseRecord> rows)
{
    std::ofstream out(path);
    out << "step,p,drift_m,section_gp,xi,axial_strain,curvature_y,curvature_z,"
           "axial_force_MN,moment_y_MNm,moment_z_MNm,tangent_ea,"
           "tangent_eiy,tangent_eiz,tangent_eiy_direct_raw,"
           "tangent_eiz_direct_raw,raw_k00,raw_k0y,raw_ky0,raw_kyy\n";
    out << std::scientific << std::setprecision(8);
    for (const auto& row : rows) {
        out << row.step << "," << row.p << "," << row.drift << ","
            << row.section_gp << "," << row.xi << "," << row.axial_strain << ","
            << row.curvature_y << "," << row.curvature_z << ","
            << row.axial_force << "," << row.moment_y << "," << row.moment_z
            << "," << row.tangent_ea << "," << row.tangent_eiy << ","
            << row.tangent_eiz << "," << row.tangent_eiy_direct_raw << ","
            << row.tangent_eiz_direct_raw << "," << row.raw_tangent_k00 << ","
            << row.raw_tangent_k0y << "," << row.raw_tangent_ky0 << ","
            << row.raw_tangent_kyy << "\n";
    }
}

void write_comparison_moment_curvature_csv(
    const std::filesystem::path& path,
    std::span<const ReducedRCColumnSectionResponseRecord> rows)
{
    if (rows.empty()) {
        std::ofstream out(path);
        out << "step,p,drift_m,section_gp,xi,curvature_y,moment_y_MNm,"
               "axial_force_MN,tangent_eiy\n";
        return;
    }

    auto controlling_gp = rows.front().section_gp;
    auto controlling_xi = rows.front().xi;
    for (const auto& row : rows) {
        if (row.xi < controlling_xi ||
            (row.xi == controlling_xi && row.section_gp < controlling_gp)) {
            controlling_gp = row.section_gp;
            controlling_xi = row.xi;
        }
    }

    std::ofstream out(path);
    out << "step,p,drift_m,section_gp,xi,curvature_y,moment_y_MNm,"
           "axial_force_MN,tangent_eiy\n";
    out << std::scientific << std::setprecision(8);
    for (const auto& row : rows) {
        if (row.section_gp != controlling_gp) {
            continue;
        }
        out << row.step << "," << row.p << "," << row.drift << ","
            << row.section_gp << "," << row.xi << "," << row.curvature_y << ","
            << row.moment_y << "," << row.axial_force << ","
            << row.tangent_eiy << "\n";
    }

    std::println(
        "  Comparison moment-curvature: {} (section_gp={}, xi={:+.6f})",
        path.string(),
        controlling_gp,
        controlling_xi);
}

void write_control_state_csv(
    const std::filesystem::path& path,
    std::span<const fall_n::validation_reboot::ReducedRCColumnStructuralControlStateRecord>
        rows)
{
    std::ofstream out(path);
    out << "runtime_step,step,p,runtime_p,target_drift_m,actual_tip_drift_m,"
           "actual_tip_total_state_drift_m,imposed_vs_total_state_tip_drift_error_m,"
           "prescribed_top_bending_rotation_rad,"
           "actual_top_bending_rotation_rad,"
           "imposed_vs_total_state_top_bending_rotation_error_rad,"
           "top_axial_displacement_m,base_shear_MN,base_axial_reaction_MN,"
           "stage,target_increment_direction,actual_increment_direction,"
           "protocol_branch_id,reversal_index,branch_step_index,"
           "accepted_substep_count,max_bisection_level,newton_iterations,"
           "newton_iterations_per_substep,solver_profile_attempt_count,"
           "solver_profile_label,solver_snes_type,solver_linesearch_type,"
           "solver_ksp_type,solver_pc_type,last_snes_reason,last_function_norm,"
           "accepted_by_small_residual_policy,"
           "accepted_function_norm_threshold,converged\n";
    out << std::scientific << std::setprecision(8);
    for (const auto& row : rows) {
        out << row.runtime_step << ","
            << row.step << ","
            << row.p << ","
            << row.runtime_p << ","
            << row.target_drift << ","
            << row.actual_tip_lateral_displacement << ","
            << row.actual_tip_lateral_total_state_displacement << ","
            << row.imposed_vs_total_state_tip_displacement_error << ","
            << row.prescribed_top_bending_rotation << ","
            << row.actual_top_bending_rotation << ","
            << row.imposed_vs_total_state_top_bending_rotation_error << ","
            << row.top_axial_displacement << ","
            << row.base_shear << ","
            << row.base_axial_reaction << ","
            << (row.preload_equilibrated ? "preload_equilibrated" : "lateral_branch") << ","
            << row.target_increment_direction << ","
            << row.actual_increment_direction << ","
            << row.protocol_branch_id << ","
            << row.reversal_index << ","
            << row.branch_step_index << ","
            << row.accepted_substep_count << ","
            << row.max_bisection_level << ","
            << row.newton_iterations << ","
            << row.newton_iterations_per_substep << ","
            << row.solver_profile_attempt_count << ","
            << row.solver_profile_label << ","
            << row.solver_snes_type << ","
            << row.solver_linesearch_type << ","
            << row.solver_ksp_type << ","
            << row.solver_pc_type << ","
            << row.last_snes_reason << ","
            << row.last_function_norm << ","
            << (row.accepted_by_small_residual_policy ? 1 : 0) << ","
            << row.accepted_function_norm_threshold << ","
            << (row.converged ? 1 : 0)
            << "\n";
    }
}

void write_section_fiber_history_csv(
    const std::filesystem::path& path,
    std::span<
        const fall_n::validation_reboot::ReducedRCColumnStructuralSectionFiberRecord>
        rows)
{
    std::ofstream out(path);
    out << "step,p,drift_m,section_gp,xi,axial_strain,curvature_y,"
           "zero_curvature_anchor,fiber_index,y,z,area,zone,material_role,"
           "strain_xx,stress_xx_MPa,tangent_xx_MPa,axial_force_contribution_MN,"
           "moment_y_contribution_MNm,raw_k00_contribution,"
           "raw_k0y_contribution,raw_kyy_contribution,"
           "history_state_code,history_min_strain,history_min_stress_MPa,"
           "history_closure_strain,history_max_tensile_strain,"
           "history_max_tensile_stress_MPa,history_committed_strain,"
           "history_committed_stress_MPa,history_cracked\n";
    out << std::scientific << std::setprecision(8);
    for (const auto& row : rows) {
        out << row.step << ","
            << row.p << ","
            << row.drift << ","
            << row.section_gp << ","
            << row.xi << ","
            << row.axial_strain << ","
            << row.curvature_y << ","
            << (row.zero_curvature_anchor ? 1 : 0) << ","
            << row.fiber_index << ","
            << row.y << ","
            << row.z << ","
            << row.area << ","
            << std::string{fall_n::to_string(row.zone)} << ","
            << std::string{fall_n::to_string(row.material_role)} << ","
            << row.strain_xx << ","
            << row.stress_xx << ","
            << row.tangent_xx << ","
            << row.axial_force_contribution << ","
            << row.moment_y_contribution << ","
            << row.raw_tangent_k00_contribution << ","
            << row.raw_tangent_k0y_contribution << ","
            << row.raw_tangent_kyy_contribution << ","
            << row.history_state_code << ","
            << row.history_min_strain << ","
            << row.history_min_stress << ","
            << row.history_closure_strain << ","
            << row.history_max_tensile_strain << ","
            << row.history_max_tensile_stress << ","
            << row.history_committed_strain << ","
            << row.history_committed_stress << ","
            << (row.history_cracked ? 1 : 0) << "\n";
    }
}

void write_preload_state_json(
    const std::filesystem::path& path,
    std::span<const fall_n::validation_reboot::ReducedRCColumnStructuralControlStateRecord>
        control_rows,
    std::span<const ReducedRCColumnSectionResponseRecord> section_rows)
{
    const auto preload_it = std::ranges::find_if(
        control_rows,
        [](const auto& row) { return row.step == 0; });

    std::ofstream out(path);
    out << "{\n";
    if (preload_it == control_rows.end()) {
        out << "  \"status\": \"unavailable\"\n}\n";
        return;
    }

    double mean_axial_strain = 0.0;
    double mean_axial_force = 0.0;
    double mean_tangent_eiy = 0.0;
    int section_count = 0;
    for (const auto& row : section_rows) {
        if (row.step != preload_it->step) {
            continue;
        }
        mean_axial_strain += row.axial_strain;
        mean_axial_force += row.axial_force;
        mean_tangent_eiy += row.tangent_eiy;
        ++section_count;
    }
    if (section_count > 0) {
        mean_axial_strain /= static_cast<double>(section_count);
        mean_axial_force /= static_cast<double>(section_count);
        mean_tangent_eiy /= static_cast<double>(section_count);
    }

    out << "  \"status\": \"available\",\n"
        << "  \"step\": " << preload_it->step << ",\n"
        << "  \"p\": " << preload_it->p << ",\n"
        << "  \"target_drift_m\": " << preload_it->target_drift << ",\n"
        << "  \"actual_tip_drift_m\": "
        << preload_it->actual_tip_lateral_displacement << ",\n"
        << "  \"actual_tip_total_state_drift_m\": "
        << preload_it->actual_tip_lateral_total_state_displacement << ",\n"
        << "  \"imposed_vs_total_state_tip_drift_error_m\": "
        << preload_it->imposed_vs_total_state_tip_displacement_error << ",\n"
        << "  \"top_axial_displacement_m\": " << preload_it->top_axial_displacement
        << ",\n"
        << "  \"base_shear_MN\": " << preload_it->base_shear << ",\n"
        << "  \"base_axial_reaction_MN\": " << preload_it->base_axial_reaction
        << ",\n"
        << "  \"accepted_substep_count\": " << preload_it->accepted_substep_count
        << ",\n"
        << "  \"max_bisection_level\": " << preload_it->max_bisection_level
        << ",\n"
        << "  \"newton_iterations\": " << preload_it->newton_iterations << ",\n"
        << "  \"newton_iterations_per_substep\": "
        << preload_it->newton_iterations_per_substep << ",\n"
        << "  \"last_snes_reason\": " << preload_it->last_snes_reason << ",\n"
        << "  \"last_function_norm\": " << preload_it->last_function_norm << ",\n"
        << "  \"mean_section_axial_strain\": " << mean_axial_strain << ",\n"
        << "  \"mean_section_axial_force_MN\": " << mean_axial_force << ",\n"
        << "  \"mean_section_tangent_eiy\": " << mean_tangent_eiy << ",\n"
        << "  \"section_station_count\": " << section_count << "\n}\n";
}

void write_section_layout_csv(
    const std::filesystem::path& path,
    const fall_n::validation_reboot::ReducedRCColumnReferenceSpec& reference_spec)
{
    const auto layout = fall_n::build_rc_column_fiber_layout(
        fall_n::validation_reboot::to_rc_column_section_spec(reference_spec));

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

void write_station_layout_csv(
    const std::filesystem::path& path,
    std::span<const ReducedRCColumnSectionResponseRecord> rows)
{
    std::vector<std::pair<std::size_t, double>> stations;
    stations.reserve(rows.size());
    for (const auto& row : rows) {
        const auto candidate = std::pair{row.section_gp, row.xi};
        if (std::ranges::find(stations, candidate) == stations.end()) {
            stations.push_back(candidate);
        }
    }
    std::ranges::sort(stations);

    std::ofstream out(path);
    out << "section_gp,xi\n";
    out << std::scientific << std::setprecision(8);
    for (const auto& [section_gp, xi] : stations) {
        out << section_gp << "," << xi << "\n";
    }
}

void write_runtime_manifest(
    const std::filesystem::path& path,
    const CliOptions& options,
    const ReducedRCColumnStructuralRunSpec& spec,
    const CyclicValidationRunConfig& cfg,
    std::span<const ProtocolPoint> comparison_protocol,
    std::span<const fall_n::table_cyclic_validation::StepRecord> comparison_hysteresis,
    std::span<const ReducedRCColumnSectionResponseRecord> comparison_sections,
    std::span<const fall_n::validation_reboot::ReducedRCColumnStructuralControlStateRecord>
        comparison_control_states,
    std::span<
        const fall_n::validation_reboot::ReducedRCColumnStructuralSectionFiberRecord>
        comparison_fiber_states,
    const ReducedRCColumnStructuralRunResult& result)
{
    (void)cfg;
    const auto input_surface =
        fall_n::validation_reboot::make_structural_benchmark_input_surface(
            parse_benchmark_analysis_kind(options.analysis));
    const auto local_model_taxonomy =
        fall_n::validation_reboot::describe_reduced_rc_column_structural_local_model(
            spec);
    const auto steel_area_summary =
        fall_n::validation_reboot::
            describe_reduced_rc_column_structural_steel_area(spec.reference_spec);
    const auto section_spec =
        fall_n::validation_reboot::to_rc_column_section_spec(spec.reference_spec);
    const auto& section_mesh = spec.reference_spec.section_mesh;
    std::ofstream out(path);
    out << std::setprecision(12);
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
        << "  \"analysis\": \"" << json_escape(options.analysis) << "\",\n"
        << "  \"material_mode\": \"" << json_escape(options.material_mode)
        << "\",\n"
        << "  \"beam_nodes\": " << options.beam_nodes << ",\n"
        << "  \"structural_element_count\": "
        << options.structural_element_count << ",\n"
        << "  \"beam_integration\": \"" << beam_integration_key(options.beam_integration)
        << "\",\n"
        << "  \"clamp_top_bending_rotation\": "
        << (options.clamp_top_bending_rotation ? "true" : "false") << ",\n"
        << "  \"prescribe_top_bending_rotation_from_drift\": "
        << (options.prescribe_top_bending_rotation_from_drift ? "true" : "false")
        << ",\n"
        << "  \"top_bending_rotation_drift_ratio\": "
        << options.top_bending_rotation_drift_ratio << ",\n"
        << "  \"element_formulation\": \"TimoshenkoBeamN mixed-interpolation displacement-based beam slice\",\n"
        << "  \"element_assembly_policy\": \"DirectAssembly (no element-level static condensation in this validation slice)\",\n"
        << "  \"beam_theory_note\": \"The current reduced structural benchmark uses a mixed-interpolation Timoshenko beam with explicit shear strains and direct global assembly; no element-level internal-DOF condensation is active on this slice.\",\n"
        << "  \"section_tangent_policy\": \"primary tangent_eiy is the axial-force-condensed dMy/dkappa_y|N=const; raw Kyy and coupling terms are exported separately for audit\",\n"
        << "  \"reinforcement_area\": {\n"
        << "    \"bar_count\": "
        << steel_area_summary.longitudinal_bar_count << ",\n"
        << "    \"single_bar_area_m2\": "
        << steel_area_summary.single_bar_area_m2 << ",\n"
        << "    \"total_steel_area_m2\": "
        << steel_area_summary.total_longitudinal_steel_area_m2 << ",\n"
        << "    \"gross_section_area_m2\": "
        << steel_area_summary.gross_section_area_m2 << ",\n"
        << "    \"steel_ratio\": "
        << steel_area_summary.longitudinal_steel_ratio << "\n"
        << "  },\n"
        << "  \"section_fiber_mesh\": {\n"
        << "    \"profile\": \""
        << section_fiber_profile_key(section_mesh) << "\",\n"
        << "    \"cover_top_bottom_ny\": "
        << section_mesh.cover_top_bottom_ny << ",\n"
        << "    \"cover_top_bottom_nz\": "
        << section_mesh.cover_top_bottom_nz << ",\n"
        << "    \"cover_side_ny\": " << section_mesh.cover_side_ny << ",\n"
        << "    \"cover_side_nz\": " << section_mesh.cover_side_nz << ",\n"
        << "    \"core_ny\": " << section_mesh.core_ny << ",\n"
        << "    \"core_nz\": " << section_mesh.core_nz << ",\n"
        << "    \"concrete_fiber_count\": "
        << fall_n::rc_column_concrete_fiber_count(section_mesh) << ",\n"
        << "    \"total_section_fiber_count\": "
        << fall_n::rc_column_fiber_count(section_spec) << ",\n"
        << "    \"invariant_reinforcement_policy\": "
        << "\"steel bar area and bar coordinates are independent of concrete fiber refinement\"\n"
        << "  },\n"
        << "  \"continuation_kind\": \""
        << json_escape(fall_n::validation_reboot::to_string(options.continuation_kind))
        << "\",\n"
        << "  \"solver_policy_kind\": \""
        << json_escape(
               fall_n::validation_reboot::to_string(spec.solver_policy_kind))
        << "\",\n"
        << "  \"solver_policy\": {\n"
        << "    \"increment_control\": \"adaptive requested-step cutback over monotone pseudo-time with PETSc SNES profile cascade\",\n"
        << "    \"profiles\": [\n"
        ;
    const auto profile_labels =
        [&]() -> std::vector<std::string_view> {
            using Kind =
                fall_n::validation_reboot::ReducedRCColumnSolverPolicyKind;
            switch (spec.solver_policy_kind) {
                case Kind::canonical_newton_profile_cascade:
                    return {
                        "newton_backtracking",
                        "newton_l2",
                        "newton_trust_region"};
                case Kind::newton_basic_only:
                    return {"newton_basic"};
                case Kind::newton_backtracking_only:
                    return {"newton_backtracking"};
                case Kind::newton_l2_only:
                    return {"newton_l2"};
                case Kind::newton_l2_lu_symbolic_reuse_only:
                    return {"newton_l2_lu_symbolic_reuse"};
                case Kind::newton_l2_gmres_ilu1_only:
                    return {"newton_l2_gmres_ilu1"};
                case Kind::newton_trust_region_only:
                    return {"newton_trust_region"};
                case Kind::newton_trust_region_dogleg_only:
                    return {"newton_trust_region_dogleg"};
                case Kind::quasi_newton_only:
                    return {"quasi_newton"};
                case Kind::nonlinear_gmres_only:
                    return {"nonlinear_gmres"};
                case Kind::nonlinear_conjugate_gradient_only:
                    return {"nonlinear_conjugate_gradient"};
                case Kind::anderson_acceleration_only:
                    return {"anderson_acceleration"};
                case Kind::nonlinear_richardson_only:
                    return {"nonlinear_richardson"};
            }
            return {"newton_backtracking"};
        }();
    for (std::size_t i = 0; i < profile_labels.size(); ++i) {
        out << "      \"" << profile_labels[i] << "\"";
        out << (i + 1 < profile_labels.size() ? ",\n" : "\n");
    }
    out << "    ]\n"
        << "  },\n"
        << "  \"axial_compression_mn\": " << std::setprecision(12)
        << options.axial_compression_mn << ",\n"
        << "  \"protocol_point_count\": " << comparison_protocol.size() << ",\n"
        << "  \"hysteresis_point_count\": " << result.hysteresis_records.size() << ",\n"
        << "  \"section_record_count\": " << result.section_response_records.size() << ",\n"
        << "  \"control_state_record_count\": " << result.control_state_records.size()
        << ",\n"
        << "  \"fiber_history_record_count\": " << result.fiber_history_records.size()
        << ",\n"
        << "  \"element_tangent_audit_record_count\": "
        << result.element_tangent_audit_records.size() << ",\n"
        << "  \"section_tangent_audit_record_count\": "
        << result.section_tangent_audit_records.size() << ",\n"
        << "  \"comparison_hysteresis_point_count\": " << comparison_hysteresis.size()
        << ",\n"
        << "  \"comparison_section_record_count\": " << comparison_sections.size()
        << ",\n"
        << "  \"comparison_control_state_point_count\": "
        << comparison_control_states.size()
        << ",\n"
        << "  \"comparison_fiber_history_point_count\": "
        << comparison_fiber_states.size()
        << ",\n"
        << "  \"failed_attempt_section_record_count\": "
        << result.failed_attempt_section_response_records.size()
        << ",\n"
        << "  \"failed_attempt_fiber_history_point_count\": "
        << result.failed_attempt_fiber_history_records.size()
        << ",\n"
        << "  \"has_failed_attempt_element_tangent_audit\": "
        << (result.has_failed_attempt_element_tangent_audit ? "true" : "false")
        << ",\n"
        << "  \"has_failed_attempt_section_tangent_audit\": "
        << (result.has_failed_attempt_section_tangent_audit ? "true" : "false")
        << ",\n"
        << "  \"comparison_branch_kind\": \""
        << (options.analysis == "monotonic"
                ? "declared_monotonic_history"
                : "full_declared_history")
        << "\",\n"
        << "  \"timing\": {\n"
        << "    \"total_wall_seconds\": " << result.timing.total_wall_seconds << ",\n"
        << "    \"analysis_wall_seconds\": " << result.timing.solve_wall_seconds << ",\n"
        << "    \"output_write_wall_seconds\": " << result.timing.output_write_wall_seconds
        << "\n"
        << "  },\n"
        << "  \"benchmark_note\": \"";

    if (options.analysis == "monotonic") {
        out << "The structural baseline compares the full declared monotonic "
               "history against the external monotonic reference.\"";
    } else {
        out << "The structural baseline compares the full declared cyclic history "
               "against the external computational reference.\"";
    }

    if (!result.completed_successfully &&
        result.has_failed_attempt_control_state) {
        const auto& row = result.failed_attempt_control_state;
        out << ",\n"
            << "  \"failed_attempt\": {\n"
            << "    \"runtime_step\": " << row.runtime_step << ",\n"
            << "    \"step\": " << row.step << ",\n"
            << "    \"p\": " << row.p << ",\n"
            << "    \"runtime_p\": " << row.runtime_p << ",\n"
            << "    \"target_drift_m\": " << row.target_drift << ",\n"
            << "    \"actual_tip_drift_m\": "
            << row.actual_tip_lateral_displacement << ",\n"
            << "    \"actual_tip_total_state_drift_m\": "
            << row.actual_tip_lateral_total_state_displacement << ",\n"
            << "    \"imposed_vs_total_state_tip_drift_error_m\": "
            << row.imposed_vs_total_state_tip_displacement_error << ",\n"
            << "    \"prescribed_top_bending_rotation_rad\": "
            << row.prescribed_top_bending_rotation << ",\n"
            << "    \"actual_top_bending_rotation_rad\": "
            << row.actual_top_bending_rotation << ",\n"
            << "    \"imposed_vs_total_state_top_bending_rotation_error_rad\": "
            << row.imposed_vs_total_state_top_bending_rotation_error << ",\n"
            << "    \"accepted_substep_count\": " << row.accepted_substep_count << ",\n"
            << "    \"max_bisection_level\": " << row.max_bisection_level << ",\n"
            << "    \"newton_iterations\": " << row.newton_iterations << ",\n"
            << "    \"newton_iterations_per_substep\": "
            << row.newton_iterations_per_substep << ",\n"
            << "    \"last_snes_reason\": " << row.last_snes_reason << ",\n"
            << "    \"last_function_norm\": " << row.last_function_norm << ",\n"
            << "    \"accepted_by_small_residual_policy\": "
            << (row.accepted_by_small_residual_policy ? "true" : "false")
            << ",\n"
            << "    \"accepted_function_norm_threshold\": "
            << row.accepted_function_norm_threshold << ",\n"
            << "    \"failed_attempt_section_response_csv\": "
            << "\"" << json_escape(
                   (path.parent_path() / "failed_attempt_section_response.csv")
                       .string())
            << "\",\n"
            << "    \"failed_attempt_moment_curvature_base_csv\": "
            << "\"" << json_escape(
                   (path.parent_path() / "failed_attempt_moment_curvature_base.csv")
                       .string())
            << "\",\n"
            << "    \"failed_attempt_section_fiber_state_history_csv\": "
            << "\"" << json_escape(
                   (path.parent_path() /
                    "failed_attempt_section_fiber_state_history.csv")
                       .string())
            << "\",\n"
            << "    \"failed_attempt_element_tangent_audit_csv\": "
            << "\"" << json_escape(
                   (path.parent_path() /
                    "failed_attempt_element_tangent_audit.csv")
                       .string())
            << "\",\n"
            << "    \"failed_attempt_section_tangent_audit_csv\": "
            << "\"" << json_escape(
                   (path.parent_path() /
                    "failed_attempt_section_tangent_audit.csv")
                       .string())
            << "\"\n"
            << "  }\n";
    }

    out << "\n}\n";
}

} // namespace

int main(int argc, char** argv)
{
    bool petsc_initialized = false;

    try {
        PetscInitializeNoArguments();
        petsc_initialized = true;
        const auto options = parse_args(argc, argv);
        if (options.material_mode != "nonlinear" &&
            options.material_mode != "elasticized") {
            throw std::invalid_argument(
                "--material-mode must be either 'nonlinear' or 'elasticized'.");
        }
        const auto protocol = make_protocol(options);
        auto reference_spec =
            fall_n::validation_reboot::default_reduced_rc_column_reference_spec_v;
        reference_spec.section_mesh =
            parse_section_fiber_profile(options.section_fiber_profile);
        auto spec = ReducedRCColumnStructuralRunSpec{
            .material_mode =
                options.material_mode == "elasticized"
                    ? fall_n::validation_reboot::
                          ReducedRCColumnStructuralMaterialMode::elasticized
                    : fall_n::validation_reboot::
                          ReducedRCColumnStructuralMaterialMode::nonlinear,
            .beam_nodes = options.beam_nodes,
            .structural_element_count = options.structural_element_count,
            .beam_axis_quadrature_family = options.beam_integration,
            .clamp_top_bending_rotation = options.clamp_top_bending_rotation,
            .prescribe_top_bending_rotation_from_drift =
                options.prescribe_top_bending_rotation_from_drift,
            .top_bending_rotation_drift_ratio =
                options.top_bending_rotation_drift_ratio,
            .axial_compression_force_mn = options.axial_compression_mn,
            .use_equilibrated_axial_preload_stage =
                options.use_equilibrated_axial_preload_stage,
            .axial_preload_steps = options.axial_preload_steps,
            .continuation_kind = options.continuation_kind,
            .solver_policy_kind = parse_solver_policy_kind(options.solver_policy),
            .continuation_segment_substep_factor =
                options.continuation_segment_substep_factor,
            .write_hysteresis_csv = true,
            .write_section_response_csv = true,
            .write_element_tangent_audit_csv =
                options.write_element_tangent_audit_csv,
            .print_progress = options.print_progress,
            .reference_spec = reference_spec,
        };

        const auto out_dir = std::filesystem::path{options.output_dir};
        std::filesystem::create_directories(out_dir);

        const auto result =
            fall_n::validation_reboot::run_reduced_rc_column_small_strain_beam_case_result(
                spec,
                out_dir.string(),
                protocol);

        const auto comparison_protocol = build_comparison_protocol(options, protocol);
        const auto comparison_hysteresis =
            select_comparison_hysteresis(options, result.hysteresis_records);
        const auto comparison_sections =
            select_comparison_section_rows(
                result.section_response_records, comparison_hysteresis);
        const auto comparison_control_states =
            select_comparison_control_states(
                result.control_state_records, comparison_hysteresis);
        const auto comparison_fiber_states =
            select_comparison_fiber_states(
                result.fiber_history_records, comparison_hysteresis);

        write_protocol_csv(out_dir / "comparison_protocol.csv", comparison_protocol);
        write_comparison_hysteresis_csv(
            out_dir / "comparison_hysteresis.csv", comparison_hysteresis);
        write_comparison_moment_curvature_csv(
            out_dir / "comparison_moment_curvature_base.csv", comparison_sections);
        write_control_state_csv(
            out_dir / "control_state.csv", comparison_control_states);
        write_section_fiber_history_csv(
            out_dir / "comparison_section_fiber_state_history.csv",
            comparison_fiber_states);
        if (!result.failed_attempt_section_response_records.empty()) {
            write_section_response_records_csv(
                out_dir / "failed_attempt_section_response.csv",
                result.failed_attempt_section_response_records);
            write_comparison_moment_curvature_csv(
                out_dir / "failed_attempt_moment_curvature_base.csv",
                result.failed_attempt_section_response_records);
        }
        if (!result.failed_attempt_fiber_history_records.empty()) {
            write_section_fiber_history_csv(
                out_dir / "failed_attempt_section_fiber_state_history.csv",
                result.failed_attempt_fiber_history_records);
        }
        write_preload_state_json(
            out_dir / "preload_state.json",
            comparison_control_states,
            comparison_sections);
        write_section_layout_csv(out_dir / "section_layout.csv", spec.reference_spec);
        write_station_layout_csv(out_dir / "section_station_layout.csv", comparison_sections);
        write_runtime_manifest(
            out_dir / "runtime_manifest.json",
            options,
            spec,
            protocol,
            comparison_protocol,
            comparison_hysteresis,
            comparison_sections,
            comparison_control_states,
            comparison_fiber_states,
            result);

        if (!result.completed_successfully) {
            std::println(
                "Reduced RC column reference benchmark did not complete on the "
                "declared continuation path.");
            if (petsc_initialized) {
                PetscFinalize();
            }
            return 2;
        }
    } catch (const std::exception& exc) {
        std::println(stderr, "{}", exc.what());
        if (petsc_initialized) {
            PetscFinalize();
        }
        return 1;
    }

    if (petsc_initialized) {
        PetscFinalize();
    }
    return 0;
}
