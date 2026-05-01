// Plan v2 §Fase 4B — Multiscale local-site BATCH plan driver.
//
// Standalone driver. Consumes any structural-history CSV, builds a
// `ReducedRCMultiscaleReplayPlan`, derives a `ReducedRCMultiscaleRuntimePolicy`,
// and finally produces a `ReducedRCLocalSiteBatchPlan`. Emits:
//   - multiscale_replay_plan.json
//   - multiscale_runtime_policy.json
//   - multiscale_local_site_batch_plan.json
//
// CLI:
//   main_local_site_batch --input <csv> --output-dir <dir>
//                         [--site-index N] [--z-over-l F]
//                         [--characteristic-length-mm F]
//                         [--max-concurrent-sites N]
//                         [--hot-state-budget-mib F]
//                         [--direct-lu-budget-mib F]

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

#include "src/validation/MultiscaleReplayDriverHelpers.hh"
#include "src/validation/ReducedRCLocalSiteBatchPlan.hh"
#include "src/validation/ReducedRCMultiscaleReplayPlan.hh"
#include "src/validation/ReducedRCMultiscaleRuntimePolicy.hh"

namespace {

struct Options {
    std::filesystem::path input_csv{};
    std::filesystem::path output_dir{};
    std::size_t site_index{0};
    double z_over_l{0.02};
    double characteristic_length_mm{100.0};
    std::size_t max_concurrent_sites{0};
    double hot_state_budget_mib{1024.0};
    double direct_lu_budget_mib{512.0};
    std::size_t num_sites{1};
    std::vector<double> site_z_list{};
    std::vector<double> site_scale_list{};
};

[[nodiscard]] std::vector<double> parse_double_list(std::string_view s) {
    std::vector<double> out;
    std::string buf;
    for (char c : s) {
        if (c == ',' || c == ';') {
            if (!buf.empty()) { out.push_back(std::strtod(buf.c_str(), nullptr)); buf.clear(); }
        } else if (!std::isspace(static_cast<unsigned char>(c))) {
            buf.push_back(c);
        }
    }
    if (!buf.empty()) out.push_back(std::strtod(buf.c_str(), nullptr));
    return out;
}

void print_usage(const char* a0) {
    std::fprintf(stderr,
        "Usage: %s --input <csv> --output-dir <dir> "
        "[--site-index N] [--z-over-l F] [--characteristic-length-mm F] "
        "[--max-concurrent-sites N] [--hot-state-budget-mib F] "
        "[--direct-lu-budget-mib F]\n", a0);
}

[[nodiscard]] bool parse_args(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        std::string_view a = argv[i];
        if (a == "--input" && i + 1 < argc) o.input_csv = argv[++i];
        else if (a == "--output-dir" && i + 1 < argc) o.output_dir = argv[++i];
        else if (a == "--site-index" && i + 1 < argc) o.site_index = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--z-over-l" && i + 1 < argc) o.z_over_l = std::strtod(argv[++i], nullptr);
        else if (a == "--characteristic-length-mm" && i + 1 < argc) o.characteristic_length_mm = std::strtod(argv[++i], nullptr);
        else if (a == "--max-concurrent-sites" && i + 1 < argc) o.max_concurrent_sites = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--hot-state-budget-mib" && i + 1 < argc) o.hot_state_budget_mib = std::strtod(argv[++i], nullptr);
        else if (a == "--direct-lu-budget-mib" && i + 1 < argc) o.direct_lu_budget_mib = std::strtod(argv[++i], nullptr);
        else if (a == "--num-sites" && i + 1 < argc) o.num_sites = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--site-z-list" && i + 1 < argc) o.site_z_list = parse_double_list(argv[++i]);
        else if (a == "--site-scale-list" && i + 1 < argc) o.site_scale_list = parse_double_list(argv[++i]);
        else if (a == "--help" || a == "-h") { print_usage(argv[0]); return false; }
        else { std::fprintf(stderr, "unknown arg: %s\n", argv[i]); print_usage(argv[0]); return false; }
    }
    if (o.input_csv.empty() || o.output_dir.empty()) { print_usage(argv[0]); return false; }
    return true;
}

void emit_batch_plan_json(const std::filesystem::path& path,
                          const fall_n::ReducedRCLocalSiteBatchPlan& bp) {
    std::ofstream f(path);
    f << "{\n"
      << "  \"schema\": \"reduced_rc_multiscale_local_site_batch_plan_v1\",\n"
      << "  \"ready_for_local_site_batch\": " << (bp.ready_for_local_site_batch ? "true" : "false") << ",\n"
      << "  \"ready_for_many_site_replay\": " << (bp.ready_for_many_site_replay ? "true" : "false") << ",\n"
      << "  \"vtk_time_series_required\": " << (bp.vtk_time_series_required ? "true" : "false") << ",\n"
      << "  \"bounded_seed_cache_required\": " << (bp.bounded_seed_cache_required ? "true" : "false") << ",\n"
      << "  \"iterative_preconditioner_expected\": " << (bp.iterative_preconditioner_expected ? "true" : "false") << ",\n"
      << "  \"selected_site_count\": " << bp.selected_site_count << ",\n"
      << "  \"batch_count\": " << bp.batch_count << ",\n"
      << "  \"max_concurrent_sites\": " << bp.max_concurrent_sites << ",\n"
      << "  \"recommended_site_threads\": " << bp.recommended_site_threads << ",\n"
      << "  \"executor_kind\": \"" << fall_n::to_string(bp.executor_kind) << "\",\n"
      << "  \"total_estimated_hot_state_mib\": " << bp.total_estimated_hot_state_mib << ",\n"
      << "  \"max_batch_hot_state_mib\": " << bp.max_batch_hot_state_mib << ",\n"
      << "  \"total_direct_factorization_risk_mib\": " << bp.total_direct_factorization_risk_mib << ",\n"
      << "  \"rationale\": \"" << bp.rationale << "\",\n"
      << "  \"rows\": [\n";
    for (std::size_t i = 0; i < bp.rows.size(); ++i) {
        const auto& r = bp.rows[i];
        f << "    {\n"
          << "      \"batch_index\": " << r.batch_index << ",\n"
          << "      \"slot_index\": " << r.slot_index << ",\n"
          << "      \"site_index\": " << r.site_index << ",\n"
          << "      \"z_over_l\": " << r.z_over_l << ",\n"
          << "      \"activation_score\": " << r.activation_score << ",\n"
          << "      \"estimated_hot_state_mib\": " << r.estimated_hot_state_mib << ",\n"
          << "      \"direct_factorization_risk_mib\": " << r.direct_factorization_risk_mib << ",\n"
          << "      \"solver_kind\": \"" << fall_n::to_string(r.solver_kind) << "\",\n"
          << "      \"seed_restore_required\": " << (r.seed_restore_required ? "true" : "false") << ",\n"
          << "      \"warm_start_required\": " << (r.warm_start_required ? "true" : "false") << ",\n"
          << "      \"vtk_time_series_required\": " << (r.vtk_time_series_required ? "true" : "false") << ",\n"
          << "      \"rationale\": \"" << r.rationale << "\"\n"
          << "    }" << (i + 1 == bp.rows.size() ? "\n" : ",\n");
    }
    f << "  ],\n  \"batches\": [\n";
    for (std::size_t i = 0; i < bp.batches.size(); ++i) {
        const auto& b = bp.batches[i];
        f << "    {\n"
          << "      \"batch_index\": " << b.batch_index << ",\n"
          << "      \"site_count\": " << b.site_count << ",\n"
          << "      \"estimated_hot_state_mib\": " << b.estimated_hot_state_mib << ",\n"
          << "      \"direct_factorization_risk_mib\": " << b.direct_factorization_risk_mib << ",\n"
          << "      \"recommended_threads\": " << b.recommended_threads << ",\n"
          << "      \"dominant_solver_kind\": \"" << fall_n::to_string(b.dominant_solver_kind) << "\",\n"
          << "      \"uses_parallel_site_loop\": " << (b.uses_parallel_site_loop ? "true" : "false") << ",\n"
          << "      \"within_hot_state_budget\": " << (b.within_hot_state_budget ? "true" : "false") << ",\n"
          << "      \"direct_lu_within_budget\": " << (b.direct_lu_within_budget ? "true" : "false") << "\n"
          << "    }" << (i + 1 == bp.batches.size() ? "\n" : ",\n");
    }
    f << "  ]\n}\n";
}

}  // namespace

int main(int argc, char** argv) {
    Options o;
    if (!parse_args(argc, argv, o)) return 1;

    namespace fs = std::filesystem;
    fs::create_directories(o.output_dir);

    const auto rows = fall_n::read_structural_history_csv(o.input_csv);
    if (rows.empty()) {
        std::fprintf(stderr, "[fase4b] empty CSV\n");
        return 3;
    }
    std::vector<fall_n::ReducedRCStructuralReplaySample> samples;
    if (o.num_sites <= 1) {
        samples = fall_n::build_replay_samples_from_csv(
            rows, o.site_index, o.z_over_l, o.characteristic_length_mm);
    } else {
        const std::vector<double> default_z = {0.02, 0.10, 0.30, 0.55, 0.85};
        const std::vector<double> default_s = {1.00, 0.70, 0.40, 0.20, 0.10};
        std::vector<fall_n::MultiSiteReplaySpec> specs;
        specs.reserve(o.num_sites);
        for (std::size_t s = 0; s < o.num_sites; ++s) {
            const double z = (!o.site_z_list.empty() && s < o.site_z_list.size())
                ? o.site_z_list[s]
                : (s < default_z.size() ? default_z[s] : default_z.back());
            const double k = (!o.site_scale_list.empty() && s < o.site_scale_list.size())
                ? o.site_scale_list[s]
                : (s < default_s.size() ? default_s[s] : default_s.back());
            specs.push_back({.site_index = s, .z_over_l = z, .demand_scale = k});
        }
        samples = fall_n::build_multi_site_replay_samples_from_csv(
            rows, specs, o.characteristic_length_mm);
    }
    fall_n::ReducedRCMultiscaleReplayPlanSettings rs{};
    rs.max_replay_sites = std::max<std::size_t>(3, o.num_sites);
    const auto plan   = fall_n::make_reduced_rc_multiscale_replay_plan(samples, rs);
    const auto policy = fall_n::make_reduced_rc_multiscale_runtime_policy(plan);

    fall_n::ReducedRCLocalSiteBatchSettings batch_settings{};
    batch_settings.max_concurrent_sites = o.max_concurrent_sites;
    batch_settings.hot_state_budget_mib = o.hot_state_budget_mib;
    batch_settings.direct_lu_factorization_budget_mib = o.direct_lu_budget_mib;
    const auto batch_plan =
        fall_n::make_reduced_rc_local_site_batch_plan(plan, policy, batch_settings);

    emit_batch_plan_json(o.output_dir / "multiscale_local_site_batch_plan.json",
                         batch_plan);

    std::printf("[fase4b] local_site_batch plan emitted | selected=%zu batches=%zu "
                "executor=%s ready=%d output=%s\n",
                batch_plan.selected_site_count, batch_plan.batch_count,
                fall_n::to_string(batch_plan.executor_kind).data(),
                batch_plan.ready_for_local_site_batch,
                o.output_dir.string().c_str());
    return batch_plan.ready_for_local_site_batch ? 0 : 4;
}
