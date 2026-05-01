// Plan v2 §Fase 4A — Multiscale one-way replay PLAN driver.
//
// Standalone driver that consumes a structural-history CSV, builds a
// `ReducedRCMultiscaleReplayPlan` + `ReducedRCMultiscaleRuntimePolicy`, and
// emits the canonical JSON artifacts:
//   - multiscale_replay_plan.json
//   - multiscale_runtime_policy.json
//
// The plan-building logic is otherwise embedded inside the heavy XFEM driver
// (`main_reduced_rc_xfem_reference_benchmark`). This driver factors it out so
// the planner can run on any structural CSV (table_cyclic_validation,
// lshaped_*, multiscale_seismic, etc.) without re-running the local solver.
//
// Input CSV (header-mapped, missing optional columns default to 0.0):
//   required: drift_mm, base_shear_MN
//   optional: p (pseudo-time), curvature_y, moment_y_MN_m, axial_reaction_MN,
//             max_abs_steel_stress_MPa, max_host_damage
//
// CLI:
//   main_multiscale_one_way_replay --input <csv> --output-dir <dir>
//                                  [--site-index N] [--z-over-l F]
//                                  [--characteristic-length-mm F]

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "src/validation/ReducedRCMultiscaleReplayPlan.hh"
#include "src/validation/ReducedRCMultiscaleRuntimePolicy.hh"

namespace {

struct Options {
    std::filesystem::path input_csv{};
    std::filesystem::path output_dir{};
    std::size_t site_index{0};
    double z_over_l{0.02};
    double characteristic_length_mm{100.0};
};

void print_usage(const char* argv0) {
    std::fprintf(stderr,
        "Usage: %s --input <csv> --output-dir <dir> "
        "[--site-index N] [--z-over-l F] [--characteristic-length-mm F]\n",
        argv0);
}

[[nodiscard]] bool parse_args(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        std::string_view a = argv[i];
        auto next = [&](double& v) {
            if (++i >= argc) return false;
            v = std::strtod(argv[i], nullptr);
            return true;
        };
        if (a == "--input" && i + 1 < argc) o.input_csv = argv[++i];
        else if (a == "--output-dir" && i + 1 < argc) o.output_dir = argv[++i];
        else if (a == "--site-index" && i + 1 < argc) o.site_index = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--z-over-l") { if (!next(o.z_over_l)) return false; }
        else if (a == "--characteristic-length-mm") { if (!next(o.characteristic_length_mm)) return false; }
        else if (a == "--help" || a == "-h") { print_usage(argv[0]); return false; }
        else { std::fprintf(stderr, "unknown arg: %s\n", argv[i]); print_usage(argv[0]); return false; }
    }
    if (o.input_csv.empty() || o.output_dir.empty()) { print_usage(argv[0]); return false; }
    return true;
}

struct CsvRow {
    double p{0.0}, drift_mm{0.0}, base_shear_mn{0.0};
    double curvature_y{0.0}, moment_y_mn_m{0.0};
    double steel_stress_mpa{0.0}, damage{0.0};
};

[[nodiscard]] std::vector<CsvRow> read_csv(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) { std::fprintf(stderr, "[fase4a] cannot open %s\n", path.string().c_str()); std::exit(1); }
    std::string header; std::getline(in, header);
    int idx_p = -1, idx_drift = -1, idx_vb = -1;
    int idx_curv = -1, idx_mom = -1, idx_steel = -1, idx_dam = -1;
    {
        std::stringstream ss(header); std::string c; int i = 0;
        while (std::getline(ss, c, ',')) {
            if (c == "p" || c == "pseudo_time") idx_p = i;
            else if (c == "drift_mm") idx_drift = i;
            else if (c == "base_shear_MN" || c == "base_shear_mn") idx_vb = i;
            else if (c == "curvature_y") idx_curv = i;
            else if (c == "moment_y_MN_m" || c == "moment_y_mn_m") idx_mom = i;
            else if (c == "max_abs_steel_stress_MPa" || c == "steel_stress_MPa" ||
                     c == "max_abs_steel_stress_mpa") idx_steel = i;
            else if (c == "max_host_damage" || c == "damage_indicator") idx_dam = i;
            ++i;
        }
    }
    if (idx_drift < 0 || idx_vb < 0) {
        std::fprintf(stderr,
            "[fase4a] CSV missing required columns drift_mm + base_shear_MN\n");
        std::exit(2);
    }
    std::vector<CsvRow> rows;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line); std::string c; int i = 0;
        CsvRow r{};
        while (std::getline(ss, c, ',')) {
            const double v = c.empty() ? 0.0 : std::strtod(c.c_str(), nullptr);
            if (i == idx_p) r.p = v;
            else if (i == idx_drift) r.drift_mm = v;
            else if (i == idx_vb) r.base_shear_mn = v;
            else if (i == idx_curv) r.curvature_y = v;
            else if (i == idx_mom) r.moment_y_mn_m = v;
            else if (i == idx_steel) r.steel_stress_mpa = v;
            else if (i == idx_dam) r.damage = v;
            ++i;
        }
        rows.push_back(r);
    }
    return rows;
}

[[nodiscard]] std::vector<fall_n::ReducedRCStructuralReplaySample>
build_samples(const std::vector<CsvRow>& rows, const Options& o) {
    std::vector<fall_n::ReducedRCStructuralReplaySample> out;
    out.reserve(rows.size());
    const double L_m = std::max(o.characteristic_length_mm / 1000.0, 1.0e-12);
    const double denom = rows.size() > 1 ? static_cast<double>(rows.size() - 1) : 1.0;
    double prev_drift = rows.empty() ? 0.0 : rows.front().drift_mm;
    double prev_vb    = rows.empty() ? 0.0 : rows.front().base_shear_mn;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const auto& r = rows[i];
        const double d_drift = i == 0 ? 0.0 : r.drift_mm - prev_drift;
        const double d_work  = i == 0 ? 0.0 : 0.5 * (r.base_shear_mn + prev_vb) * d_drift;
        // If no curvature column, derive a proxy from drift (chord rotation / L).
        const double curv = r.curvature_y != 0.0
                                ? r.curvature_y
                                : (r.drift_mm / 1000.0) / L_m;
        out.push_back(fall_n::ReducedRCStructuralReplaySample{
            .site_index = o.site_index,
            .pseudo_time = r.p > 0.0 ? r.p : static_cast<double>(i) / denom,
            .physical_time = static_cast<double>(i) / denom,
            .z_over_l = o.z_over_l,
            .drift_mm = r.drift_mm,
            .curvature_y = curv,
            .moment_y_mn_m = r.moment_y_mn_m,
            .base_shear_mn = r.base_shear_mn,
            .steel_stress_mpa = r.steel_stress_mpa,
            .damage_indicator = r.damage,
            .work_increment_mn_mm = d_work,
        });
        prev_drift = r.drift_mm; prev_vb = r.base_shear_mn;
    }
    return out;
}

const char* activation_label(fall_n::ReducedRCReplaySiteActivationKind k) {
    switch (k) {
        case fall_n::ReducedRCReplaySiteActivationKind::monitor_only: return "monitor_only";
        case fall_n::ReducedRCReplaySiteActivationKind::elastic_local_control: return "elastic_local_control";
        case fall_n::ReducedRCReplaySiteActivationKind::xfem_enriched_replay: return "xfem_enriched_replay";
        case fall_n::ReducedRCReplaySiteActivationKind::guarded_two_way_candidate: return "guarded_two_way_candidate";
    }
    return "unknown";
}

void emit_plan_json(const std::filesystem::path& path,
                    const fall_n::ReducedRCMultiscaleReplayPlan& plan) {
    std::ofstream f(path);
    f << "{\n"
      << "  \"schema\": \"reduced_rc_multiscale_one_way_replay_plan_v1\",\n"
      << "  \"scientific_status\": \"one_way_replay_before_two_way_fe2\",\n"
      << "  \"history_sample_count\": " << plan.history_sample_count << ",\n"
      << "  \"candidate_site_count\": " << plan.candidate_site_count << ",\n"
      << "  \"selected_site_count\": " << plan.selected_site_count << ",\n"
      << "  \"ready_for_one_way_replay\": " << (plan.ready_for_one_way_replay ? "true" : "false") << ",\n"
      << "  \"ready_for_two_way_fe2\": " << (plan.ready_for_two_way_fe2 ? "true" : "false") << ",\n"
      << "  \"vtk_contract_satisfied\": " << (plan.vtk_contract_satisfied ? "true" : "false") << ",\n"
      << "  \"execution_policy\": {\n"
      << "    \"seed_state_cache_recommended\": " << (plan.seed_state_cache_recommended ? "true" : "false") << ",\n"
      << "    \"newton_warm_start_recommended\": " << (plan.newton_warm_start_recommended ? "true" : "false") << ",\n"
      << "    \"site_level_openmp_recommended\": " << (plan.site_level_openmp_recommended ? "true" : "false") << ",\n"
      << "    \"avoid_direct_lu_for_batch\": " << (plan.avoid_direct_lu_for_batch ? "true" : "false") << ",\n"
      << "    \"selected_hot_state_mib\": " << plan.selected_hot_state_mib << ",\n"
      << "    \"selected_direct_factorization_risk_mib\": " << plan.selected_direct_factorization_risk_mib << "\n"
      << "  },\n"
      << "  \"sites\": [\n";
    for (std::size_t i = 0; i < plan.sites.size(); ++i) {
        const auto& s = plan.sites[i];
        f << "    {\n"
          << "      \"rank\": " << i << ",\n"
          << "      \"site_index\": " << s.site_index << ",\n"
          << "      \"z_over_l\": " << s.z_over_l << ",\n"
          << "      \"sample_count\": " << s.sample_count << ",\n"
          << "      \"selected_for_replay\": " << (s.selected_for_replay ? "true" : "false") << ",\n"
          << "      \"activation_kind\": \"" << activation_label(s.activation_kind) << "\",\n"
          << "      \"activation_score\": " << s.activation_score << ",\n"
          << "      \"peak_abs_curvature_y\": " << s.peak_abs_curvature_y << ",\n"
          << "      \"peak_abs_moment_y_mn_m\": " << s.peak_abs_moment_y_mn_m << ",\n"
          << "      \"peak_abs_base_shear_mn\": " << s.peak_abs_base_shear_mn << ",\n"
          << "      \"peak_abs_steel_stress_mpa\": " << s.peak_abs_steel_stress_mpa << ",\n"
          << "      \"max_damage_indicator\": " << s.max_damage_indicator << ",\n"
          << "      \"accumulated_abs_work_mn_mm\": " << s.accumulated_abs_work_mn_mm << "\n"
          << "    }" << (i + 1 == plan.sites.size() ? "\n" : ",\n");
    }
    f << "  ]\n}\n";
}

void emit_policy_json(const std::filesystem::path& path,
                      const fall_n::ReducedRCMultiscaleRuntimePolicy& policy) {
    std::ofstream f(path);
    f << "{\n"
      << "  \"schema\": \"reduced_rc_multiscale_runtime_policy_v1\",\n"
      << "  \"ready_for_local_site_batch\": " << (policy.ready_for_local_site_batch ? "true" : "false") << ",\n"
      << "  \"executor_kind\": \"" << fall_n::to_string(policy.executor_kind) << "\",\n"
      << "  \"recommended_site_threads\": " << policy.recommended_site_threads << ",\n"
      << "  \"local_sites_run_in_parallel\": " << (policy.local_sites_run_in_parallel ? "true" : "false") << ",\n"
      << "  \"cache_budget_is_bounded\": " << (policy.cache_budget_is_bounded ? "true" : "false") << ",\n"
      << "  \"direct_lu_kept_as_reference_only\": " << (policy.direct_lu_kept_as_reference_only ? "true" : "false") << ",\n"
      << "  \"iterative_preconditioner_expected\": " << (policy.iterative_preconditioner_expected ? "true" : "false") << ",\n"
      << "  \"rationale\": \"" << policy.rationale << "\",\n"
      << "  \"local_runtime_settings\": {\n"
      << "    \"profiling_enabled\": " << (policy.local_runtime_settings.profiling_enabled ? "true" : "false") << ",\n"
      << "    \"seed_state_reuse_enabled\": " << (policy.local_runtime_settings.seed_state_reuse_enabled ? "true" : "false") << ",\n"
      << "    \"restore_seed_before_solve\": " << (policy.local_runtime_settings.restore_seed_before_solve ? "true" : "false") << ",\n"
      << "    \"max_cached_seed_states\": " << policy.local_runtime_settings.max_cached_seed_states << ",\n"
      << "    \"adaptive_activation_enabled\": " << (policy.local_runtime_settings.adaptive_activation_enabled ? "true" : "false") << "\n"
      << "  }\n}\n";
}

}  // namespace

int main(int argc, char** argv) {
    Options o;
    if (!parse_args(argc, argv, o)) return 1;

    namespace fs = std::filesystem;
    fs::create_directories(o.output_dir);

    const auto rows = read_csv(o.input_csv);
    if (rows.empty()) {
        std::fprintf(stderr, "[fase4a] empty CSV: %s\n", o.input_csv.string().c_str());
        return 3;
    }
    const auto samples = build_samples(rows, o);

    fall_n::ReducedRCMultiscaleReplayPlanSettings settings{};
    settings.max_replay_sites = 3;
    // Default thresholds match the XFEM driver settings (Plan v2 Fase 4 anchor).
    const auto plan = fall_n::make_reduced_rc_multiscale_replay_plan(samples, settings);
    const auto policy = fall_n::make_reduced_rc_multiscale_runtime_policy(plan);

    emit_plan_json(o.output_dir / "multiscale_replay_plan.json", plan);
    emit_policy_json(o.output_dir / "multiscale_runtime_policy.json", policy);

    std::printf("[fase4a] one_way_replay plan emitted | samples=%zu sites=%zu selected=%zu "
                "ready_one_way=%d output=%s\n",
                plan.history_sample_count, plan.candidate_site_count, plan.selected_site_count,
                plan.ready_for_one_way_replay, o.output_dir.string().c_str());
    return 0;
}
