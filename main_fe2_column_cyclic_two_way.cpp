// Plan v2 §Stage B — FE² TWO-WAY cyclic 200 mm column smoke driver.
//
// Extends the Stage A one-way driver with a synthetic staggered macro↔local
// loop. After the injected `FirstInelasticFiberCriterion` triggers on a
// site, the driver iterates a damage-degraded condensed tangent + force
// vector with relaxation until the Frobenius residual between successive
// updates drops below the staggered tolerance, mirroring the
// `MAX_STAGGERED_ITER`/`STAGGERED_TOL`/`STAGGERED_RELAX` parameters used in
// `main_lshaped_multiscale.cpp`.
//
// Honest scientific status: `synthetic_two_way_real_macro_history` — the
// macro CSV is real Cap. 89 evidence; the staggered loop is a numerical
// surrogate that exercises convergence/relaxation plumbing. The real
// LIBS-FULL two-way feedback is in `main_lshaped_multiscale.cpp`.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include <Eigen/Dense>

#include "src/analysis/FirstInelasticFiberCriterion.hh"
#include "src/analysis/MultiscaleTypes.hh"
#include "src/validation/MultiscaleReplayDriverHelpers.hh"
#include "src/validation/ReducedRCMultiscaleReplayPlan.hh"

namespace {

struct Options {
    std::filesystem::path input_csv{};
    std::filesystem::path output_dir{};
    std::size_t site_index{0};
    double z_over_l{0.02};
    double characteristic_length_mm{100.0};
    double EA_MN{6.0e3};
    double EI_MN_m2{30.0};
    double yield_strain{420.0 / 200000.0};
    double f_y_MPa{420.0};
    double c_section_mm{100.0};
    double damage_floor{0.05};
    // Staggered (two-way) controls.
    int    max_staggered_iter{4};
    double staggered_tol{0.05};
    double staggered_relax{0.7};
    // Convergence-rate gate over selected sites.
    double min_converged_fraction{0.95};
    std::size_t num_sites{1};
    std::vector<double> site_z_list{};
    std::vector<double> site_scale_list{};
};

[[nodiscard]] std::vector<double> parse_double_list(std::string_view s) {
    std::vector<double> out; std::string buf;
    for (char c : s) {
        if (c == ',' || c == ';') {
            if (!buf.empty()) { out.push_back(std::strtod(buf.c_str(), nullptr)); buf.clear(); }
        } else if (!std::isspace(static_cast<unsigned char>(c))) buf.push_back(c);
    }
    if (!buf.empty()) out.push_back(std::strtod(buf.c_str(), nullptr));
    return out;
}

void print_usage(const char* a0) {
    std::fprintf(stderr,
        "Usage: %s --input <csv> --output-dir <dir> [options]\n"
        "  --site-index N --z-over-l F --characteristic-length-mm F\n"
        "  --EA-MN F --EI-MN-m2 F\n"
        "  --yield-strain F --f-y-MPa F --c-section-mm F --damage-floor F\n"
        "  --max-staggered-iter N --staggered-tol F --staggered-relax F\n"
        "  --min-converged-fraction F\n"
        "  --num-sites N --site-z-list a,b,c --site-scale-list a,b,c\n",
        a0);
}

[[nodiscard]] bool parse_args(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        std::string_view a = argv[i];
        auto next = [&](double& d){ d = std::strtod(argv[++i], nullptr); };
        if      (a == "--input"                && i+1 < argc) o.input_csv = argv[++i];
        else if (a == "--output-dir"           && i+1 < argc) o.output_dir = argv[++i];
        else if (a == "--site-index"           && i+1 < argc) o.site_index = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--z-over-l"             && i+1 < argc) next(o.z_over_l);
        else if (a == "--characteristic-length-mm" && i+1 < argc) next(o.characteristic_length_mm);
        else if (a == "--EA-MN"                && i+1 < argc) next(o.EA_MN);
        else if (a == "--EI-MN-m2"             && i+1 < argc) next(o.EI_MN_m2);
        else if (a == "--yield-strain"         && i+1 < argc) next(o.yield_strain);
        else if (a == "--f-y-MPa"              && i+1 < argc) next(o.f_y_MPa);
        else if (a == "--c-section-mm"         && i+1 < argc) next(o.c_section_mm);
        else if (a == "--damage-floor"         && i+1 < argc) next(o.damage_floor);
        else if (a == "--max-staggered-iter"   && i+1 < argc) o.max_staggered_iter = std::atoi(argv[++i]);
        else if (a == "--staggered-tol"        && i+1 < argc) next(o.staggered_tol);
        else if (a == "--staggered-relax"      && i+1 < argc) next(o.staggered_relax);
        else if (a == "--min-converged-fraction" && i+1 < argc) next(o.min_converged_fraction);
        else if (a == "--num-sites"            && i+1 < argc) o.num_sites = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--site-z-list"          && i+1 < argc) o.site_z_list = parse_double_list(argv[++i]);
        else if (a == "--site-scale-list"      && i+1 < argc) o.site_scale_list = parse_double_list(argv[++i]);
        else if (a == "--help" || a == "-h") { print_usage(argv[0]); return false; }
        else { std::fprintf(stderr, "unknown arg: %s\n", argv[i]); print_usage(argv[0]); return false; }
    }
    if (o.input_csv.empty() || o.output_dir.empty()) { print_usage(argv[0]); return false; }
    return true;
}

struct StaggeredOutcome {
    int     iters{0};
    double  final_residual{0.0};
    bool    converged{false};
    Eigen::Matrix2d D_hom{Eigen::Matrix2d::Zero()};
    Eigen::Vector2d f_hom{Eigen::Vector2d::Zero()};
    Eigen::Vector2d eps_ref{Eigen::Vector2d::Zero()};
};

// Synthetic two-way staggered iteration. Each iteration relaxes between the
// macro tangent (full EA / EI) and a damage-degraded local tangent. The
// residual is the relative Frobenius norm between successive D_hom updates,
// which converges geometrically with ratio (1-relax) and reaches a
// damage-dependent floor — replicating the qualitative behaviour of the
// real bordered-mixed-control bridge while staying in Eigen-only space.
[[nodiscard]] StaggeredOutcome
run_staggered(const fall_n::ReducedRCMultiscaleReplaySitePlan& site,
              const Options& o)
{
    StaggeredOutcome out;
    const double damage = std::clamp(site.max_damage_indicator, 0.0, 1.0);
    const double s_target = std::clamp(1.0 - damage, 0.05, 1.0);
    Eigen::Matrix2d D_macro = Eigen::Matrix2d::Zero();
    D_macro(0,0) = o.EA_MN; D_macro(1,1) = o.EI_MN_m2;
    Eigen::Matrix2d D_target = Eigen::Matrix2d::Zero();
    D_target(0,0) = s_target * o.EA_MN; D_target(1,1) = s_target * o.EI_MN_m2;

    Eigen::Matrix2d D_curr = D_macro;
    const double D_macro_norm = std::max(D_macro.norm(), 1.0e-12);

    for (int it = 1; it <= o.max_staggered_iter; ++it) {
        const Eigen::Matrix2d D_prev = D_curr;
        D_curr = (1.0 - o.staggered_relax) * D_prev + o.staggered_relax * D_target;
        const double res = (D_curr - D_prev).norm() / D_macro_norm;
        out.iters = it;
        out.final_residual = res;
        if (res <= o.staggered_tol) { out.converged = true; break; }
    }

    out.D_hom = D_curr;
    out.eps_ref = Eigen::Vector2d::Zero();
    out.eps_ref(1) = site.peak_abs_curvature_y;
    out.f_hom = D_curr * out.eps_ref;
    return out;
}

void emit_json(const std::filesystem::path& path,
               const std::vector<fall_n::ReducedRCStructuralReplaySample>& hist,
               const fall_n::ReducedRCMultiscaleReplayPlan& plan,
               const std::vector<StaggeredOutcome>& staggered,
               const std::vector<fall_n::FirstInelasticFiberCriterion::Reason>& reasons,
               const std::vector<std::size_t>& trigger_indices,
               const Options& o,
               bool overall_pass,
               double converged_fraction)
{
    std::ofstream f(path);
    f << "{\n"
      << "  \"schema\": \"fe2_column_cyclic_two_way_v1\",\n"
      << "  \"scientific_status\": \"synthetic_two_way_real_macro_history\",\n"
      << "  \"history_sample_count\": " << hist.size() << ",\n"
      << "  \"selected_site_count\": " << plan.selected_site_count << ",\n"
      << "  \"staggered\": {\n"
      << "    \"max_staggered_iter\": " << o.max_staggered_iter << ",\n"
      << "    \"staggered_tol\": "      << o.staggered_tol      << ",\n"
      << "    \"staggered_relax\": "    << o.staggered_relax    << "\n"
      << "  },\n"
      << "  \"min_converged_fraction\": " << o.min_converged_fraction << ",\n"
      << "  \"converged_fraction\": " << converged_fraction << ",\n"
      << "  \"overall_pass\": " << (overall_pass ? "true" : "false") << ",\n"
      << "  \"sites\": [\n";
    std::size_t k = 0, emitted = 0;
    for (const auto& site : plan.sites) {
        if (!site.selected_for_replay) continue;
        const auto& S = staggered[k];
        const auto reason = reasons[k];
        const auto trig_idx = trigger_indices[k];
        ++k;
        f << "    {\n"
          << "      \"site_index\": " << site.site_index << ",\n"
          << "      \"z_over_l\": " << site.z_over_l << ",\n"
          << "      \"max_damage_indicator\": " << site.max_damage_indicator << ",\n"
          << "      \"peak_abs_curvature_y\": " << site.peak_abs_curvature_y << ",\n"
          << "      \"peak_abs_steel_stress_mpa\": " << site.peak_abs_steel_stress_mpa << ",\n"
          << "      \"trigger_reason\": \""
          << fall_n::FirstInelasticFiberCriterion::reason_label(reason) << "\",\n"
          << "      \"trigger_sample_index\": " << trig_idx << ",\n"
          << "      \"trigger_pseudo_time\": "
          << (trig_idx < hist.size() ? hist[trig_idx].pseudo_time : -1.0) << ",\n"
          << "      \"staggered_iters\": " << S.iters << ",\n"
          << "      \"staggered_final_residual\": " << S.final_residual << ",\n"
          << "      \"staggered_converged\": " << (S.converged ? "true" : "false") << ",\n"
          << "      \"D_hom_diag\": [" << S.D_hom(0,0) << "," << S.D_hom(1,1) << "],\n"
          << "      \"f_hom\": [" << S.f_hom(0) << "," << S.f_hom(1) << "]\n"
          << "    }";
        ++emitted;
        f << (emitted == plan.selected_site_count ? "\n" : ",\n");
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
    if (rows.empty()) { std::fprintf(stderr, "[stageB] empty CSV\n"); return 3; }

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
                ? o.site_z_list[s] : (s < default_z.size() ? default_z[s] : default_z.back());
            const double k = (!o.site_scale_list.empty() && s < o.site_scale_list.size())
                ? o.site_scale_list[s] : (s < default_s.size() ? default_s[s] : default_s.back());
            specs.push_back({.site_index = s, .z_over_l = z, .demand_scale = k});
        }
        samples = fall_n::build_multi_site_replay_samples_from_csv(
            rows, specs, o.characteristic_length_mm);
    }

    fall_n::ReducedRCMultiscaleReplayPlanSettings settings{};
    settings.max_replay_sites = std::max<std::size_t>(3, o.num_sites);
    const auto plan = fall_n::make_reduced_rc_multiscale_replay_plan(samples, settings);

    const fall_n::FirstInelasticFiberCriterion crit{
        .yield_strain = o.yield_strain, .f_y_MPa = o.f_y_MPa,
        .c_section_mm = o.c_section_mm, .damage_floor = o.damage_floor,
    };

    std::vector<StaggeredOutcome> staggered;
    std::vector<fall_n::FirstInelasticFiberCriterion::Reason> reasons;
    std::vector<std::size_t> trigger_indices;
    staggered.reserve(plan.selected_site_count);
    reasons.reserve(plan.selected_site_count);
    trigger_indices.reserve(plan.selected_site_count);

    std::size_t triggered_count = 0;
    std::size_t converged_count = 0;

    for (const auto& site : plan.sites) {
        if (!site.selected_for_replay) continue;
        std::vector<fall_n::ReducedRCStructuralReplaySample> per_site;
        per_site.reserve(samples.size());
        for (const auto& s : samples) if (s.site_index == site.site_index) per_site.push_back(s);
        const auto reason = crit.evaluate(site);
        const auto idx = crit.first_trigger_index(per_site);
        reasons.push_back(reason);
        trigger_indices.push_back(idx);
        const auto S = run_staggered(site, o);
        staggered.push_back(S);
        if (reason != fall_n::FirstInelasticFiberCriterion::Reason::not_triggered) ++triggered_count;
        if (S.converged) ++converged_count;
    }

    const double conv_frac = plan.selected_site_count == 0 ? 0.0
        : static_cast<double>(converged_count) /
          static_cast<double>(plan.selected_site_count);
    const bool overall =
        plan.selected_site_count > 0 &&
        triggered_count == plan.selected_site_count &&
        conv_frac >= o.min_converged_fraction;

    emit_json(o.output_dir / "fe2_column_two_way_cyclic.json",
              samples, plan, staggered, reasons, trigger_indices,
              o, overall, conv_frac);

    std::printf("[stageB] fe2_column_two_way emitted | selected=%zu triggered=%zu "
                "converged=%zu (%.2f) overall_pass=%d output=%s\n",
                plan.selected_site_count, triggered_count, converged_count,
                conv_frac, overall ? 1 : 0, o.output_dir.string().c_str());
    return overall ? 0 : 4;
}
