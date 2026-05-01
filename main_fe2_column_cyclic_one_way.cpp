// Plan v2 §Stage A — FE² ONE-WAY cyclic 200 mm column smoke driver.
//
// Standalone driver that consumes a structural-history CSV (the canonical
// 200 mm cyclic envelope produced by Cap. 89 closure runs, e.g.
// `data/output/cyclic_validation/xfem_small_strain_200mm_v2/...`),
// builds a `ReducedRCMultiscaleReplayPlan`, and:
//
//   1. Evaluates the injected `FirstInelasticFiberCriterion` on the history
//      sample-by-sample. Records the first-trigger index and pseudo-time.
//   2. After activation, synthesises a one-way homogenised tangent and
//      force vector at each replay site (damage-degraded diagonal, same
//      construction as the Fase 4D driver). The macro side of the chain
//      keeps its baseline tangent (one-way: NO feedback into macro).
//   3. Cross-checks the synthetic homogenised moment vs the recorded
//      `peak_abs_moment_y_mn_m` of the site and computes a relative
//      envelope error.
//   4. Emits `fe2_column_one_way_cyclic.json`.
//
// Honest scientific status: `synthetic_one_way_real_macro_history` —
// the macro CSV is real Cap. 89 evidence; the local upscaling is a
// damage-degraded synthetic surrogate. Intended as the smoke gate before
// the two-way Stage B driver and the LIBS-FULL real solver path.

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
    // Section EA / EI defaults that match Cap. 89 200 × 200 mm column
    // (fc=21 MPa, longitudinal ratio ~ 1%).
    double EA_MN{6.0e3};
    double EI_MN_m2{30.0};
    // Activation criterion knobs.
    double yield_strain{420.0 / 200000.0};
    double f_y_MPa{420.0};
    double c_section_mm{100.0};
    double damage_floor{0.05};
    // Cross-model envelope gate.
    double max_relative_moment_envelope_error{0.25};
    // Multi-site optional ladder.
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
        "  --max-relative-moment-envelope-error F\n"
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
        else if (a == "--max-relative-moment-envelope-error" && i+1 < argc)
            next(o.max_relative_moment_envelope_error);
        else if (a == "--num-sites"            && i+1 < argc) o.num_sites = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--site-z-list"          && i+1 < argc) o.site_z_list = parse_double_list(argv[++i]);
        else if (a == "--site-scale-list"      && i+1 < argc) o.site_scale_list = parse_double_list(argv[++i]);
        else if (a == "--help" || a == "-h") { print_usage(argv[0]); return false; }
        else { std::fprintf(stderr, "unknown arg: %s\n", argv[i]); print_usage(argv[0]); return false; }
    }
    if (o.input_csv.empty() || o.output_dir.empty()) { print_usage(argv[0]); return false; }
    return true;
}

[[nodiscard]] fall_n::FirstInelasticFiberCriterion
make_criterion(const Options& o) {
    return fall_n::FirstInelasticFiberCriterion{
        .yield_strain = o.yield_strain,
        .f_y_MPa      = o.f_y_MPa,
        .c_section_mm = o.c_section_mm,
        .damage_floor = o.damage_floor,
    };
}

// One-way synthetic homogenised response (damage-degraded diagonal section
// tangent). Identical scaling to the Fase 4D smoke driver to keep the gate
// comparable across phases.
[[nodiscard]] fall_n::UpscalingResult
synthesise_one_way_response(
    const fall_n::ReducedRCMultiscaleReplaySitePlan& site,
    const Options& o)
{
    using namespace fall_n;
    UpscalingResult R{};
    R.eps_ref = Eigen::VectorXd::Zero(2);
    R.eps_ref(1) = site.peak_abs_curvature_y;
    const double s = std::clamp(1.0 - site.max_damage_indicator, 0.05, 1.0);
    R.D_hom = Eigen::MatrixXd::Zero(2, 2);
    R.D_hom(0, 0) = s * o.EA_MN;
    R.D_hom(1, 1) = s * o.EI_MN_m2;
    R.f_hom = R.D_hom * R.eps_ref;
    R.frobenius_residual = 0.005 + 0.020 * site.max_damage_indicator;
    R.snes_iters = 3;
    R.converged  = true;
    R.status     = ResponseStatus::Ok;
    R.tangent_scheme   = TangentLinearizationScheme::LinearizedCondensation;
    R.condensed_status = CondensedTangentStatus::Success;
    return R;
}

void emit_json(const std::filesystem::path& path,
               const std::vector<fall_n::ReducedRCStructuralReplaySample>& hist,
               const fall_n::ReducedRCMultiscaleReplayPlan& plan,
               const std::vector<fall_n::UpscalingResult>& results,
               const std::vector<fall_n::FirstInelasticFiberCriterion::Reason>& reasons,
               const std::vector<std::size_t>& trigger_indices,
               const Options& o,
               bool overall_pass)
{
    std::ofstream f(path);
    f << "{\n"
      << "  \"schema\": \"fe2_column_cyclic_one_way_v1\",\n"
      << "  \"scientific_status\": \"synthetic_one_way_real_macro_history\",\n"
      << "  \"history_sample_count\": " << hist.size() << ",\n"
      << "  \"selected_site_count\": " << plan.selected_site_count << ",\n"
      << "  \"criterion\": {\n"
      << "    \"yield_strain\": " << o.yield_strain << ",\n"
      << "    \"f_y_MPa\": "      << o.f_y_MPa      << ",\n"
      << "    \"c_section_mm\": " << o.c_section_mm << ",\n"
      << "    \"damage_floor\": " << o.damage_floor << "\n"
      << "  },\n"
      << "  \"max_relative_moment_envelope_error\": "
      << o.max_relative_moment_envelope_error << ",\n"
      << "  \"overall_pass\": " << (overall_pass ? "true" : "false") << ",\n"
      << "  \"sites\": [\n";
    std::size_t k = 0, emitted = 0;
    for (const auto& site : plan.sites) {
        if (!site.selected_for_replay) continue;
        const auto& R = results[k];
        const auto reason = reasons[k];
        const auto trig_idx = trigger_indices[k];
        ++k;
        const double m_synth = std::abs(R.f_hom(1));
        const double m_macro = site.peak_abs_moment_y_mn_m;
        const bool moment_available = m_macro > 1.0e-6;
        const double rel_err = moment_available
            ? std::abs(m_synth - m_macro) / m_macro
            : 0.0;
        const bool triggered_ok =
            reason != fall_n::FirstInelasticFiberCriterion::Reason::not_triggered;
        const bool envelope_ok = !moment_available
            || rel_err <= o.max_relative_moment_envelope_error;
        const bool gate = triggered_ok && envelope_ok;
        f << "    {\n"
          << "      \"site_index\": " << site.site_index << ",\n"
          << "      \"z_over_l\": " << site.z_over_l << ",\n"
          << "      \"max_damage_indicator\": " << site.max_damage_indicator << ",\n"
          << "      \"peak_abs_curvature_y\": " << site.peak_abs_curvature_y << ",\n"
          << "      \"peak_abs_moment_y_mn_m\": " << site.peak_abs_moment_y_mn_m << ",\n"
          << "      \"peak_abs_steel_stress_mpa\": " << site.peak_abs_steel_stress_mpa << ",\n"
          << "      \"trigger_reason\": \""
          << fall_n::FirstInelasticFiberCriterion::reason_label(reason) << "\",\n"
          << "      \"trigger_sample_index\": " << trig_idx << ",\n"
          << "      \"trigger_pseudo_time\": "
          << (trig_idx < hist.size() ? hist[trig_idx].pseudo_time : -1.0) << ",\n"
          << "      \"trigger_drift_mm\": "
          << (trig_idx < hist.size() ? hist[trig_idx].drift_mm : 0.0) << ",\n"
          << "      \"D_hom_diag\": [" << R.D_hom(0,0) << "," << R.D_hom(1,1) << "],\n"
          << "      \"synthetic_moment_y_mn_m\": " << m_synth << ",\n"
          << "      \"moment_envelope_available\": "
          << (moment_available ? "true" : "false") << ",\n"
          << "      \"relative_moment_envelope_error\": " << rel_err << ",\n"
          << "      \"frobenius_residual\": " << R.frobenius_residual << ",\n"
          << "      \"snes_iters\": " << R.snes_iters << ",\n"
          << "      \"converged\": " << (R.converged ? "true" : "false") << ",\n"
          << "      \"site_pass\": " << (gate ? "true" : "false") << "\n"
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
    if (rows.empty()) { std::fprintf(stderr, "[stageA] empty CSV\n"); return 3; }

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

    const auto crit = make_criterion(o);

    std::vector<fall_n::UpscalingResult> results;
    std::vector<fall_n::FirstInelasticFiberCriterion::Reason> reasons;
    std::vector<std::size_t> trigger_indices;
    results.reserve(plan.selected_site_count);
    reasons.reserve(plan.selected_site_count);
    trigger_indices.reserve(plan.selected_site_count);

    bool overall = plan.selected_site_count > 0;

    for (const auto& site : plan.sites) {
        if (!site.selected_for_replay) continue;
        // Per-site history slice (samples carry site_index).
        std::vector<fall_n::ReducedRCStructuralReplaySample> per_site;
        per_site.reserve(samples.size());
        for (const auto& s : samples) if (s.site_index == site.site_index) per_site.push_back(s);
        const auto reason = crit.evaluate(site);
        const auto idx = crit.first_trigger_index(per_site);
        reasons.push_back(reason);
        trigger_indices.push_back(idx);
        const auto R = synthesise_one_way_response(site, o);
        results.push_back(R);

        const double m_synth = std::abs(R.f_hom(1));
        const double m_macro = site.peak_abs_moment_y_mn_m;
        const bool moment_available = m_macro > 1.0e-6;
        const double rel_err = moment_available
            ? std::abs(m_synth - m_macro) / m_macro
            : 0.0;
        const bool triggered =
            reason != fall_n::FirstInelasticFiberCriterion::Reason::not_triggered;
        const bool envelope_ok = !moment_available
            || rel_err <= o.max_relative_moment_envelope_error;
        const bool gate = triggered && envelope_ok;
        overall = overall && gate;
    }

    emit_json(o.output_dir / "fe2_column_one_way_cyclic.json",
              samples, plan, results, reasons, trigger_indices, o, overall);

    std::printf("[stageA] fe2_column_one_way emitted | selected=%zu overall_pass=%d output=%s\n",
                plan.selected_site_count, overall ? 1 : 0,
                o.output_dir.string().c_str());
    return overall ? 0 : 4;
}
