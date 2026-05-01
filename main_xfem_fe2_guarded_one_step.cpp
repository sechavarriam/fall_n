// Plan v2 §Fase 4D — Multiscale ENRICHED (XFEM) FE² guarded ONE-STEP smoke.
//
// Standalone smoke driver. Consumes any structural-history CSV, builds a
// `ReducedRCMultiscaleReplayPlan`, derives a `ReducedRCMultiscaleRuntimePolicy`,
// and for each selected replay site:
//   1. Probes `EnrichmentActivationPolicy` against the site's peak damage and
//      principal-strain proxy (curvature × characteristic length).
//   2. If the probe activates, synthesises a guarded-FE² `UpscalingResult`
//      with a damage-degraded diagonal beam-section tangent.
//   3. Asserts `passes_guarded_smoke_gate(0.03, 6)`.
//
// This is the *gate* between elastic-only smoke (§Fase 4C) and the real
// bordered-mixed-control + hybrid SNES-L2 fallback. The honest scientific
// status is `synthetic_one_step_no_real_xfem_local_solve` — the real local
// solve is the heavy XFEM benchmark, which lives in
// `main_reduced_rc_xfem_reference_benchmark`. This driver verifies the
// activation predicate + readiness gate plumbing.
//
// Emits:
//   - multiscale_enriched_fe2_guarded_smoke.json
//
// CLI:
//   main_xfem_fe2_guarded_one_step --input <csv> --output-dir <dir>
//      [--site-index N] [--z-over-l F] [--characteristic-length-mm F]
//      [--EI-MN-m2 F] [--EA-MN F]
//      [--macro-load-step N] [--damage-threshold F]
//      [--principal-strain-threshold F]
//      [--activation-macro-step N]

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

#include "src/analysis/MultiscaleTypes.hh"
#include "src/reconstruction/EnrichmentActivationPolicy.hh"
#include "src/reconstruction/LocalModelKind.hh"
#include "src/validation/MultiscaleReplayDriverHelpers.hh"
#include "src/validation/ReducedRCMultiscaleReplayPlan.hh"

namespace {

struct Options {
    std::filesystem::path input_csv{};
    std::filesystem::path output_dir{};
    std::size_t site_index{0};
    double z_over_l{0.02};
    double characteristic_length_mm{100.0};
    double EI_MN_m2{30.0};
    double EA_MN{6.0e3};
    int macro_load_step{20};
    double damage_threshold{0.20};
    double principal_strain_threshold{2.5e-3};
    int activation_macro_step{10};
    double max_frobenius_residual{0.03};
    std::size_t max_snes_iters{6};
};

void print_usage(const char* a0) {
    std::fprintf(stderr,
        "Usage: %s --input <csv> --output-dir <dir> [options]\n", a0);
}

[[nodiscard]] bool parse_args(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        std::string_view a = argv[i];
        if (a == "--input" && i + 1 < argc) o.input_csv = argv[++i];
        else if (a == "--output-dir" && i + 1 < argc) o.output_dir = argv[++i];
        else if (a == "--site-index" && i + 1 < argc) o.site_index = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--z-over-l" && i + 1 < argc) o.z_over_l = std::strtod(argv[++i], nullptr);
        else if (a == "--characteristic-length-mm" && i + 1 < argc) o.characteristic_length_mm = std::strtod(argv[++i], nullptr);
        else if (a == "--EI-MN-m2" && i + 1 < argc) o.EI_MN_m2 = std::strtod(argv[++i], nullptr);
        else if (a == "--EA-MN" && i + 1 < argc) o.EA_MN = std::strtod(argv[++i], nullptr);
        else if (a == "--macro-load-step" && i + 1 < argc) o.macro_load_step = std::atoi(argv[++i]);
        else if (a == "--damage-threshold" && i + 1 < argc) o.damage_threshold = std::strtod(argv[++i], nullptr);
        else if (a == "--principal-strain-threshold" && i + 1 < argc) o.principal_strain_threshold = std::strtod(argv[++i], nullptr);
        else if (a == "--activation-macro-step" && i + 1 < argc) o.activation_macro_step = std::atoi(argv[++i]);
        else if (a == "--max-frobenius-residual" && i + 1 < argc) o.max_frobenius_residual = std::strtod(argv[++i], nullptr);
        else if (a == "--max-snes-iters" && i + 1 < argc) o.max_snes_iters = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--help" || a == "-h") { print_usage(argv[0]); return false; }
        else { std::fprintf(stderr, "unknown arg: %s\n", argv[i]); print_usage(argv[0]); return false; }
    }
    if (o.input_csv.empty() || o.output_dir.empty()) { print_usage(argv[0]); return false; }
    return true;
}

[[nodiscard]] fall_n::EnrichmentActivationProbe
make_probe(const fall_n::ReducedRCMultiscaleReplaySitePlan& site,
           const Options& o)
{
    // Principal-strain magnitude proxy: peak |curvature_y| × char_length / 2.
    const double L_m = std::max(o.characteristic_length_mm / 1000.0, 1.0e-12);
    return fall_n::EnrichmentActivationProbe{
        .site_kind = fall_n::LocalModelKind::xfem_shifted_heaviside,
        .damage_index = site.max_damage_indicator,
        .principal_strain_magnitude = site.peak_abs_curvature_y * 0.5 * L_m,
        .macro_load_step = o.macro_load_step,
    };
}

[[nodiscard]] fall_n::UpscalingResult
synthesise_guarded_upscaling_result(
    const fall_n::ReducedRCMultiscaleReplaySitePlan& site,
    const Options& o)
{
    using namespace fall_n;
    UpscalingResult R{};
    R.eps_ref = Eigen::VectorXd::Zero(2);
    R.eps_ref(1) = site.peak_abs_curvature_y;
    // Damage-degraded diagonal section tangent. clamp(1 - d, 0.05, 1.0) keeps
    // the gate tractable even at full damage (the real solver would not
    // collapse to zero either).
    const double s = std::clamp(1.0 - site.max_damage_indicator, 0.05, 1.0);
    R.D_hom = Eigen::MatrixXd::Zero(2, 2);
    R.D_hom(0, 0) = s * o.EA_MN;
    R.D_hom(1, 1) = s * o.EI_MN_m2;
    R.f_hom = R.D_hom * R.eps_ref;
    // Synthetic guarded residual scaled with damage; stays under 0.03 unless
    // damage > 0.85, mirroring how the real bordered-mixed-control bridge
    // tightens with localisation.
    R.frobenius_residual = 0.005 + 0.025 * site.max_damage_indicator;
    R.snes_iters = 3;
    R.converged = true;
    R.status = ResponseStatus::Ok;
    R.tangent_scheme = TangentLinearizationScheme::LinearizedCondensation;
    R.condensed_status = CondensedTangentStatus::Success;
    return R;
}

void emit_smoke_json(const std::filesystem::path& path,
                     const fall_n::ReducedRCMultiscaleReplayPlan& plan,
                     const std::vector<fall_n::UpscalingResult>& results,
                     const std::vector<fall_n::EnrichmentActivationReason>& reasons,
                     const Options& o,
                     bool overall_pass)
{
    std::ofstream f(path);
    f << "{\n"
      << "  \"schema\": \"reduced_rc_multiscale_enriched_fe2_guarded_smoke_v1\",\n"
      << "  \"scientific_status\": \"synthetic_one_step_no_real_xfem_local_solve\",\n"
      << "  \"selected_site_count\": " << plan.selected_site_count << ",\n"
      << "  \"max_frobenius_residual\": " << o.max_frobenius_residual << ",\n"
      << "  \"max_snes_iters\": " << o.max_snes_iters << ",\n"
      << "  \"macro_load_step\": " << o.macro_load_step << ",\n"
      << "  \"thresholds\": {\n"
      << "    \"damage_threshold\": " << o.damage_threshold << ",\n"
      << "    \"principal_strain_threshold\": " << o.principal_strain_threshold << ",\n"
      << "    \"activation_macro_step\": " << o.activation_macro_step << "\n"
      << "  },\n"
      << "  \"overall_pass\": " << (overall_pass ? "true" : "false") << ",\n"
      << "  \"sites\": [\n";
    std::size_t k = 0;
    std::size_t emitted = 0;
    for (const auto& site : plan.sites) {
        if (!site.selected_for_replay) continue;
        const auto& R = results[k];
        const auto reason = reasons[k];
        ++k;
        const bool activated =
            reason == fall_n::EnrichmentActivationReason::activated;
        const bool gate_pass = R.passes_guarded_smoke_gate(
            o.max_frobenius_residual, o.max_snes_iters);
        f << "    {\n"
          << "      \"site_index\": " << site.site_index << ",\n"
          << "      \"z_over_l\": " << site.z_over_l << ",\n"
          << "      \"activation_kind\": \""
          << fall_n::activation_kind_label(site.activation_kind) << "\",\n"
          << "      \"max_damage_indicator\": " << site.max_damage_indicator << ",\n"
          << "      \"peak_abs_curvature_y\": " << site.peak_abs_curvature_y << ",\n"
          << "      \"enrichment_reason\": \""
          << fall_n::enrichment_activation_reason_label(reason) << "\",\n"
          << "      \"enrichment_activated\": " << (activated ? "true" : "false") << ",\n"
          << "      \"frobenius_residual\": " << R.frobenius_residual << ",\n"
          << "      \"snes_iters\": " << R.snes_iters << ",\n"
          << "      \"converged\": " << (R.converged ? "true" : "false") << ",\n"
          << "      \"D_hom_diag\": ["
          << R.D_hom(0, 0) << "," << R.D_hom(1, 1) << "],\n"
          << "      \"passes_guarded_smoke_gate\": "
          << (gate_pass ? "true" : "false") << "\n"
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
    if (rows.empty()) { std::fprintf(stderr, "[fase4d] empty CSV\n"); return 3; }
    const auto samples = fall_n::build_replay_samples_from_csv(
        rows, o.site_index, o.z_over_l, o.characteristic_length_mm);
    const auto plan = fall_n::make_reduced_rc_multiscale_replay_plan(samples, {});

    fall_n::EnrichmentActivationThresholds th{
        .damage_threshold = o.damage_threshold,
        .principal_strain_threshold = o.principal_strain_threshold,
        .activation_macro_step = o.activation_macro_step,
    };

    std::vector<fall_n::UpscalingResult> results;
    std::vector<fall_n::EnrichmentActivationReason> reasons;
    results.reserve(plan.selected_site_count);
    reasons.reserve(plan.selected_site_count);
    bool overall = plan.selected_site_count > 0;
    for (const auto& site : plan.sites) {
        if (!site.selected_for_replay) continue;
        const auto probe = make_probe(site, o);
        const auto reason = fall_n::classify_enrichment_activation(probe, th);
        reasons.push_back(reason);
        const auto R = synthesise_guarded_upscaling_result(site, o);
        results.push_back(R);
        const bool gate_pass = R.passes_guarded_smoke_gate(
            o.max_frobenius_residual, o.max_snes_iters);
        const bool activated =
            reason == fall_n::EnrichmentActivationReason::activated;
        // Smoke is overall_pass iff every selected site (a) activates the
        // enrichment policy at the supplied macro step, and (b) the synthetic
        // guarded UpscalingResult passes the Cap. 79 gate.
        overall = overall && activated && gate_pass;
    }

    emit_smoke_json(
        o.output_dir / "multiscale_enriched_fe2_guarded_smoke.json",
        plan, results, reasons, o, overall);

    std::printf("[fase4d] enriched_fe2_guarded_smoke emitted | selected=%zu "
                "overall_pass=%d output=%s\n",
                plan.selected_site_count, overall ? 1 : 0,
                o.output_dir.string().c_str());
    return overall ? 0 : 4;
}
