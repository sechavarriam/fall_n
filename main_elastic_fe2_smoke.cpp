// Plan v2 §Fase 4C — Multiscale ELASTIC FE² guarded smoke driver.
//
// Standalone smoke driver. Consumes any structural-history CSV, builds a
// `ReducedRCMultiscaleReplayPlan`, derives a `ReducedRCMultiscaleRuntimePolicy`,
// and emits a synthetic ELASTIC `UpscalingResult` for each selected replay
// site. No PETSc / SNES / mesh — this is the *gating* smoke that proves the
// downstream consumer (MaterialSection::set_homogenized_response shim and
// `passes_guarded_smoke_gate`) is wired correctly before we plug in the real
// bordered-mixed-control bridge.
//
// The synthetic `D_hom` is a diagonal beam-section tangent assembled from EI/EA
// inputs (kept honest: the result is tagged
// `tangent_scheme = LinearizedCondensation` and is ELASTIC by construction).
//
// Emits:
//   - multiscale_elastic_fe2_smoke.json  (one entry per selected site)
//
// Exit code: 0 if every selected site passes_guarded_smoke_gate(0.03, 6),
//            non-zero otherwise.
//
// CLI:
//   main_elastic_fe2_smoke --input <csv> --output-dir <dir>
//                          [--site-index N] [--z-over-l F]
//                          [--characteristic-length-mm F]
//                          [--EI-MN-m2 F] [--EA-MN F]
//                          [--max-frobenius-residual F]
//                          [--max-snes-iters N]

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

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
    double EI_MN_m2{30.0};   // canonical reduced-RC column, EI ≈ 30 MN·m²
    double EA_MN{6.0e3};     // ≈ 6 GN axial stiffness for the same column
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
        else if (a == "--max-frobenius-residual" && i + 1 < argc) o.max_frobenius_residual = std::strtod(argv[++i], nullptr);
        else if (a == "--max-snes-iters" && i + 1 < argc) o.max_snes_iters = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--help" || a == "-h") { print_usage(argv[0]); return false; }
        else { std::fprintf(stderr, "unknown arg: %s\n", argv[i]); print_usage(argv[0]); return false; }
    }
    if (o.input_csv.empty() || o.output_dir.empty()) { print_usage(argv[0]); return false; }
    return true;
}

[[nodiscard]] fall_n::UpscalingResult
synthesise_elastic_upscaling_result(
    const fall_n::ReducedRCMultiscaleReplaySitePlan& site,
    const Options& o)
{
    using namespace fall_n;
    UpscalingResult R{};
    // Beam-section generalised strain order: [eps_axial, kappa_y].
    // Reference strain at the linearisation point: site peak curvature only
    // (axial small for cyclic flexural protocol).
    R.eps_ref = Eigen::VectorXd::Zero(2);
    R.eps_ref(1) = site.peak_abs_curvature_y;
    // Elastic homogenised force at eps_ref using diagonal D_hom:
    //   f_hom = D_hom * eps_ref.
    R.D_hom = Eigen::MatrixXd::Zero(2, 2);
    R.D_hom(0, 0) = o.EA_MN;
    R.D_hom(1, 1) = o.EI_MN_m2;
    R.f_hom = R.D_hom * R.eps_ref;
    // Synthetic residual: 0 (elastic, no FE² coupling consumed).
    R.frobenius_residual = 0.0;
    R.snes_iters = 1;
    R.converged = true;
    R.status = ResponseStatus::Ok;
    R.tangent_scheme = TangentLinearizationScheme::LinearizedCondensation;
    R.condensed_status = CondensedTangentStatus::Success;
    return R;
}

void emit_smoke_json(const std::filesystem::path& path,
                     const fall_n::ReducedRCMultiscaleReplayPlan& plan,
                     const std::vector<fall_n::UpscalingResult>& results,
                     const Options& o,
                     bool overall_pass)
{
    std::ofstream f(path);
    f << "{\n"
      << "  \"schema\": \"reduced_rc_multiscale_elastic_fe2_smoke_v1\",\n"
      << "  \"scientific_status\": \"synthetic_elastic_smoke_no_fe2_coupling\",\n"
      << "  \"selected_site_count\": " << plan.selected_site_count << ",\n"
      << "  \"max_frobenius_residual\": " << o.max_frobenius_residual << ",\n"
      << "  \"max_snes_iters\": " << o.max_snes_iters << ",\n"
      << "  \"EI_MN_m2\": " << o.EI_MN_m2 << ",\n"
      << "  \"EA_MN\": " << o.EA_MN << ",\n"
      << "  \"overall_pass\": " << (overall_pass ? "true" : "false") << ",\n"
      << "  \"sites\": [\n";
    std::size_t k = 0;
    std::size_t emitted = 0;
    for (const auto& site : plan.sites) {
        if (!site.selected_for_replay) continue;
        const auto& R = results[k++];
        const bool pass = R.passes_guarded_smoke_gate(
            o.max_frobenius_residual, o.max_snes_iters);
        f << "    {\n"
          << "      \"site_index\": " << site.site_index << ",\n"
          << "      \"z_over_l\": " << site.z_over_l << ",\n"
          << "      \"activation_kind\": \""
          << fall_n::activation_kind_label(site.activation_kind) << "\",\n"
          << "      \"eps_ref\": [" << R.eps_ref(0) << "," << R.eps_ref(1) << "],\n"
          << "      \"f_hom\": ["  << R.f_hom(0)  << "," << R.f_hom(1)  << "],\n"
          << "      \"D_hom_diag\": ["
          << R.D_hom(0, 0) << "," << R.D_hom(1, 1) << "],\n"
          << "      \"frobenius_residual\": " << R.frobenius_residual << ",\n"
          << "      \"snes_iters\": " << R.snes_iters << ",\n"
          << "      \"converged\": " << (R.converged ? "true" : "false") << ",\n"
          << "      \"status\": \"Ok\",\n"
          << "      \"tangent_scheme\": \"LinearizedCondensation\",\n"
          << "      \"passes_guarded_smoke_gate\": "
          << (pass ? "true" : "false") << "\n"
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
    if (rows.empty()) { std::fprintf(stderr, "[fase4c] empty CSV\n"); return 3; }
    const auto samples = fall_n::build_replay_samples_from_csv(
        rows, o.site_index, o.z_over_l, o.characteristic_length_mm);
    const auto plan = fall_n::make_reduced_rc_multiscale_replay_plan(samples, {});

    std::vector<fall_n::UpscalingResult> results;
    results.reserve(plan.selected_site_count);
    bool overall = plan.selected_site_count > 0;
    for (const auto& site : plan.sites) {
        if (!site.selected_for_replay) continue;
        const auto R = synthesise_elastic_upscaling_result(site, o);
        const bool pass = R.passes_guarded_smoke_gate(
            o.max_frobenius_residual, o.max_snes_iters);
        overall = overall && pass;
        results.push_back(R);
    }

    emit_smoke_json(o.output_dir / "multiscale_elastic_fe2_smoke.json",
                    plan, results, o, overall);

    std::printf("[fase4c] elastic_fe2_smoke emitted | selected=%zu overall_pass=%d "
                "output=%s\n",
                plan.selected_site_count, overall ? 1 : 0,
                o.output_dir.string().c_str());
    return overall ? 0 : 4;
}
