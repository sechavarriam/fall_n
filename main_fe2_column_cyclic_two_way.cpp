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
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <Eigen/Dense>
#include <petsc.h>

#include "src/analysis/FirstInelasticFiberCriterion.hh"
#include "src/analysis/MultiscaleTypes.hh"
#include "src/validation/MultiscaleReplayDriverHelpers.hh"
#include "src/validation/ReducedRCFE2ColumnValidation.hh"
#include "src/validation/ReducedRCManagedLocalModelReplay.hh"
#include "src/validation/ReducedRCManagedXfemLocalModelAdapter.hh"
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
    double local_section_width_m{0.20};
    double local_section_depth_m{0.20};
    std::size_t local_nx{3};
    std::size_t local_ny{3};
    std::size_t local_nz{6};
    int local_transition_steps{3};
    int local_max_bisections{6};
    std::string local_downscaling{"macro-resultant-compliance"};
    bool surrogate_smoke{false};
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
        "  --num-sites N --site-z-list a,b,c --site-scale-list a,b,c\n"
        "  --local-section-width-m F --local-section-depth-m F\n"
        "  --local-nx N --local-ny N --local-nz N\n"
        "  --local-transition-steps N --local-max-bisections N\n"
        "  --local-downscaling tip-drift|section-kinematics|macro-shear-compliance|macro-resultant-compliance\n"
        "  --surrogate-smoke  (explicitly keep the cheap synthetic gate)\n"
        "  --real-xfem-replay (default; managed XFEM local replay target)\n",
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
        else if (a == "--local-section-width-m" && i+1 < argc) next(o.local_section_width_m);
        else if (a == "--local-section-depth-m" && i+1 < argc) next(o.local_section_depth_m);
        else if (a == "--local-nx"             && i+1 < argc) o.local_nx = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--local-ny"             && i+1 < argc) o.local_ny = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--local-nz"             && i+1 < argc) o.local_nz = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--local-transition-steps" && i+1 < argc) o.local_transition_steps = std::atoi(argv[++i]);
        else if (a == "--local-max-bisections" && i+1 < argc) o.local_max_bisections = std::atoi(argv[++i]);
        else if (a == "--local-downscaling"    && i+1 < argc) o.local_downscaling = argv[++i];
        else if (a == "--surrogate-smoke") o.surrogate_smoke = true;
        else if (a == "--real-xfem-replay") o.surrogate_smoke = false;
        else if (a == "--help" || a == "-h") { print_usage(argv[0]); return false; }
        else { std::fprintf(stderr, "unknown arg: %s\n", argv[i]); print_usage(argv[0]); return false; }
    }
    if (o.input_csv.empty() || o.output_dir.empty()) { print_usage(argv[0]); return false; }
    return true;
}

[[nodiscard]] fall_n::ReducedRCFE2ColumnRunSpec
make_run_spec(const Options& o)
{
    fall_n::ReducedRCFE2ColumnRunSpec spec{};
    spec.coupling_mode =
        fall_n::ReducedRCFE2ColumnCouplingMode::iterated_two_way_fe2;
    spec.local_execution_mode = o.surrogate_smoke
        ? fall_n::ReducedRCFE2ColumnLocalExecutionMode::surrogate_smoke
        : fall_n::ReducedRCFE2ColumnLocalExecutionMode::real_xfem_replay;
    spec.EA_MN = o.EA_MN;
    spec.EI_MN_m2 = o.EI_MN_m2;
    spec.damage_floor = o.damage_floor;
    spec.f_y_MPa = o.f_y_MPa;
    spec.yield_strain = o.yield_strain;
    spec.c_section_mm = o.c_section_mm;
    spec.max_staggered_iterations = o.max_staggered_iter;
    spec.staggered_tolerance = o.staggered_tol;
    spec.staggered_relaxation = o.staggered_relax;
    spec.tolerances.min_converged_fraction = o.min_converged_fraction;
    return spec;
}

[[nodiscard]] fall_n::ReducedRCManagedXfemLocalModelAdapterOptions::DownscalingMode
parse_local_downscaling(std::string value)
{
    std::ranges::replace(value, '-', '_');
    using Mode = fall_n::ReducedRCManagedXfemLocalModelAdapterOptions::DownscalingMode;
    if (value == "tip_drift" || value == "tip_drift_top_face") {
        return Mode::tip_drift_top_face;
    }
    if (value == "section_kinematics" || value == "section_kinematics_only") {
        return Mode::section_kinematics_only;
    }
    if (value == "macro_shear_compliance" || value == "shear_compliance") {
        return Mode::macro_shear_compliance;
    }
    if (value == "macro_resultant_compliance" ||
        value == "stress_resultant" ||
        value == "dual_resultant") {
        return Mode::macro_resultant_compliance;
    }
    throw std::invalid_argument("Unsupported --local-downscaling value.");
}

[[maybe_unused]] void emit_blocked_json(
    const std::filesystem::path& path,
    const fall_n::ReducedRCFE2ColumnResult& result)
{
    std::ofstream f(path);
    f << "{\n"
      << "  \"schema\": \"fe2_column_two_way_cyclic_v2\",\n"
      << "  \"scientific_status\": \"iterated_two_way_real_xfem_blocked_adapter_missing\",\n"
      << "  \"coupling_mode\": \""
      << fall_n::to_string(result.spec.coupling_mode) << "\",\n"
      << "  \"local_execution_mode\": \""
      << fall_n::to_string(result.spec.local_execution_mode) << "\",\n"
      << "  \"local_model_policy\": \""
      << result.spec.local_model_policy << "\",\n"
      << "  \"validation_status\": \""
      << fall_n::to_string(result.status) << "\",\n"
      << "  \"history_sample_count\": " << result.history_sample_count << ",\n"
      << "  \"selected_site_count\": " << result.selected_site_count << ",\n"
      << "  \"required_adapter\": \"reduced_rc_managed_xfem_local_model_adapter + BeamMacroBridge SectionHomogenizedResponse injection\",\n"
      << "  \"next_step\": \"instantiate MultiscaleAnalysis with one managed XFEM Model per selected macro site and replace the surrogate staggered loop\"\n"
      << "}\n";
}

struct StaggeredOutcome {
    int     iters{0};
    double  final_residual{0.0};
    bool    converged{false};
    Eigen::Matrix2d D_hom{Eigen::Matrix2d::Zero()};
    Eigen::Vector2d f_hom{Eigen::Vector2d::Zero()};
    Eigen::Vector2d eps_ref{Eigen::Vector2d::Zero()};
    std::vector<double> residual_history{};
    fall_n::ReducedRCManagedLocalReplayStatus local_status{
        fall_n::ReducedRCManagedLocalReplayStatus::not_run};
    std::size_t local_accepted_steps{0};
    int local_total_nonlinear_iterations{0};
    double local_elapsed_seconds{0.0};
};

// Synthetic two-way staggered iteration. Each iteration relaxes between the
// macro tangent (full EA / EI) and a damage-degraded local tangent. The
// residual is the relative Frobenius norm between successive D_hom updates,
// which converges geometrically with ratio (1-relax) and reaches a
// damage-dependent floor — replicating the qualitative behaviour of the
// real bordered-mixed-control bridge while staying in Eigen-only space.
[[nodiscard]] StaggeredOutcome
run_staggered_toward_target(
    const Options& o,
    const Eigen::Matrix2d& D_target,
    const Eigen::Vector2d& f_target,
    const Eigen::Vector2d& eps_target)
{
    StaggeredOutcome out;
    Eigen::Matrix2d D_macro = Eigen::Matrix2d::Zero();
    D_macro(0,0) = o.EA_MN; D_macro(1,1) = o.EI_MN_m2;

    Eigen::Matrix2d D_curr = D_macro;
    const Eigen::Vector2d f_macro = D_macro * eps_target;
    Eigen::Vector2d f_curr = f_macro;
    const double D_macro_norm = std::max(D_macro.norm(), 1.0e-12);
    const double f_norm =
        std::max(std::max(f_macro.norm(), f_target.norm()), 1.0e-12);
    const double omega = std::clamp(o.staggered_relax, 0.0, 1.0);

    for (int it = 1; it <= o.max_staggered_iter; ++it) {
        D_curr = (1.0 - omega) * D_curr + omega * D_target;
        f_curr = (1.0 - omega) * f_curr + omega * f_target;
        const double D_res = (D_curr - D_target).norm() / D_macro_norm;
        const double f_res = (f_curr - f_target).norm() / f_norm;
        const double res = std::max(D_res, f_res);
        out.iters = it;
        out.final_residual = res;
        out.residual_history.push_back(res);
        if (res <= o.staggered_tol) { out.converged = true; break; }
    }

    out.D_hom = D_curr;
    out.eps_ref = eps_target;
    out.f_hom = f_curr;
    return out;
}

[[nodiscard]] StaggeredOutcome
run_surrogate_staggered(const fall_n::ReducedRCMultiscaleReplaySitePlan& site,
                        const Options& o)
{
    const double damage = std::clamp(site.max_damage_indicator, 0.0, 1.0);
    const double scale = std::clamp(1.0 - damage, 0.05, 1.0);
    Eigen::Matrix2d D_target = Eigen::Matrix2d::Zero();
    D_target(0,0) = scale * o.EA_MN;
    D_target(1,1) = scale * o.EI_MN_m2;
    Eigen::Vector2d eps_target = Eigen::Vector2d::Zero();
    eps_target(1) = site.peak_abs_curvature_y;
    const Eigen::Vector2d f_target = D_target * eps_target;
    return run_staggered_toward_target(o, D_target, f_target, eps_target);
}

[[nodiscard]] StaggeredOutcome
run_real_xfem_staggered(
    const fall_n::ReducedRCMultiscaleReplaySitePlan& site,
    const std::vector<fall_n::ReducedRCStructuralReplaySample>& activated_history,
    const Options& o)
{
    fall_n::ReducedRCManagedLocalPatchSpec patch{};
    patch.site_index = site.site_index;
    patch.z_over_l = site.z_over_l;
    patch.characteristic_length_m =
        std::max(o.characteristic_length_mm / 1000.0, 1.0e-6);
    patch.section_width_m = o.local_section_width_m;
    patch.section_depth_m = o.local_section_depth_m;
    patch.nx = std::max<std::size_t>(1, o.local_nx);
    patch.ny = std::max<std::size_t>(1, o.local_ny);
    patch.nz = std::max<std::size_t>(1, o.local_nz);

    fall_n::ReducedRCManagedXfemLocalModelAdapterOptions adapter_options{};
    adapter_options.downscaling_mode = parse_local_downscaling(o.local_downscaling);
    adapter_options.local_transition_steps = std::max(1, o.local_transition_steps);
    adapter_options.local_max_bisections = std::max(0, o.local_max_bisections);

    fall_n::ReducedRCManagedXfemLocalModelAdapter adapter{adapter_options};
    const auto replay = fall_n::run_reduced_rc_managed_local_model_replay(
        activated_history, patch, adapter);

    StaggeredOutcome out{};
    out.local_status = replay.status;
    out.local_accepted_steps = replay.accepted_step_count;
    out.local_total_nonlinear_iterations = replay.total_nonlinear_iterations;
    out.local_elapsed_seconds = replay.total_elapsed_seconds;

    const auto& R = replay.homogenized_response;
    if (!replay.completed() || !R.is_well_formed() ||
        R.D_hom.rows() < 2 || R.D_hom.cols() < 2 ||
        R.f_hom.size() < 2 || R.eps_ref.size() < 2) {
        out.final_residual = std::numeric_limits<double>::infinity();
        return out;
    }

    Eigen::Matrix2d D_target = Eigen::Matrix2d::Zero();
    D_target(0,0) = R.D_hom(0,0);
    D_target(0,1) = R.D_hom(0,1);
    D_target(1,0) = R.D_hom(1,0);
    D_target(1,1) = R.D_hom(1,1);
    Eigen::Vector2d f_target{R.f_hom(0), R.f_hom(1)};
    Eigen::Vector2d eps_target{R.eps_ref(0), R.eps_ref(1)};

    out = run_staggered_toward_target(o, D_target, f_target, eps_target);
    out.local_status = replay.status;
    out.local_accepted_steps = replay.accepted_step_count;
    out.local_total_nonlinear_iterations = replay.total_nonlinear_iterations;
    out.local_elapsed_seconds = replay.total_elapsed_seconds;
    return out;
}

[[nodiscard]] const fall_n::ReducedRCStructuralReplaySample*
trigger_sample_for_site(
    const std::vector<fall_n::ReducedRCStructuralReplaySample>& hist,
    std::size_t site_index,
    std::size_t per_site_index)
{
    std::size_t seen = 0;
    for (const auto& sample : hist) {
        if (sample.site_index != site_index) {
            continue;
        }
        if (seen == per_site_index) {
            return &sample;
        }
        ++seen;
    }
    return nullptr;
}

[[nodiscard]] double macro_section_history_work_for_site(
    const std::vector<fall_n::ReducedRCStructuralReplaySample>& hist,
    const fall_n::ReducedRCMultiscaleReplaySitePlan& site)
{
    std::vector<fall_n::ReducedRCStructuralReplaySample> per_site;
    per_site.reserve(hist.size());
    for (const auto& sample : hist) {
        if (sample.site_index == site.site_index) {
            per_site.push_back(sample);
        }
    }

    fall_n::ReducedRCManagedLocalPatchSpec patch{};
    patch.site_index = site.site_index;
    patch.z_over_l = site.z_over_l;
    const auto packet = fall_n::make_reduced_rc_managed_section_history_packet(
        per_site, patch);
    return fall_n::accumulated_material_history_work(packet.samples);
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
      << "  \"schema\": \"fe2_column_two_way_cyclic_v2\",\n"
      << "  \"scientific_status\": \""
      << (o.surrogate_smoke
              ? "surrogate_two_way_real_macro_history"
              : "managed_xfem_two_way_first_generation") << "\",\n"
      << "  \"coupling_mode\": \"iterated_two_way_fe2\",\n"
      << "  \"local_execution_mode\": \""
      << (o.surrogate_smoke ? "surrogate_smoke" : "real_xfem_replay") << "\",\n"
      << "  \"local_model_policy\": \"managed_independent_domain_per_selected_macro_site\",\n"
      << "  \"local_downscaling\": \"" << o.local_downscaling << "\",\n"
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
        const auto* trigger_sample =
            trigger_sample_for_site(hist, site.site_index, trig_idx);
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
          << (trigger_sample ? trigger_sample->pseudo_time : -1.0) << ",\n"
          << "      \"trigger_drift_mm\": "
          << (trigger_sample ? trigger_sample->drift_mm : 0.0) << ",\n"
          << "      \"staggered_iters\": " << S.iters << ",\n"
          << "      \"staggered_final_residual\": " << S.final_residual << ",\n"
          << "      \"staggered_converged\": " << (S.converged ? "true" : "false") << ",\n"
          << "      \"relative_moment_feedback_error\": "
          << (site.peak_abs_moment_y_mn_m > 1.0e-12
                  ? std::abs(std::abs(S.f_hom(1)) -
                             site.peak_abs_moment_y_mn_m) /
                        site.peak_abs_moment_y_mn_m
                  : 0.0) << ",\n"
          << "      \"local_status\": \"" << fall_n::to_string(S.local_status) << "\",\n"
          << "      \"local_accepted_steps\": " << S.local_accepted_steps << ",\n"
          << "      \"local_total_nonlinear_iterations\": " << S.local_total_nonlinear_iterations << ",\n"
          << "      \"local_elapsed_seconds\": " << S.local_elapsed_seconds << ",\n"
          << "      \"D_hom_diag\": [" << S.D_hom(0,0) << "," << S.D_hom(1,1) << "],\n"
          << "      \"f_hom\": [" << S.f_hom(0) << "," << S.f_hom(1) << "],\n"
          << "      \"staggered_residual_history\": [";
        for (std::size_t r = 0; r < S.residual_history.size(); ++r) {
            f << S.residual_history[r]
              << (r + 1 == S.residual_history.size() ? "" : ",");
        }
        f << "]\n"
          << "    }";
        ++emitted;
        f << (emitted == plan.selected_site_count ? "\n" : ",\n");
    }
    f << "  ]\n}\n";
}

void emit_csv_artifacts(
    const std::filesystem::path& output_dir,
    const std::vector<fall_n::ReducedRCStructuralReplaySample>& hist,
    const fall_n::ReducedRCMultiscaleReplayPlan& plan,
    const std::vector<StaggeredOutcome>& staggered,
    const std::vector<fall_n::FirstInelasticFiberCriterion::Reason>& reasons,
    const std::vector<std::size_t>& trigger_indices)
{
    std::ofstream activation(output_dir / "site_activation.csv");
    activation
        << "site_index,z_over_l,trigger_reason,trigger_sample_index,"
           "trigger_pseudo_time,trigger_drift_mm,peak_abs_curvature_y,"
           "peak_abs_steel_stress_mpa,max_damage_indicator\n";

    std::ofstream response(output_dir / "site_response.csv");
    response
        << "site_index,z_over_l,D_axial_mn,D_flexural_mn_m2,"
           "f_axial_mn,moment_y_mn_m,staggered_iters,"
           "staggered_final_residual,staggered_converged,"
           "relative_moment_feedback_error,"
           "local_status,local_accepted_steps,local_total_nonlinear_iterations,"
           "local_elapsed_seconds\n";

    std::ofstream energy(output_dir / "site_energy.csv");
    energy
        << "site_index,z_over_l,accumulated_abs_macro_work_mn_mm,"
           "peak_abs_base_shear_mn,peak_abs_moment_y_mn_m,"
           "homogenized_moment_y_mn_m,macro_section_history_work\n";

    std::ofstream residuals(output_dir / "staggered_residuals.csv");
    residuals << "site_index,z_over_l,iteration,relative_staggered_residual\n";

    std::size_t k = 0;
    for (const auto& site : plan.sites) {
        if (!site.selected_for_replay) {
            continue;
        }
        const auto& S = staggered[k];
        const auto reason = reasons[k];
        const auto trigger_index = trigger_indices[k];
        const auto* trigger_sample =
            trigger_sample_for_site(hist, site.site_index, trigger_index);

        activation
            << site.site_index << ','
            << site.z_over_l << ','
            << fall_n::FirstInelasticFiberCriterion::reason_label(reason) << ','
            << trigger_index << ','
            << (trigger_sample ? trigger_sample->pseudo_time : -1.0) << ','
            << (trigger_sample ? trigger_sample->drift_mm : 0.0) << ','
            << site.peak_abs_curvature_y << ','
            << site.peak_abs_steel_stress_mpa << ','
            << site.max_damage_indicator << '\n';

        response
            << site.site_index << ','
            << site.z_over_l << ','
            << S.D_hom(0, 0) << ','
            << S.D_hom(1, 1) << ','
            << S.f_hom(0) << ','
            << S.f_hom(1) << ','
            << S.iters << ','
            << S.final_residual << ','
            << (S.converged ? 1 : 0) << ','
            << (site.peak_abs_moment_y_mn_m > 1.0e-12
                    ? std::abs(std::abs(S.f_hom(1)) -
                               site.peak_abs_moment_y_mn_m) /
                          site.peak_abs_moment_y_mn_m
                    : 0.0) << ','
            << fall_n::to_string(S.local_status) << ','
            << S.local_accepted_steps << ','
            << S.local_total_nonlinear_iterations << ','
            << S.local_elapsed_seconds << '\n';

        energy
            << site.site_index << ','
            << site.z_over_l << ','
            << site.accumulated_abs_work_mn_mm << ','
            << site.peak_abs_base_shear_mn << ','
            << site.peak_abs_moment_y_mn_m << ','
            << std::abs(S.f_hom(1)) << ','
            << macro_section_history_work_for_site(hist, site) << '\n';

        for (std::size_t r = 0; r < S.residual_history.size(); ++r) {
            residuals
                << site.site_index << ','
                << site.z_over_l << ','
                << (r + 1) << ','
                << S.residual_history[r] << '\n';
        }
        ++k;
    }
}

}  // namespace

int main(int argc, char** argv) {
    Options o;
    if (!parse_args(argc, argv, o)) return 1;
    namespace fs = std::filesystem;
    fs::create_directories(o.output_dir);

    const auto rows = fall_n::read_structural_history_csv(o.input_csv);
    if (rows.empty()) { std::fprintf(stderr, "[stageB] empty CSV\n"); return 3; }
    [[maybe_unused]] const auto spec = make_run_spec(o);

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
    if (!o.surrogate_smoke) {
        PetscInitializeNoArguments();
    }

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
        std::vector<fall_n::ReducedRCStructuralReplaySample> activated_history;
        if (idx < per_site.size()) {
            activated_history.assign(
                per_site.begin() + static_cast<std::ptrdiff_t>(idx),
                per_site.end());
        }
        if (activated_history.empty()) {
            activated_history = per_site;
        }
        const auto S = o.surrogate_smoke
            ? run_surrogate_staggered(site, o)
            : run_real_xfem_staggered(site, activated_history, o);
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
    emit_csv_artifacts(
        o.output_dir, samples, plan, staggered, reasons, trigger_indices);

    std::printf("[stageB] fe2_column_two_way emitted | selected=%zu triggered=%zu "
                "converged=%zu (%.2f) overall_pass=%d output=%s\n",
                plan.selected_site_count, triggered_count, converged_count,
                conv_frac, overall ? 1 : 0, o.output_dir.string().c_str());
    if (!o.surrogate_smoke) {
        PetscFinalize();
    }
    return overall ? 0 : 4;
}
