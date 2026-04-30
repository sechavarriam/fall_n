#include "src/validation/ReducedRCColumnReferenceSpec.hh"
#include "src/validation/ReducedRCLocalMeshScaleAudit.hh"
#include "src/validation/ReducedRCLocalSiteBatchPlan.hh"
#include "src/validation/ReducedRCLocalSiteReplayRunner.hh"
#include "src/validation/ReducedRCMultiscaleReadinessGate.hh"
#include "src/validation/ReducedRCMultiscaleReplayPlan.hh"
#include "src/validation/ReducedRCMultiscaleRuntimePolicy.hh"
#include "src/validation/ReducedRCMultiscaleValidationStartCatalog.hh"
#include "src/analysis/MixedControlArcLengthContinuation.hh"
#include "src/analysis/NLAnalysis.hh"
#include "src/analysis/PetscNonlinearAnalysisBorderedAdapter.hh"
#include "src/xfem/CohesiveCrackLaw.hh"
#include "src/xfem/ShiftedHeavisideCrackCrossingRebarElement.hh"
#include "src/xfem/ShiftedHeavisideRebarCoupling.hh"
#include "src/xfem/ShiftedHeavisideSolidElement.hh"
#include "src/xfem/XFEMDofManager.hh"
#include "src/xfem/XFEMEnrichment.hh"
#include "src/elements/ElementPolicy.hh"
#include "src/elements/TrussElement.hh"
#include "src/model/Model.hh"
#include "src/model/PrismaticDomainBuilder.hh"
#include "src/materials/LinealElasticMaterial.hh"
#include "src/materials/Material.hh"
#include "src/materials/MaterialPolicy.hh"
#include "src/materials/constitutive_models/non_lineal/MenegottoPintoSteel.hh"

#include <Eigen/Dense>
#include <petsc.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using fall_n::validation_reboot::ReducedRCColumnReferenceSpec;
using fall_n::validation_reboot::default_reduced_rc_column_reference_spec_v;
using fall_n::validation_reboot::to_rc_column_section_spec;
using fall_n::RebarBar;
using fall_n::RebarSpec;
using fall_n::ReinforcedDomainResult;

struct Options {
    std::filesystem::path output_dir{
        "data/output/cyclic_validation/xfem_local_hinge_200mm"};
    std::string amplitudes_mm{"50,100,150,200"};
    int steps_per_segment{8};
    int section_cells_x{24};
    int section_cells_y{24};
    double axial_compression_mn{0.02};
    double top_rotation_drift_ratio{0.5};
    double tangential_slip_drift_ratio{0.02};
    double cohesive_penalty_scale{1.0};
    double characteristic_length_mm{100.0};
    double residual_shear_fraction{0.05};
    double shear_cap_mpa{6.0};
    double compression_cap_mpa{36.3};
    double steel_gauge_length_mm{100.0};
    std::string shear_transfer_law{"compression-gated-opening"};
    double global_xfem_shear_cap_mpa{0.05};
    std::string global_xfem_cohesive_tangent{"central-fallback"};
    std::string global_xfem_cohesive_surface_tangent{
        "frozen-surface-frame"};
    std::string global_xfem_cohesive_traction_measure{"current-spatial"};
    std::string global_xfem_concrete_material{"elastic"};
    std::string global_xfem_crack_band_tangent{"secant"};
    std::string global_xfem_kinematic_formulation{"small-strain"};
    bool allow_guarded_xfem_finite_kinematics{false};
    double global_xfem_residual_tension_fraction{0.02};
    int global_xfem_max_bisections{8};
    int global_xfem_solver_max_iterations{120};
    std::string global_xfem_solver_profile{"backtracking"};
    bool global_xfem_solver_cascade{false};
    bool global_xfem_adaptive_increments{false};
    std::string global_xfem_continuation{"fixed-increment"};
    int global_xfem_bordered_hybrid_disable_streak{3};
    int global_xfem_bordered_hybrid_retry_interval{12};
    double global_xfem_mixed_arc_target{0.075};
    double global_xfem_mixed_arc_reject_factor{1.50};
    double global_xfem_mixed_arc_min_increment{
        std::numeric_limits<double>::quiet_NaN()};
    double global_xfem_mixed_arc_max_increment{
        std::numeric_limits<double>::quiet_NaN()};
    double global_xfem_mixed_arc_reaction_scale_mn{0.02};
    double global_xfem_mixed_arc_damage_weight{0.10};
    bool run_global_xfem_newton{true};
    bool global_xfem_scale_audit_only{false};
    bool global_xfem_incremental_logging{false};
    int global_xfem_nx{2};
    int global_xfem_ny{2};
    int global_xfem_nz{4};
    double global_xfem_bias_power{2.0};
    std::string global_xfem_bias_location{"fixed-end"};
    double global_xfem_crack_z_m{
        std::numeric_limits<double>::quiet_NaN()};
    double global_xfem_rebar_coupling_alpha_scale_over_ec{1.0e4};
    double global_xfem_crack_crossing_rebar_area_scale{0.0};
    double global_xfem_crack_crossing_gauge_length_mm{100.0};
    std::string global_xfem_crack_crossing_rebar_mode{"axial"};
    std::string global_xfem_crack_crossing_bridge_law{"material"};
    std::string global_xfem_crack_crossing_axis_frame{"fixed-global"};
    std::string global_xfem_crack_crossing_host_axis_tangent{"frozen"};
    double global_xfem_crack_crossing_yield_slip_mm{0.25};
    double global_xfem_crack_crossing_yield_force_mn{
        std::numeric_limits<double>::quiet_NaN()};
    double global_xfem_crack_crossing_hardening_ratio{0.0};
    double global_xfem_crack_crossing_force_cap_mn{
        std::numeric_limits<double>::quiet_NaN()};
};

struct HistoryRow {
    int step{0};
    double drift_mm{0.0};
    double theta_y_rad{0.0};
    double tangential_slip_mm{0.0};
    double base_shear_mn{0.0};
    double flexural_shear_mn{0.0};
    double cohesive_shear_mn{0.0};
    double steel_shear_mn{0.0};
    double moment_mn_m{0.0};
    double concrete_moment_mn_m{0.0};
    double steel_moment_mn_m{0.0};
    double axial_reaction_mn{0.0};
    double max_opening_mm{0.0};
    double min_opening_mm{0.0};
    double max_damage{0.0};
    double cracked_area_fraction{0.0};
    double max_abs_steel_stress_mpa{0.0};
};

struct CellState {
    fall_n::xfem::CohesiveCrackState cohesive{};
};

struct PetscLayoutAudit {
    bool completed{false};
    bool global_xfem_solid_kernel_completed{false};
    int solid_elements{0};
    int total_nodes{0};
    int enriched_nodes{0};
    int standard_node_dofs{0};
    int enriched_node_dofs{0};
    int first_cut_element_gather_dofs{0};
    int first_cut_element_expected_xfem_dofs{0};
    int enriched_x_local_index{0};
    int enriched_x_global_index{0};
    double trial_enriched_residual_norm{0.0};
    double trial_enriched_tangent_norm{0.0};
};

struct GlobalXFEMNewtonRow {
    int step{0};
    double p{0.0};
    double drift_mm{0.0};
    double base_shear_mn{0.0};
    double axial_reaction_mn{0.0};
    double max_abs_steel_stress_mpa{0.0};
    double max_host_damage{0.0};
    int damaged_host_points{0};
    int accepted_substeps{0};
    int total_newton_iterations{0};
    int failed_attempts{0};
    int solver_profile_attempts{0};
    int max_bisection_level{0};
    int last_snes_reason{0};
    int last_ksp_reason{0};
    bool accepted_by_small_residual_policy{false};
    double residual_norm{0.0};
    std::string solver_profile_label{};
};

struct GlobalXFEMNewtonSummary {
    bool attempted{false};
    bool completed{false};
    int element_count{0};
    int node_count{0};
    int enriched_node_count{0};
    int local_state_dofs{0};
    int solver_global_dofs{0};
    int rebar_bar_count{0};
    int rebar_element_count{0};
    int rebar_coupling_count{0};
    int crack_crossing_rebar_element_count{0};
    int point_count{0};
    int total_accepted_substeps{0};
    int total_nonlinear_iterations{0};
    int total_failed_attempts{0};
    int total_solver_profile_attempts{0};
    int max_requested_step_nonlinear_iterations{0};
    int max_bisection_level{0};
    int hard_step_count{0};
    int mixed_control_accepted_steps{0};
    int mixed_control_failed_solver_attempts{0};
    int mixed_control_rejected_arc_attempts{0};
    int mixed_control_total_cutbacks{0};
    int bordered_hybrid_attempted_steps{0};
    int bordered_hybrid_successful_steps{0};
    int bordered_hybrid_skipped_steps{0};
    double crack_z_m{0.0};
    std::string crack_z_source{"first_element_midpoint"};
    double total_rebar_area_m2{0.0};
    double peak_abs_drift_mm{0.0};
    double peak_abs_base_shear_mn{0.0};
    double peak_abs_steel_stress_mpa{0.0};
    double max_host_damage{0.0};
    int max_damaged_host_points{0};
    double elapsed_seconds{0.0};
    std::string status{"not_run"};
    std::string failure_reason{};
    std::string requested_kinematic_formulation{"small-strain"};
    std::string effective_kinematic_formulation{"small-strain"};
    std::string kinematic_support_status{
        "xfem_shifted_heaviside_small_strain_only"};
    std::string large_amplitude_kinematic_recommendation{
        "corotational_xfem_is_the_next_supported_extension"};
    std::string continuation_kind{"fixed_increment"};
    std::string mixed_control_status{"not_used"};
    double mixed_control_max_arc_length{0.0};
    double mixed_control_mean_arc_length{0.0};
};

[[nodiscard]] std::string json_escape(const std::string& value)
{
    std::string out;
    out.reserve(value.size());
    for (const char ch : value) {
        switch (ch) {
            case '\\':
                out += "\\\\";
                break;
            case '"':
                out += "\\\"";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                out += ch;
                break;
        }
    }
    return out;
}

[[nodiscard]] std::string json_number_or_null(double value)
{
    if (!std::isfinite(value)) {
        return "null";
    }
    std::ostringstream ss;
    ss << std::setprecision(12) << value;
    return ss.str();
}

[[nodiscard]] std::vector<double> parse_csv_doubles(const std::string& raw)
{
    std::vector<double> values;
    std::stringstream ss(raw);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (!token.empty()) {
            values.push_back(std::stod(token));
        }
    }
    if (values.empty()) {
        throw std::invalid_argument("At least one amplitude must be declared.");
    }
    return values;
}

[[nodiscard]] std::vector<double> cyclic_protocol_mm(
    const std::vector<double>& amplitudes,
    int steps_per_segment)
{
    const int n = std::max(steps_per_segment, 1);
    std::vector<double> protocol;
    protocol.push_back(0.0);

    auto append_segment = [&](double target) {
        const double start = protocol.back();
        for (int i = 1; i <= n; ++i) {
            const double alpha = static_cast<double>(i) / static_cast<double>(n);
            protocol.push_back(start + alpha * (target - start));
        }
    };

    for (const double amplitude : amplitudes) {
        append_segment(amplitude);
        append_segment(-amplitude);
        append_segment(0.0);
    }
    return protocol;
}

[[nodiscard]] fall_n::fracture::CrackShearTransferLawKind
parse_shear_law(std::string value)
{
    std::ranges::replace(value, '_', '-');
    if (value == "constant-residual") {
        return fall_n::fracture::CrackShearTransferLawKind::constant_residual;
    }
    if (value == "opening-exponential") {
        return fall_n::fracture::CrackShearTransferLawKind::opening_exponential;
    }
    if (value == "compression-gated-opening") {
        return fall_n::fracture::CrackShearTransferLawKind::
            compression_gated_opening;
    }
    throw std::invalid_argument("Unsupported XFEM shear transfer law.");
}

void write_usage()
{
    std::cerr
        << "Usage: fall_n_reduced_rc_xfem_reference_benchmark "
        << "[--output-dir path] [--amplitudes-mm 50,100,150,200] "
        << "[--steps-per-segment N] [--section-cells-x N] "
        << "[--section-cells-y N] [--axial-compression-mn value] "
        << "[--top-rotation-drift-ratio value] "
        << "[--tangential-slip-drift-ratio value] "
        << "[--cohesive-penalty-scale value] "
        << "[--characteristic-length-mm value] "
        << "[--residual-shear-fraction value] [--shear-cap-mpa value] "
        << "[--compression-cap-mpa value] [--steel-gauge-length-mm value] "
        << "[--shear-transfer-law constant-residual|opening-exponential|compression-gated-opening] "
        << "[--global-xfem-shear-cap-mpa value] "
        << "[--global-xfem-cohesive-tangent secant|active-set|central|central-fallback] "
        << "[--global-xfem-cohesive-surface-tangent frozen-surface-frame|nanson-geometric-surface-frame|finite-difference-surface-frame] "
        << "[--global-xfem-cohesive-traction-measure reference-nominal|current-spatial|audit-dual] "
        << "[--global-xfem-concrete-material elastic|componentwise-kent-park|cyclic-crack-band] "
        << "[--global-xfem-crack-band-tangent secant|central|central-fallback] "
        << "[--global-xfem-kinematic-formulation small-strain|corotational|total-lagrangian|updated-lagrangian] "
        << "[--allow-guarded-xfem-finite-kinematics] "
        << "[--global-xfem-residual-tension-fraction value] "
        << "[--global-xfem-max-bisections N] "
        << "[--global-xfem-solver-max-iterations N] "
        << "[--global-xfem-solver-profile backtracking|l2|l2-gmres-ilu|l2-gmres-asm|trust-region|dogleg|cascade|robust-cascade|quasi-newton|ngmres|ncg|anderson|richardson] "
        << "[--global-xfem-solver-cascade] "
        << "[--global-xfem-adaptive-increments] "
        << "[--global-xfem-continuation fixed-increment|mixed-arc-length|bordered-fixed-control|bordered-fixed-control-hybrid] "
        << "[--global-xfem-bordered-hybrid-disable-streak N] "
        << "[--global-xfem-bordered-hybrid-retry-interval N] "
        << "[--global-xfem-mixed-arc-target value] "
        << "[--global-xfem-mixed-arc-min-increment value] "
        << "[--global-xfem-mixed-arc-max-increment value] "
        << "[--global-xfem-mixed-arc-reaction-scale-mn value] "
        << "[--skip-global-xfem-newton] [--global-xfem-scale-audit-only] "
        << "[--global-xfem-log-incremental] "
        << "[--global-xfem-nx N] "
        << "[--global-xfem-ny N] [--global-xfem-nz N] "
        << "[--global-xfem-bias-power value] "
        << "[--global-xfem-bias-location fixed-end|loaded-end|both-ends] "
        << "[--global-xfem-crack-z-m value] "
        << "[--global-xfem-rebar-coupling-alpha-scale-over-ec value] "
        << "[--global-xfem-crack-crossing-axis-frame fixed-global|corotational-host] "
        << "[--global-xfem-crack-crossing-host-axis-tangent frozen|finite-difference]\n";
}

[[nodiscard]] Options parse_args(int argc, char** argv)
{
    Options options{};
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag{argv[i]};
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument(
                    "Missing value after " + std::string(flag));
            }
            return argv[++i];
        };

        if (flag == "--help" || flag == "-h") {
            write_usage();
            std::exit(0);
        } else if (flag == "--output-dir") {
            options.output_dir = value();
        } else if (flag == "--amplitudes-mm") {
            options.amplitudes_mm = value();
        } else if (flag == "--steps-per-segment") {
            options.steps_per_segment = std::stoi(value());
        } else if (flag == "--section-cells-x") {
            options.section_cells_x = std::stoi(value());
        } else if (flag == "--section-cells-y") {
            options.section_cells_y = std::stoi(value());
        } else if (flag == "--axial-compression-mn") {
            options.axial_compression_mn = std::stod(value());
        } else if (flag == "--top-rotation-drift-ratio") {
            options.top_rotation_drift_ratio = std::stod(value());
        } else if (flag == "--tangential-slip-drift-ratio") {
            options.tangential_slip_drift_ratio = std::stod(value());
        } else if (flag == "--cohesive-penalty-scale") {
            options.cohesive_penalty_scale = std::stod(value());
        } else if (flag == "--characteristic-length-mm") {
            options.characteristic_length_mm = std::stod(value());
        } else if (flag == "--residual-shear-fraction") {
            options.residual_shear_fraction = std::stod(value());
        } else if (flag == "--shear-cap-mpa") {
            options.shear_cap_mpa = std::stod(value());
        } else if (flag == "--compression-cap-mpa") {
            options.compression_cap_mpa = std::stod(value());
        } else if (flag == "--steel-gauge-length-mm") {
            options.steel_gauge_length_mm = std::stod(value());
        } else if (flag == "--shear-transfer-law") {
            options.shear_transfer_law = value();
        } else if (flag == "--global-xfem-shear-cap-mpa") {
            options.global_xfem_shear_cap_mpa = std::stod(value());
        } else if (flag == "--global-xfem-cohesive-tangent") {
            options.global_xfem_cohesive_tangent = value();
        } else if (flag == "--global-xfem-cohesive-surface-tangent") {
            options.global_xfem_cohesive_surface_tangent = value();
        } else if (flag == "--global-xfem-cohesive-traction-measure") {
            options.global_xfem_cohesive_traction_measure = value();
        } else if (flag == "--global-xfem-concrete-material") {
            options.global_xfem_concrete_material = value();
        } else if (flag == "--global-xfem-crack-band-tangent") {
            options.global_xfem_crack_band_tangent = value();
        } else if (flag == "--global-xfem-kinematic-formulation") {
            options.global_xfem_kinematic_formulation = value();
        } else if (flag == "--allow-guarded-xfem-finite-kinematics") {
            options.allow_guarded_xfem_finite_kinematics = true;
        } else if (flag == "--global-xfem-residual-tension-fraction") {
            options.global_xfem_residual_tension_fraction =
                std::stod(value());
        } else if (flag == "--global-xfem-max-bisections") {
            options.global_xfem_max_bisections = std::stoi(value());
        } else if (flag == "--global-xfem-solver-max-iterations") {
            options.global_xfem_solver_max_iterations = std::stoi(value());
        } else if (flag == "--global-xfem-solver-profile") {
            options.global_xfem_solver_profile = value();
        } else if (flag == "--global-xfem-solver-cascade") {
            options.global_xfem_solver_cascade = true;
        } else if (flag == "--global-xfem-adaptive-increments") {
            options.global_xfem_adaptive_increments = true;
        } else if (flag == "--global-xfem-continuation") {
            options.global_xfem_continuation = value();
        } else if (flag == "--global-xfem-bordered-hybrid-disable-streak") {
            options.global_xfem_bordered_hybrid_disable_streak =
                std::stoi(value());
        } else if (flag == "--global-xfem-bordered-hybrid-retry-interval") {
            options.global_xfem_bordered_hybrid_retry_interval =
                std::stoi(value());
        } else if (flag == "--global-xfem-mixed-arc-target") {
            options.global_xfem_mixed_arc_target = std::stod(value());
        } else if (flag == "--global-xfem-mixed-arc-reject-factor") {
            options.global_xfem_mixed_arc_reject_factor = std::stod(value());
        } else if (flag == "--global-xfem-mixed-arc-min-increment") {
            options.global_xfem_mixed_arc_min_increment =
                std::stod(value());
        } else if (flag == "--global-xfem-mixed-arc-max-increment") {
            options.global_xfem_mixed_arc_max_increment =
                std::stod(value());
        } else if (flag == "--global-xfem-mixed-arc-reaction-scale-mn") {
            options.global_xfem_mixed_arc_reaction_scale_mn =
                std::stod(value());
        } else if (flag == "--global-xfem-mixed-arc-damage-weight") {
            options.global_xfem_mixed_arc_damage_weight =
                std::stod(value());
        } else if (flag == "--skip-global-xfem-newton") {
            options.run_global_xfem_newton = false;
        } else if (flag == "--global-xfem-scale-audit-only") {
            options.global_xfem_scale_audit_only = true;
        } else if (flag == "--global-xfem-log-incremental") {
            options.global_xfem_incremental_logging = true;
        } else if (flag == "--global-xfem-nx") {
            options.global_xfem_nx = std::stoi(value());
        } else if (flag == "--global-xfem-ny") {
            options.global_xfem_ny = std::stoi(value());
        } else if (flag == "--global-xfem-nz") {
            options.global_xfem_nz = std::stoi(value());
        } else if (flag == "--global-xfem-bias-power") {
            options.global_xfem_bias_power = std::stod(value());
        } else if (flag == "--global-xfem-bias-location") {
            options.global_xfem_bias_location = value();
        } else if (flag == "--global-xfem-crack-z-m") {
            options.global_xfem_crack_z_m = std::stod(value());
        } else if (flag == "--global-xfem-rebar-coupling-alpha-scale-over-ec") {
            options.global_xfem_rebar_coupling_alpha_scale_over_ec =
                std::stod(value());
        } else if (flag == "--global-xfem-crack-crossing-rebar-area-scale") {
            options.global_xfem_crack_crossing_rebar_area_scale =
                std::stod(value());
        } else if (flag == "--global-xfem-crack-crossing-gauge-length-mm") {
            options.global_xfem_crack_crossing_gauge_length_mm =
                std::stod(value());
        } else if (flag == "--global-xfem-crack-crossing-rebar-mode") {
            options.global_xfem_crack_crossing_rebar_mode = value();
        } else if (flag == "--global-xfem-crack-crossing-bridge-law") {
            options.global_xfem_crack_crossing_bridge_law = value();
        } else if (flag == "--global-xfem-crack-crossing-axis-frame") {
            options.global_xfem_crack_crossing_axis_frame = value();
        } else if (
            flag == "--global-xfem-crack-crossing-host-axis-tangent") {
            options.global_xfem_crack_crossing_host_axis_tangent = value();
        } else if (flag == "--global-xfem-crack-crossing-yield-slip-mm") {
            options.global_xfem_crack_crossing_yield_slip_mm =
                std::stod(value());
        } else if (flag == "--global-xfem-crack-crossing-yield-force-mn") {
            options.global_xfem_crack_crossing_yield_force_mn =
                std::stod(value());
        } else if (flag == "--global-xfem-crack-crossing-hardening-ratio") {
            options.global_xfem_crack_crossing_hardening_ratio =
                std::stod(value());
        } else if (flag == "--global-xfem-crack-crossing-force-cap-mn") {
            options.global_xfem_crack_crossing_force_cap_mn =
                std::stod(value());
        } else if (flag == "--disable-global-xfem-crack-crossing-rebar") {
            options.global_xfem_crack_crossing_rebar_area_scale = 0.0;
        } else {
            throw std::invalid_argument("Unsupported option: " + std::string(flag));
        }
    }
    return options;
}

[[nodiscard]] std::string normalized_material_token(std::string value)
{
    std::ranges::replace(value, '_', '-');
    return value;
}

[[nodiscard]] std::string normalized_xfem_kinematic_formulation(
    std::string value)
{
    value = normalized_material_token(std::move(value));
    if (value == "small" || value == "small-strain" ||
        value == "small-strain-xfem") {
        return "small-strain";
    }
    if (value == "corotational" || value == "co-rotational" ||
        value == "corotational-xfem") {
        return "corotational";
    }
    if (value == "tl" || value == "total-lagrangian") {
        return "total-lagrangian";
    }
    if (value == "ul" || value == "updated-lagrangian") {
        return "updated-lagrangian";
    }
    return value;
}

void validate_supported_global_xfem_kinematics(const Options& options)
{
    const auto formulation = normalized_xfem_kinematic_formulation(
        options.global_xfem_kinematic_formulation);
    if (formulation == "small-strain" || formulation == "corotational") {
        return;
    }
    if (formulation == "total-lagrangian" ||
        formulation == "updated-lagrangian") {
        (void)options;
        return;
    }
    throw std::invalid_argument(
        "The global shifted-Heaviside XFEM element currently supports "
        "--global-xfem-kinematic-formulation small-strain|corotational|"
        "total-lagrangian|updated-lagrangian. Requested '" +
        formulation +
        "'.");
}

[[nodiscard]] std::string global_xfem_kinematic_support_status(
    std::string formulation)
{
    formulation = normalized_xfem_kinematic_formulation(std::move(formulation));
    if (formulation == "corotational") {
        return "corotational-bulk-and-crack-frame: the shifted-Heaviside "
               "bulk strain filters rigid rotations through a frozen polar "
               "frame, and the cohesive crack normal is rotated with that "
               "frame. The default cohesive tangent freezes dR/du, while "
               "nanson-geometric-surface-frame adds the finite-surface "
               "measure derivative for TL/UL only and "
               "--global-xfem-cohesive-surface-tangent "
               "finite-difference-surface-frame can audit the surface-frame "
               "derivative at higher cost. The crack-crossing bridge can "
               "optionally add a directional host-axis tangent. This is the "
               "first large-rotation audit path, not the final finite-strain "
               "interface formulation.";
    }
    if (formulation == "total-lagrangian") {
        return "promoted total-Lagrangian XFEM audit: the shifted-Heaviside "
               "bulk uses Green-Lagrange strain and the nonlinear material "
               "B operator in the reference configuration. The cohesive "
               "interface now exposes reference-nominal, current-spatial, "
               "and audit-dual traction measures. Use reference-nominal for "
               "the strict reference-surface work-conjugate path and "
               "current-spatial plus nanson-geometric-surface-frame when "
               "auditing the pushed current surface. The finite-difference "
               "surface-frame tangent remains the full residual audit at "
               "higher cost.";
    }
    if (formulation == "updated-lagrangian") {
        return "guarded updated-Lagrangian XFEM audit: the shifted-Heaviside "
               "bulk uses Almansi strain, spatial gradients, and current "
               "volume scaling. The cohesive interface exposes the same "
               "traction-measure policy as TL, but UL remains a second-stage "
               "promotion path until current-configuration history updates "
               "and larger cyclic audits are closed.";
    }
    return "small-strain: the shifted-Heaviside bulk strain and cohesive "
           "crack frame remain in the reference/global small-strain frame.";
}

[[nodiscard]] std::string global_xfem_large_amplitude_recommendation(
    std::string formulation)
{
    formulation = normalized_xfem_kinematic_formulation(std::move(formulation));
    if (formulation == "corotational") {
        return "Use corotational XFEM as the first large-amplitude audit "
               "path. It is appropriate for filtering rigid-body rotation "
               "bias while preserving the current small-strain material laws; "
               "promote it only after rigid-rotation, 200 mm equivalence, and "
               "larger-mesh convergence checks pass.";
    }
    if (formulation == "total-lagrangian" ||
        formulation == "updated-lagrangian") {
        return "Use TL as the first promoted finite-kinematics XFEM path "
               "with reference-nominal cohesive tractions. Use UL as the "
               "current-configuration audit path and require the dual-work "
               "traction audit before treating large cyclic runs as closure "
               "evidence.";
    }
    return "Use small-strain XFEM as the calibrated 200 mm local-model "
           "closure baseline. For larger amplitudes, switch to the "
           "corotational-XFEM audit path before considering TL/UL.";
}

[[nodiscard]] fall_n::xfem::CohesiveCrackTangentMode
parse_global_xfem_cohesive_tangent_mode(std::string value)
{
    std::ranges::replace(value, '_', '-');
    if (value == "secant" || value == "secant-positive") {
        return fall_n::xfem::CohesiveCrackTangentMode::secant_positive;
    }
    if (value == "active-set" || value == "active-set-consistent" ||
        value == "active" || value == "cheap-consistent") {
        return fall_n::xfem::CohesiveCrackTangentMode::
            active_set_consistent;
    }
    if (value == "central" || value == "adaptive-central" ||
        value == "consistent") {
        return fall_n::xfem::CohesiveCrackTangentMode::
            adaptive_central_difference;
    }
    if (value == "central-fallback" ||
        value == "adaptive-central-fallback" ||
        value == "consistent-fallback") {
        return fall_n::xfem::CohesiveCrackTangentMode::
            adaptive_central_difference_with_secant_fallback;
    }
    throw std::invalid_argument(
        "Unsupported global XFEM cohesive tangent mode: " + value);
}

[[nodiscard]] fall_n::xfem::ShiftedHeavisideSolidOptions::
    CohesiveSurfaceTangentMode
    parse_global_xfem_cohesive_surface_tangent_mode(std::string value)
{
    std::ranges::replace(value, '_', '-');
    if (value == "frozen" || value == "frozen-frame" ||
        value == "frozen-surface" ||
        value == "frozen-surface-frame") {
        return fall_n::xfem::ShiftedHeavisideSolidOptions::
            CohesiveSurfaceTangentMode::frozen_surface_frame;
    }
    if (value == "nanson" || value == "nanson-geometric" ||
        value == "nanson-geometric-frame" ||
        value == "nanson-geometric-surface-frame" ||
        value == "analytic-nanson" ||
        value == "analytic-nanson-surface-frame") {
        return fall_n::xfem::ShiftedHeavisideSolidOptions::
            CohesiveSurfaceTangentMode::nanson_geometric_surface_frame;
    }
    if (value == "fd" || value == "finite-difference" ||
        value == "finite-difference-frame" ||
        value == "finite-difference-surface-frame") {
        return fall_n::xfem::ShiftedHeavisideSolidOptions::
            CohesiveSurfaceTangentMode::finite_difference_surface_frame;
    }
    throw std::invalid_argument(
        "Unsupported global XFEM cohesive surface tangent mode: " + value);
}

[[nodiscard]] fall_n::xfem::ShiftedHeavisideCohesiveTractionMeasureKind
parse_global_xfem_cohesive_traction_measure(std::string value)
{
    std::ranges::replace(value, '_', '-');
    if (value == "reference" || value == "nominal" ||
        value == "reference-nominal" || value == "pk" ||
        value == "reference-piola") {
        return fall_n::xfem::ShiftedHeavisideCohesiveTractionMeasureKind::
            reference_nominal;
    }
    if (value == "current" || value == "spatial" ||
        value == "current-spatial" || value == "cauchy") {
        return fall_n::xfem::ShiftedHeavisideCohesiveTractionMeasureKind::
            current_spatial;
    }
    if (value == "audit" || value == "dual" || value == "audit-dual") {
        return fall_n::xfem::ShiftedHeavisideCohesiveTractionMeasureKind::
            audit_dual;
    }
    throw std::invalid_argument(
        "Unsupported global XFEM cohesive traction measure: " + value);
}

[[nodiscard]] fall_n::LongitudinalBiasLocation
parse_longitudinal_bias_location(std::string value)
{
    value = normalized_material_token(std::move(value));
    if (value == "fixed-end" || value == "fixed" || value == "base") {
        return fall_n::LongitudinalBiasLocation::FixedEnd;
    }
    if (value == "loaded-end" || value == "loaded" || value == "tip") {
        return fall_n::LongitudinalBiasLocation::LoadedEnd;
    }
    if (value == "both-ends" || value == "both") {
        return fall_n::LongitudinalBiasLocation::BothEnds;
    }
    throw std::invalid_argument(
        "Unsupported global XFEM bias location: " + value);
}

[[nodiscard]] bool crack_plane_intersects_grid_level(
    const fall_n::PrismaticGrid& grid,
    double crack_z_m)
{
    const double tolerance =
        100.0 * std::numeric_limits<double>::epsilon() *
        std::max(1.0, grid.length);
    for (int iz = 0; iz < grid.nodes_z(); ++iz) {
        if (std::abs(grid.z_coordinate(iz) - crack_z_m) <= tolerance) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] double choose_global_xfem_crack_z(
    const Options& options,
    const fall_n::PrismaticGrid& grid,
    GlobalXFEMNewtonSummary& summary)
{
    if (std::isfinite(options.global_xfem_crack_z_m)) {
        if (!(options.global_xfem_crack_z_m > 0.0 &&
              options.global_xfem_crack_z_m < grid.length)) {
            throw std::invalid_argument(
                "--global-xfem-crack-z-m must lie strictly inside the column.");
        }
        if (crack_plane_intersects_grid_level(
                grid,
                options.global_xfem_crack_z_m)) {
            throw std::invalid_argument(
                "--global-xfem-crack-z-m cannot coincide with a mesh level; "
                "place the crack inside an element to avoid Heaviside "
                "enrichment degeneracy.");
        }
        summary.crack_z_source = "user_prescribed_physical_position";
        return options.global_xfem_crack_z_m;
    }

    summary.crack_z_source = "first_element_midpoint";
    return 0.5 * (grid.z_coordinate(0) + grid.z_coordinate(grid.step));
}

[[nodiscard]] CyclicCrackBandConcrete3DTangentMode
parse_crack_band_tangent_mode(std::string value)
{
    std::ranges::replace(value, '_', '-');
    if (value == "secant" || value == "secant-positive") {
        return CyclicCrackBandConcrete3DTangentMode::SecantPositive;
    }
    if (value == "central" || value == "adaptive-central" ||
        value == "consistent") {
        return CyclicCrackBandConcrete3DTangentMode::
            AdaptiveCentralDifference;
    }
    if (value == "central-fallback" ||
        value == "adaptive-central-fallback" ||
        value == "consistent-fallback") {
        return CyclicCrackBandConcrete3DTangentMode::
            AdaptiveCentralDifferenceWithSecantFallback;
    }
    throw std::invalid_argument(
        "Unsupported cyclic crack-band tangent mode: " + value);
}

[[nodiscard]] Material<ThreeDimensionalMaterial> make_global_xfem_concrete_material(
    const Options& options,
    const ReducedRCColumnReferenceSpec& spec)
{
    const double ec_mpa = 4700.0 * std::sqrt(spec.concrete_fpc_mpa);
    const std::string token =
        normalized_material_token(options.global_xfem_concrete_material);

    if (token == "elastic" || token == "linear-elastic") {
        return Material<ThreeDimensionalMaterial>{
            ContinuumIsotropicElasticMaterial{ec_mpa, spec.concrete_nu},
            ElasticUpdate{}};
    }

    if (token == "cyclic-crack-band" || token == "cyclic-crackband") {
        const double ft_mpa =
            spec.concrete_ft_ratio * spec.concrete_fpc_mpa;
        CyclicCrackBandConcrete3D relation{
            spec.concrete_fpc_mpa,
            ec_mpa,
            spec.concrete_nu,
            ft_mpa,
            0.06,
            options.characteristic_length_mm,
            std::clamp(
                options.global_xfem_residual_tension_fraction,
                1.0e-8,
                1.0),
            options.residual_shear_fraction,
            0.05,
            -0.002,
            0.20,
            0.0,
            options.residual_shear_fraction,
            1.0,
            parse_shear_law(options.shear_transfer_law),
            1.0};
        relation.set_material_tangent_mode(
            parse_crack_band_tangent_mode(
                options.global_xfem_crack_band_tangent));
        return Material<ThreeDimensionalMaterial>{
            ContinuumCyclicCrackBandConcreteMaterial{relation},
            InelasticUpdate{}};
    }

    if (token == "componentwise-kent-park" ||
        token == "componentwise-kentpark" ||
        token == "kent-park-control") {
        const KentParkConcreteTensionConfig tension{
            .tensile_strength =
                spec.concrete_ft_ratio * spec.concrete_fpc_mpa,
            .softening_multiplier =
                spec.concrete_tension_softening_multiplier,
            .residual_tangent_ratio =
                std::clamp(
                    options.global_xfem_residual_tension_fraction,
                    1.0e-8,
                    1.0),
            .crack_transition_multiplier =
                spec.concrete_tension_transition_multiplier};
        return Material<ThreeDimensionalMaterial>{
            ContinuumComponentwiseKentParkConcreteMaterial{
                spec.concrete_fpc_mpa,
                tension,
                spec.concrete_nu},
            InelasticUpdate{}};
    }

    throw std::invalid_argument(
        "Unsupported global XFEM concrete material: " +
        options.global_xfem_concrete_material);
}

[[nodiscard]] bool global_xfem_concrete_material_is_nonlinear(
    const Options& options)
{
    const std::string token =
        normalized_material_token(options.global_xfem_concrete_material);
    return token != "elastic" && token != "linear-elastic";
}

[[nodiscard]] double clamp(double value, double lo, double hi)
{
    return std::min(std::max(value, lo), hi);
}

[[nodiscard]] double interpolate_protocol(
    const std::vector<double>& protocol,
    double p)
{
    if (protocol.empty()) {
        return 0.0;
    }
    if (protocol.size() == 1) {
        return protocol.front();
    }
    const double clamped_p = clamp(p, 0.0, 1.0);
    const double scaled =
        clamped_p * static_cast<double>(protocol.size() - 1);
    const auto i0 = static_cast<std::size_t>(std::floor(scaled));
    if (i0 + 1 >= protocol.size()) {
        return protocol.back();
    }
    const double alpha = scaled - static_cast<double>(i0);
    return protocol[i0] + alpha * (protocol[i0 + 1] - protocol[i0]);
}

[[nodiscard]] RebarSpec make_global_xfem_rebar_spec(
    const ReducedRCColumnReferenceSpec& spec)
{
    const auto section = to_rc_column_section_spec(spec);
    const auto positions = rc_column_longitudinal_bar_positions(section);
    const double area = rc_column_longitudinal_bar_area(section);
    const double diameter = std::sqrt(4.0 * area / std::numbers::pi);

    RebarSpec rebar{};
    rebar.bars.reserve(positions.size());
    for (const auto& [x, y] : positions) {
        rebar.bars.push_back(RebarBar{
            .ly = x,
            .lz = y,
            .area = area,
            .diameter = diameter,
            .group = "XFEMLongitudinalRebar"});
    }
    return rebar;
}

[[nodiscard]] double total_rebar_area(const RebarSpec& rebar) noexcept
{
    double area = 0.0;
    for (const auto& bar : rebar.bars) {
        area += bar.area;
    }
    return area;
}

[[nodiscard]] fall_n::ReducedRCLocalConstitutiveCostKind
global_xfem_constitutive_cost_kind(const Options& options)
{
    const std::string token =
        normalized_material_token(options.global_xfem_concrete_material);
    if (token == "elastic" || token == "linear-elastic") {
        return fall_n::ReducedRCLocalConstitutiveCostKind::elastic_proxy;
    }
    if (token == "ko-bathe" || token == "ko-bathe-heavy-reference") {
        return fall_n::ReducedRCLocalConstitutiveCostKind::
            ko_bathe_heavy_reference;
    }
    return fall_n::ReducedRCLocalConstitutiveCostKind::
        cyclic_crack_band_xfem;
}

[[nodiscard]] fall_n::ReducedRCLocalMeshScaleInput
make_global_xfem_local_mesh_scale_input(
    const Options& options,
    const ReducedRCColumnReferenceSpec& spec)
{
    const auto rebar = make_global_xfem_rebar_spec(spec);
    return fall_n::ReducedRCLocalMeshScaleInput{
        .nx = static_cast<std::size_t>(std::max(options.global_xfem_nx, 1)),
        .ny = static_cast<std::size_t>(std::max(options.global_xfem_ny, 1)),
        .nz = static_cast<std::size_t>(std::max(options.global_xfem_nz, 2)),
        .topology = fall_n::ReducedRCLocalCellTopologyKind::hex8_lagrange,
        .constitutive_cost = global_xfem_constitutive_cost_kind(options),
        .shifted_heaviside_xfem = true,
        .planar_crack_count = 1,
        .longitudinal_bar_count = rebar.bars.size(),
        .bars_have_independent_nodes = true,
        .rebar_subsegments_per_host_element = 1,
        .crack_crossing_bridge_enabled =
            options.global_xfem_crack_crossing_rebar_area_scale > 0.0};
}

[[nodiscard]] fall_n::ReducedRCLocalMeshScaleAudit
make_global_xfem_scale_audit(
    const Options& options,
    const ReducedRCColumnReferenceSpec& spec)
{
    return fall_n::make_reduced_rc_local_mesh_scale_audit(
        make_global_xfem_local_mesh_scale_input(options, spec));
}

void write_global_xfem_scale_audit_json(
    const std::filesystem::path& path,
    const Options& options,
    const ReducedRCColumnReferenceSpec& spec,
    const fall_n::ReducedRCLocalMeshScaleAudit& audit)
{
    std::ofstream out(path);
    out << std::setprecision(12)
        << "{\n"
        << "  \"tool\": \"fall_n\",\n"
        << "  \"driver_kind\": \"global_shifted_heaviside_xfem_scale_audit\",\n"
        << "  \"dry_run\": true,\n"
        << "  \"mesh\": {\n"
        << "    \"topology\": \""
        << fall_n::to_string(fall_n::ReducedRCLocalCellTopologyKind::
                                 hex8_lagrange)
        << "\",\n"
        << "    \"nx\": " << std::max(options.global_xfem_nx, 1) << ",\n"
        << "    \"ny\": " << std::max(options.global_xfem_ny, 1) << ",\n"
        << "    \"nz\": " << std::max(options.global_xfem_nz, 2) << ",\n"
        << "    \"longitudinal_bias_power\": "
        << options.global_xfem_bias_power << ",\n"
        << "    \"longitudinal_bias_location\": \""
        << json_escape(normalized_material_token(
               options.global_xfem_bias_location))
        << "\"\n"
        << "  },\n"
        << "  \"physics\": {\n"
        << "    \"host_material\": \""
        << json_escape(normalized_material_token(
               options.global_xfem_concrete_material))
        << "\",\n"
        << "    \"constitutive_cost_kind\": \""
        << fall_n::to_string(global_xfem_constitutive_cost_kind(options))
        << "\",\n"
        << "    \"concrete_fpc_mpa\": " << spec.concrete_fpc_mpa << ",\n"
        << "    \"shifted_heaviside_xfem\": true,\n"
        << "    \"cohesive_surface_tangent\": \""
        << json_escape(std::string(fall_n::xfem::to_string(
               parse_global_xfem_cohesive_surface_tangent_mode(
                   options.global_xfem_cohesive_surface_tangent))))
        << "\",\n"
        << "    \"cohesive_traction_measure\": \""
        << json_escape(std::string(fall_n::xfem::to_string(
               parse_global_xfem_cohesive_traction_measure(
                   options.global_xfem_cohesive_traction_measure))))
        << "\",\n"
        << "    \"longitudinal_bar_count\": "
        << make_global_xfem_rebar_spec(spec).bars.size() << ",\n"
        << "    \"crack_crossing_bridge_enabled\": "
        << (options.global_xfem_crack_crossing_rebar_area_scale > 0.0
                ? "true"
                : "false")
        << "\n"
        << "  },\n"
        << "  \"kinematics\": {\n"
        << "    \"requested_formulation\": \""
        << json_escape(normalized_xfem_kinematic_formulation(
               options.global_xfem_kinematic_formulation))
        << "\",\n"
        << "    \"effective_formulation\": \""
        << json_escape(normalized_xfem_kinematic_formulation(
               options.global_xfem_kinematic_formulation))
        << "\",\n"
        << "    \"guarded_finite_kinematics_allowed\": "
        << (options.allow_guarded_xfem_finite_kinematics ? "true" : "false")
        << ",\n"
        << "    \"xfem_formulation_guard\": \""
        << json_escape(global_xfem_kinematic_support_status(
               options.global_xfem_kinematic_formulation))
        << "\",\n"
        << "    \"large_amplitude_recommendation\": \""
        << json_escape(global_xfem_large_amplitude_recommendation(
               options.global_xfem_kinematic_formulation))
        << "\"\n"
        << "  },\n"
        << "  \"counts\": {\n"
        << "    \"host_element_count\": " << audit.host_element_count << ",\n"
        << "    \"host_node_count\": " << audit.host_node_count << ",\n"
        << "    \"host_gauss_points_per_element\": "
        << audit.host_gauss_points_per_element << ",\n"
        << "    \"host_material_point_count\": "
        << audit.host_material_point_count << ",\n"
        << "    \"enriched_node_count\": "
        << audit.enriched_node_count << ",\n"
        << "    \"host_displacement_dofs\": "
        << audit.host_displacement_dofs << ",\n"
        << "    \"enrichment_dofs\": " << audit.enrichment_dofs << ",\n"
        << "    \"rebar_node_count\": " << audit.rebar_node_count << ",\n"
        << "    \"rebar_dofs\": " << audit.rebar_dofs << ",\n"
        << "    \"crack_crossing_bridge_element_count\": "
        << audit.crack_crossing_bridge_element_count << ",\n"
        << "    \"estimated_total_state_dofs\": "
        << audit.estimated_total_state_dofs << ",\n"
        << "    \"estimated_sparse_nonzeros\": "
        << audit.estimated_sparse_nonzeros << "\n"
        << "  },\n"
        << "  \"memory_mib\": {\n"
        << "    \"one_vector\": " << audit.vector_mib << ",\n"
        << "    \"newton_workspace\": "
        << audit.newton_workspace_mib << ",\n"
        << "    \"sparse_matrix\": " << audit.sparse_matrix_mib << ",\n"
        << "    \"direct_factorization_risk\": "
        << audit.direct_factorization_risk_mib << ",\n"
        << "    \"material_state\": "
        << audit.material_state_mib << ",\n"
        << "    \"estimated_hot_state\": "
        << audit.estimated_hot_state_mib << "\n"
        << "  },\n"
        << "  \"recommendations\": {\n"
        << "    \"solver_advice\": \""
        << fall_n::to_string(audit.solver_advice) << "\",\n"
        << "    \"seed_state_cache_recommended\": "
        << (audit.seed_state_cache_recommended ? "true" : "false")
        << ",\n"
        << "    \"newton_warm_start_recommended\": "
        << (audit.newton_warm_start_recommended ? "true" : "false")
        << ",\n"
        << "    \"site_level_openmp_recommended\": "
        << (audit.site_level_openmp_recommended ? "true" : "false")
        << ",\n"
        << "    \"global_petsc_assembly_openmp_recommended\": "
        << (audit.global_petsc_assembly_openmp_recommended ? "true"
                                                           : "false")
        << ",\n"
        << "    \"symmetric_matrix_storage_recommended\": "
        << (audit.symmetric_matrix_storage_recommended ? "true" : "false")
        << ",\n"
        << "    \"symmetric_matrix_storage_requires_tangent_audit\": "
        << (audit.symmetric_matrix_storage_requires_tangent_audit ? "true"
                                                                  : "false")
        << ",\n"
        << "    \"block_matrix_storage_candidate\": "
        << (audit.block_matrix_storage_candidate ? "true" : "false")
        << ",\n"
        << "    \"field_split_or_asm_preconditioner_recommended\": "
        << (audit.field_split_or_asm_preconditioner_recommended ? "true"
                                                                : "false")
        << ",\n"
        << "    \"plain_gmres_ilu_rejected_for_enriched_branch\": "
        << (audit.plain_gmres_ilu_rejected_for_enriched_branch ? "true"
                                                               : "false")
        << ",\n"
        << "    \"parallelism_note\": "
           "\"OpenMP is recommended first across independent local sites; "
           "single-model PETSc Mat/Vec assembly remains serial unless an "
           "explicit thread-safe assembly backend is introduced.\"\n"
        << "  }\n"
        << "}\n";
}

struct AxisLocation {
    int element_index{0};
    double parent_coordinate{0.0};
};

[[nodiscard]] std::vector<double> prismatic_corner_levels(
    const std::vector<double>& expanded_levels,
    int element_count,
    int step)
{
    std::vector<double> levels;
    levels.reserve(static_cast<std::size_t>(element_count + 1));
    for (int i = 0; i <= element_count; ++i) {
        levels.push_back(
            expanded_levels.at(static_cast<std::size_t>(i * step)));
    }
    return levels;
}

[[nodiscard]] AxisLocation locate_axis_position(
    const std::vector<double>& corner_levels,
    double coordinate)
{
    if (corner_levels.size() < 2) {
        throw std::invalid_argument(
            "Crack-crossing rebar placement needs at least two corner levels.");
    }

    const double lo = corner_levels.front();
    const double hi = corner_levels.back();
    const double tol = 1.0e-12 * std::max(1.0, std::abs(hi - lo));
    if (coordinate < lo - tol || coordinate > hi + tol) {
        throw std::out_of_range(
            "Crack-crossing rebar coordinate is outside the host mesh.");
    }

    for (std::size_t i = 0; i + 1 < corner_levels.size(); ++i) {
        const double a = corner_levels[i];
        const double b = corner_levels[i + 1];
        const bool last = (i + 2 == corner_levels.size());
        if (coordinate < a - tol || coordinate > b + (last ? tol : -tol)) {
            continue;
        }
        const double length = b - a;
        if (std::abs(length) <= tol) {
            throw std::runtime_error(
                "Crack-crossing rebar encountered a degenerate mesh interval.");
        }
        return AxisLocation{
            static_cast<int>(i),
            -1.0 + 2.0 * (coordinate - a) / length};
    }

    throw std::out_of_range(
        "Crack-crossing rebar coordinate could not be located in the host mesh.");
}

struct CrackCrossingRebarSite {
    std::size_t host_element_index{0};
    std::array<double, 3> local_coordinates{};
    double area{0.0};
};

[[nodiscard]] std::vector<CrackCrossingRebarSite>
make_crack_crossing_rebar_sites(
    const ReinforcedDomainResult& reinforced,
    const RebarSpec& rebar,
    double crack_z_m,
    double area_scale)
{
    if (!(area_scale > 0.0)) {
        return {};
    }

    const auto& grid = reinforced.grid;
    const auto x_levels =
        prismatic_corner_levels(grid.x_coordinates, grid.nx, grid.step);
    const auto y_levels =
        prismatic_corner_levels(grid.y_coordinates, grid.ny, grid.step);
    const auto z_levels =
        prismatic_corner_levels(grid.z_coordinates, grid.nz, grid.step);
    const auto z_location = locate_axis_position(z_levels, crack_z_m);

    std::vector<CrackCrossingRebarSite> sites;
    sites.reserve(rebar.bars.size());
    for (const auto& bar : rebar.bars) {
        const auto x_location = locate_axis_position(x_levels, bar.ly);
        const auto y_location = locate_axis_position(y_levels, bar.lz);
        const auto host_element_index = static_cast<std::size_t>(
            z_location.element_index * grid.nx * grid.ny +
            y_location.element_index * grid.nx +
            x_location.element_index);
        if (host_element_index >= reinforced.rebar_range.first) {
            throw std::runtime_error(
                "Crack-crossing rebar site resolved outside the host solid range.");
        }
        sites.push_back(CrackCrossingRebarSite{
            .host_element_index = host_element_index,
            .local_coordinates = {
                x_location.parent_coordinate,
                y_location.parent_coordinate,
                z_location.parent_coordinate},
            .area = area_scale * bar.area});
    }
    return sites;
}

[[nodiscard]] std::vector<Eigen::Vector3d> crack_crossing_rebar_bridge_axes(
    std::string mode)
{
    std::ranges::replace(mode, '_', '-');
    if (mode == "none" || mode == "off" || mode == "disabled") {
        return {};
    }
    if (mode == "axial" || mode == "normal") {
        return {Eigen::Vector3d::UnitZ()};
    }
    if (mode == "dowel-x" || mode == "shear-x" || mode == "tangential-x") {
        return {Eigen::Vector3d::UnitX()};
    }
    if (mode == "dowel-y" || mode == "shear-y" || mode == "tangential-y") {
        return {Eigen::Vector3d::UnitY()};
    }
    if (mode == "axial+dowel-x" || mode == "normal+dowel-x") {
        return {Eigen::Vector3d::UnitZ(), Eigen::Vector3d::UnitX()};
    }
    if (mode == "axial+dowel-y" || mode == "normal+dowel-y") {
        return {Eigen::Vector3d::UnitZ(), Eigen::Vector3d::UnitY()};
    }
    throw std::invalid_argument(
        "Unsupported crack-crossing rebar bridge mode: " + mode);
}

[[nodiscard]] bool crack_crossing_bridge_uses_bounded_slip(std::string law)
{
    std::ranges::replace(law, '_', '-');
    if (law == "material" || law == "material-strain" ||
        law == "menegotto" || law == "steel") {
        return false;
    }
    if (law == "bounded-slip" || law == "bounded" ||
        law == "elastoplastic-slip" || law == "dowel-slip") {
        return true;
    }
    throw std::invalid_argument(
        "Unsupported crack-crossing bridge law: " + law);
}

[[nodiscard]] fall_n::xfem::CrackCrossingRebarAxisFrameKind
parse_crack_crossing_axis_frame(std::string frame)
{
    frame = normalized_material_token(std::move(frame));
    if (frame == "fixed" || frame == "fixed-global" ||
        frame == "global") {
        return fall_n::xfem::CrackCrossingRebarAxisFrameKind::fixed_global;
    }
    if (frame == "corotational" || frame == "corotational-host" ||
        frame == "host-corotational") {
        return fall_n::xfem::CrackCrossingRebarAxisFrameKind::
            corotational_host;
    }
    throw std::invalid_argument(
        "Unsupported crack-crossing bridge axis frame: " + frame);
}

[[nodiscard]] bool parse_crack_crossing_host_axis_tangent(std::string tangent)
{
    tangent = normalized_material_token(std::move(tangent));
    if (tangent == "frozen" || tangent == "frozen-axis" ||
        tangent == "none" || tangent == "off") {
        return false;
    }
    if (tangent == "finite-difference" || tangent == "fd" ||
        tangent == "directional-fd" || tangent == "on") {
        return true;
    }
    throw std::invalid_argument(
        "Unsupported crack-crossing host-axis tangent mode: " + tangent);
}

[[nodiscard]] fall_n::xfem::BoundedSlipBridgeParameters
make_bounded_slip_bridge_parameters(
    const Options& options,
    const ReducedRCColumnReferenceSpec& spec,
    std::size_t bridge_count,
    double area_scale)
{
    const double default_total_cap =
        options.global_xfem_shear_cap_mpa *
        spec.section_b_m *
        spec.section_h_m *
        std::max(area_scale, 0.0);
    const double default_yield_force =
        default_total_cap /
        static_cast<double>(std::max<std::size_t>(bridge_count, 1));
    const double yield_force = std::isfinite(
        options.global_xfem_crack_crossing_yield_force_mn)
        ? options.global_xfem_crack_crossing_yield_force_mn
        : default_yield_force;
    const double yield_slip_m =
        std::max(
            options.global_xfem_crack_crossing_yield_slip_mm,
            1.0e-9) / 1000.0;
    const double force_cap = std::isfinite(
        options.global_xfem_crack_crossing_force_cap_mn)
        ? options.global_xfem_crack_crossing_force_cap_mn
        : yield_force;
    return fall_n::xfem::BoundedSlipBridgeParameters{
        .initial_stiffness_mn_per_m =
            std::max(yield_force, 1.0e-18) / yield_slip_m,
        .yield_force_mn = std::max(yield_force, 0.0),
        .hardening_ratio =
            std::clamp(
                options.global_xfem_crack_crossing_hardening_ratio,
                0.0,
                0.999),
        .force_cap_mn = std::max(force_cap, 0.0)};
}

[[nodiscard]] std::vector<std::size_t> global_xfem_rebar_end_nodes(
    const ReinforcedDomainResult& reinforced,
    bool top)
{
    const auto nodes_per_bar =
        static_cast<std::size_t>(
            reinforced.grid.step * reinforced.grid.nz + 1);
    std::vector<std::size_t> nodes;
    nodes.reserve(reinforced.bar_diameters.size());
    for (std::size_t bar = 0; bar < reinforced.bar_diameters.size(); ++bar) {
        const auto offset =
            bar * nodes_per_bar + (top ? nodes_per_bar - 1 : 0);
        nodes.push_back(static_cast<std::size_t>(
            reinforced.embeddings.at(offset).rebar_node_id));
    }
    return nodes;
}

[[nodiscard]] PetscLayoutAudit run_global_petsc_xfem_layout_audit(
    const ReducedRCColumnReferenceSpec& spec)
{
    using fall_n::HexOrder;
    using fall_n::PrismaticSpec;
    using fall_n::make_prismatic_domain;
    using fall_n::xfem::PlaneCrackLevelSet;
    using fall_n::xfem::petsc_global_dof_index;
    using fall_n::xfem::petsc_local_dof_index;
    using fall_n::xfem::shifted_heaviside_enriched_component;

    PetscLayoutAudit audit{};
    auto [domain, grid] = make_prismatic_domain(PrismaticSpec{
        .width = spec.section_b_m,
        .height = spec.section_h_m,
        .length = spec.column_height_m,
        .nx = 1,
        .ny = 1,
        .nz = 2,
        .hex_order = HexOrder::Linear,
        .physical_group = "XFEMLayoutAuditSolid"});
    (void)grid;

    const PlaneCrackLevelSet base_crack{
        Eigen::Vector3d{0.0, 0.0, 0.25 * spec.column_height_m},
        Eigen::Vector3d::UnitZ()};

    ContinuumIsotropicElasticMaterial mat_site{
        4700.0 * std::sqrt(spec.concrete_fpc_mpa),
        spec.concrete_nu};
    Material<ThreeDimensionalMaterial> material{
        mat_site,
        ElasticUpdate{}};

    const double ec_mpa = 4700.0 * std::sqrt(spec.concrete_fpc_mpa);
    const double gc_mpa = ec_mpa / (2.0 * (1.0 + spec.concrete_nu));
    auto cohesive = fall_n::xfem::make_crack_band_consistent_cohesive_law(
        ec_mpa,
        gc_mpa,
        spec.concrete_ft_ratio * spec.concrete_fpc_mpa,
        0.06,
        100.0,
        1.0,
        1.0,
        1.0,
        0.05);

    using XFEMElement =
        fall_n::xfem::ShiftedHeavisideSolidElement<ThreeDimensionalMaterial>;
    std::vector<XFEMElement> elements;
    elements.reserve(domain.num_elements());
    for (auto& geometry : domain.elements()) {
        elements.emplace_back(&geometry, material, base_crack, cohesive);
    }

    using XFEMModel = Model<
        ThreeDimensionalMaterial,
        continuum::SmallStrain,
        3,
        SingleElementPolicy<XFEMElement>>;
    XFEMModel model{domain, std::move(elements)};
    model.setup();

    audit.completed = true;
    audit.solid_elements = static_cast<int>(domain.num_elements());
    audit.total_nodes = static_cast<int>(domain.num_nodes());
    audit.standard_node_dofs = static_cast<int>(domain.node(8).num_dof());
    audit.enriched_node_dofs = static_cast<int>(domain.node(0).num_dof());
    for (const auto& node : domain.nodes()) {
        if (fall_n::xfem::node_has_shifted_heaviside_enrichment(node)) {
            ++audit.enriched_nodes;
        }
    }

    auto& first_element = model.elements().front();
    const auto& xfem_dofs = first_element.get_dof_indices();
    audit.first_cut_element_gather_dofs =
        static_cast<int>(xfem_dofs.size());
    audit.first_cut_element_expected_xfem_dofs =
        static_cast<int>(2 * 3 * first_element.num_nodes());
    Eigen::VectorXd trial_u =
        Eigen::VectorXd::Zero(static_cast<Eigen::Index>(xfem_dofs.size()));
    for (Eigen::Index local = 5; local < trial_u.size(); local += 6) {
        trial_u[local] = 1.0e-3;
    }
    const auto trial_f = first_element.compute_internal_force_vector(trial_u);
    const auto trial_K = first_element.compute_tangent_stiffness_matrix(trial_u);
    audit.trial_enriched_residual_norm =
        trial_f.tail(trial_f.size() / 2).norm();
    audit.trial_enriched_tangent_norm =
        trial_K.bottomRightCorner(trial_K.rows() / 2, trial_K.cols() / 2).norm();
    audit.global_xfem_solid_kernel_completed =
        audit.trial_enriched_residual_norm > 0.0 &&
        audit.trial_enriched_tangent_norm > 0.0;

    PetscSection section = nullptr;
    ISLocalToGlobalMapping local_to_global = nullptr;
    DMGetLocalSection(model.get_plex(), &section);
    DMGetLocalToGlobalMapping(model.get_plex(), &local_to_global);
    const auto enriched_x = static_cast<PetscInt>(
        shifted_heaviside_enriched_component<3>(0));
    audit.enriched_x_local_index = static_cast<int>(
        petsc_local_dof_index(
            section,
            domain.node(0).sieve_id.value(),
            enriched_x));
    audit.enriched_x_global_index = static_cast<int>(
        petsc_global_dof_index(
            section,
            local_to_global,
            domain.node(0).sieve_id.value(),
            enriched_x));

    if (audit.enriched_nodes <= 0 ||
        audit.standard_node_dofs != 3 ||
        audit.enriched_node_dofs != 6 ||
        audit.first_cut_element_gather_dofs !=
            audit.first_cut_element_expected_xfem_dofs ||
        audit.enriched_x_local_index < 0 ||
        audit.enriched_x_global_index < 0 ||
        !audit.global_xfem_solid_kernel_completed) {
        throw std::runtime_error(
            "Global PETSc XFEM solid-element audit failed.");
    }

    return audit;
}

template <typename ModelT>
[[nodiscard]] double sum_support_internal_force_component(
    const ModelT& model,
    const std::vector<std::size_t>& support_nodes,
    std::size_t component)
{
    Vec f_int = nullptr;
    VecDuplicate(model.state_vector(), &f_int);
    VecSet(f_int, 0.0);

    auto& mutable_model = const_cast<ModelT&>(model);
    for (auto& element : mutable_model.elements()) {
        element.compute_internal_forces(model.state_vector(), f_int);
    }

    VecAssemblyBegin(f_int);
    VecAssemblyEnd(f_int);

    double resultant = 0.0;
    for (const auto node_id : support_nodes) {
        const auto dofs = model.get_domain().node(node_id).dof_index();
        if (component >= dofs.size()) {
            continue;
        }
        PetscScalar value{};
        const PetscInt dof_index = static_cast<PetscInt>(dofs[component]);
        VecGetValues(f_int, 1, &dof_index, &value);
        resultant += static_cast<double>(value);
    }

    VecDestroy(&f_int);
    return resultant;
}

void write_global_xfem_newton_csv_header(std::ostream& out)
{
    out << "step,p,drift_mm,base_shear_MN,axial_reaction_MN,"
           "max_abs_steel_stress_MPa,"
           "max_host_damage,damaged_host_points,"
           "accepted_substeps,total_newton_iterations,failed_attempts,"
           "solver_profile_attempts,max_bisection_level,last_snes_reason,"
           "last_ksp_reason,accepted_by_small_residual_policy,residual_norm,"
           "solver_profile_label\n";
}

void write_global_xfem_newton_csv_row(
    std::ostream& out,
    const GlobalXFEMNewtonRow& row)
{
    out << row.step << ','
        << row.p << ','
        << row.drift_mm << ','
        << row.base_shear_mn << ','
        << row.axial_reaction_mn << ','
        << row.max_abs_steel_stress_mpa << ','
        << row.max_host_damage << ','
        << row.damaged_host_points << ','
        << row.accepted_substeps << ','
        << row.total_newton_iterations << ','
        << row.failed_attempts << ','
        << row.solver_profile_attempts << ','
        << row.max_bisection_level << ','
        << row.last_snes_reason << ','
        << row.last_ksp_reason << ','
        << (row.accepted_by_small_residual_policy ? 1 : 0) << ','
        << row.residual_norm << ','
        << json_escape(row.solver_profile_label) << '\n';
}

void write_global_xfem_newton_csv(
    const std::filesystem::path& path,
    const std::vector<GlobalXFEMNewtonRow>& rows)
{
    std::ofstream out(path);
    out << std::setprecision(10);
    write_global_xfem_newton_csv_header(out);
    for (const auto& row : rows) {
        write_global_xfem_newton_csv_row(out, row);
    }
}

void write_global_xfem_newton_manifest(
    const std::filesystem::path& path,
    const Options& options,
    const ReducedRCColumnReferenceSpec& spec,
    const GlobalXFEMNewtonSummary& summary,
    const fall_n::xfem::BilinearCohesiveLawParameters& cohesive)
{
    const auto scale_audit = make_global_xfem_scale_audit(options, spec);
    std::ofstream out(path);
    out << std::setprecision(12)
        << "{\n"
        << "  \"tool\": \"fall_n\",\n"
        << "  \"driver_kind\": \"global_shifted_heaviside_xfem_newton_column_trial\",\n"
        << "  \"attempted\": " << (summary.attempted ? "true" : "false") << ",\n"
        << "  \"completed_successfully\": "
        << (summary.completed ? "true" : "false") << ",\n"
        << "  \"status\": \"" << json_escape(summary.status) << "\",\n"
        << "  \"failure_reason\": \""
        << json_escape(summary.failure_reason) << "\",\n"
        << "  \"mesh\": {\n"
        << "    \"hex_order\": \"Hex8\",\n"
        << "    \"nx\": " << options.global_xfem_nx << ",\n"
        << "    \"ny\": " << options.global_xfem_ny << ",\n"
        << "    \"nz\": " << options.global_xfem_nz << ",\n"
        << "    \"longitudinal_bias_power\": "
        << options.global_xfem_bias_power << ",\n"
        << "    \"longitudinal_bias_location\": \""
        << json_escape(normalized_material_token(
               options.global_xfem_bias_location))
        << "\",\n"
        << "    \"element_count\": " << summary.element_count << ",\n"
        << "    \"node_count\": " << summary.node_count << ",\n"
        << "    \"enriched_node_count\": "
        << summary.enriched_node_count << "\n"
        << "  },\n"
        << "  \"dofs\": {\n"
        << "    \"local_state_dofs\": " << summary.local_state_dofs << ",\n"
        << "    \"solver_global_dofs\": " << summary.solver_global_dofs << "\n"
        << "  },\n"
        << "  \"scale_audit\": {\n"
        << "    \"estimated_total_state_dofs\": "
        << scale_audit.estimated_total_state_dofs << ",\n"
        << "    \"estimated_sparse_nonzeros\": "
        << scale_audit.estimated_sparse_nonzeros << ",\n"
        << "    \"host_material_point_count\": "
        << scale_audit.host_material_point_count << ",\n"
        << "    \"sparse_matrix_mib\": "
        << scale_audit.sparse_matrix_mib << ",\n"
        << "    \"direct_factorization_risk_mib\": "
        << scale_audit.direct_factorization_risk_mib << ",\n"
        << "    \"material_state_mib\": "
        << scale_audit.material_state_mib << ",\n"
        << "    \"solver_advice\": \""
        << fall_n::to_string(scale_audit.solver_advice) << "\",\n"
        << "    \"seed_state_cache_recommended\": "
        << (scale_audit.seed_state_cache_recommended ? "true" : "false")
        << ",\n"
        << "    \"newton_warm_start_recommended\": "
        << (scale_audit.newton_warm_start_recommended ? "true" : "false")
        << ",\n"
        << "    \"site_level_openmp_recommended\": "
        << (scale_audit.site_level_openmp_recommended ? "true" : "false")
        << ",\n"
        << "    \"global_petsc_assembly_openmp_recommended\": "
        << (scale_audit.global_petsc_assembly_openmp_recommended ? "true"
                                                                 : "false")
        << ",\n"
        << "    \"symmetric_matrix_storage_recommended\": "
        << (scale_audit.symmetric_matrix_storage_recommended ? "true"
                                                            : "false")
        << ",\n"
        << "    \"symmetric_matrix_storage_requires_tangent_audit\": "
        << (scale_audit.symmetric_matrix_storage_requires_tangent_audit
                ? "true"
                : "false")
        << ",\n"
        << "    \"block_matrix_storage_candidate\": "
        << (scale_audit.block_matrix_storage_candidate ? "true" : "false")
        << ",\n"
        << "    \"field_split_or_asm_preconditioner_recommended\": "
        << (scale_audit.field_split_or_asm_preconditioner_recommended ? "true"
                                                                      : "false")
        << ",\n"
        << "    \"plain_gmres_ilu_rejected_for_enriched_branch\": "
        << (scale_audit.plain_gmres_ilu_rejected_for_enriched_branch ? "true"
                                                                     : "false")
        << "\n"
        << "  },\n"
        << "  \"reinforcement\": {\n"
        << "    \"bar_count\": " << summary.rebar_bar_count << ",\n"
        << "    \"truss_element_count\": " << summary.rebar_element_count << ",\n"
        << "    \"xfem_enriched_penalty_coupling_count\": "
        << summary.rebar_coupling_count << ",\n"
        << "    \"crack_crossing_rebar_element_count\": "
        << summary.crack_crossing_rebar_element_count << ",\n"
        << "    \"crack_crossing_rebar_area_scale\": "
        << options.global_xfem_crack_crossing_rebar_area_scale << ",\n"
        << "    \"crack_crossing_gauge_length_mm\": "
        << options.global_xfem_crack_crossing_gauge_length_mm << ",\n"
        << "    \"crack_crossing_rebar_mode\": \""
        << json_escape(normalized_material_token(
               options.global_xfem_crack_crossing_rebar_mode))
        << "\",\n"
        << "    \"crack_crossing_bridge_law\": \""
        << json_escape(normalized_material_token(
               options.global_xfem_crack_crossing_bridge_law))
        << "\",\n"
        << "    \"crack_crossing_axis_frame\": \""
        << json_escape(normalized_material_token(
               options.global_xfem_crack_crossing_axis_frame))
        << "\",\n"
        << "    \"crack_crossing_host_axis_tangent\": \""
        << json_escape(normalized_material_token(
               options.global_xfem_crack_crossing_host_axis_tangent))
        << "\",\n"
        << "    \"crack_crossing_yield_slip_mm\": "
        << options.global_xfem_crack_crossing_yield_slip_mm << ",\n"
        << "    \"crack_crossing_yield_force_mn\": "
        << json_number_or_null(
               options.global_xfem_crack_crossing_yield_force_mn)
        << ",\n"
        << "    \"crack_crossing_hardening_ratio\": "
        << options.global_xfem_crack_crossing_hardening_ratio << ",\n"
        << "    \"crack_crossing_force_cap_mn\": "
        << json_number_or_null(
               options.global_xfem_crack_crossing_force_cap_mn)
        << ",\n"
        << "    \"total_area_m2\": " << summary.total_rebar_area_m2 << ",\n"
        << "    \"coupling_alpha_scale_over_ec\": "
        << options.global_xfem_rebar_coupling_alpha_scale_over_ec << "\n"
        << "  },\n"
        << "  \"physics\": {\n"
        << "    \"host_material\": \""
        << json_escape(normalized_material_token(
               options.global_xfem_concrete_material))
        << "\",\n"
        << "    \"residual_tension_fraction\": "
        << options.global_xfem_residual_tension_fraction << ",\n"
        << "    \"global_xfem_shear_cap_mpa\": "
        << options.global_xfem_shear_cap_mpa << ",\n"
        << "    \"crack_band_tangent\": \""
        << json_escape(options.global_xfem_crack_band_tangent) << "\",\n"
        << "    \"fracture_representation\": \"shifted_heaviside_planar_base_crack_with_cohesive_surface\",\n"
        << "    \"requested_kinematic_formulation\": \""
        << json_escape(summary.requested_kinematic_formulation) << "\",\n"
        << "    \"effective_kinematic_formulation\": \""
        << json_escape(summary.effective_kinematic_formulation) << "\",\n"
        << "    \"guarded_finite_kinematics_allowed\": "
        << (options.allow_guarded_xfem_finite_kinematics ? "true" : "false")
        << ",\n"
        << "    \"xfem_kinematic_support_status\": \""
        << json_escape(summary.kinematic_support_status) << "\",\n"
        << "    \"large_amplitude_kinematic_recommendation\": \""
        << json_escape(summary.large_amplitude_kinematic_recommendation)
        << "\",\n"
        << "    \"reinforcement_representation\": \"menegotto_pinto_truss_bars_with_shifted_heaviside_host_penalty_coupling_and_crack_crossing_rebar_bridge\",\n"
        << "    \"cohesive_jump_unit\": \"m\",\n"
        << "    \"cohesive_normal_stiffness_mpa_per_m\": "
        << cohesive.normal_stiffness << ",\n"
        << "    \"cohesive_shear_stiffness_mpa_per_m\": "
        << cohesive.shear_stiffness << ",\n"
        << "    \"cohesive_fracture_energy_mn_per_m\": "
        << cohesive.fracture_energy << ",\n"
        << "    \"cohesive_shear_traction_cap_mpa\": "
        << cohesive.shear_traction_cap << ",\n"
        << "    \"cohesive_tangent\": \""
        << json_escape(normalized_material_token(
               options.global_xfem_cohesive_tangent))
        << "\",\n"
        << "    \"cohesive_surface_tangent\": \""
        << json_escape(std::string(fall_n::xfem::to_string(
               parse_global_xfem_cohesive_surface_tangent_mode(
                   options.global_xfem_cohesive_surface_tangent))))
        << "\",\n"
        << "    \"cohesive_traction_measure\": \""
        << json_escape(std::string(fall_n::xfem::to_string(
               parse_global_xfem_cohesive_traction_measure(
                   options.global_xfem_cohesive_traction_measure))))
        << "\",\n"
        << "    \"crack_z_m\": " << summary.crack_z_m << ",\n"
        << "    \"crack_z_source\": \""
        << json_escape(summary.crack_z_source) << "\",\n"
        << "    \"axial_compression_mn\": "
        << options.axial_compression_mn << ",\n"
        << "    \"concrete_fpc_mpa\": " << spec.concrete_fpc_mpa << "\n"
        << "  },\n"
        << "  \"protocol\": {\n"
        << "    \"amplitudes_mm\": \"" << options.amplitudes_mm << "\",\n"
        << "    \"steps_per_segment\": " << options.steps_per_segment << ",\n"
        << "    \"point_count\": " << summary.point_count << ",\n"
        << "    \"peak_abs_drift_mm\": "
        << summary.peak_abs_drift_mm << "\n"
        << "  },\n"
        << "  \"solve_control\": {\n"
        << "    \"continuation_kind\": \""
        << json_escape(summary.continuation_kind) << "\",\n"
        << "    \"max_bisections\": "
        << options.global_xfem_max_bisections << ",\n"
        << "    \"solver_max_iterations\": "
        << options.global_xfem_solver_max_iterations << ",\n"
        << "    \"solver_profile\": \""
        << json_escape(normalized_material_token(
               options.global_xfem_solver_profile))
        << "\",\n"
        << "    \"solver_cascade\": "
        << (options.global_xfem_solver_cascade ? "true" : "false") << ",\n"
        << "    \"adaptive_increments\": "
        << (options.global_xfem_adaptive_increments ? "true" : "false")
        << ",\n"
        << "    \"mixed_arc_length_status\": \""
        << json_escape(summary.mixed_control_status) << "\",\n"
        << "    \"mixed_arc_length_target\": "
        << options.global_xfem_mixed_arc_target << ",\n"
        << "    \"mixed_arc_length_reject_factor\": "
        << options.global_xfem_mixed_arc_reject_factor << ",\n"
        << "    \"mixed_arc_length_reaction_scale_mn\": "
        << options.global_xfem_mixed_arc_reaction_scale_mn << ",\n"
        << "    \"mixed_arc_length_damage_weight\": "
        << options.global_xfem_mixed_arc_damage_weight << ",\n"
        << "    \"bordered_hybrid_disable_streak\": "
        << options.global_xfem_bordered_hybrid_disable_streak << ",\n"
        << "    \"bordered_hybrid_retry_interval\": "
        << options.global_xfem_bordered_hybrid_retry_interval << ",\n"
        << "    \"mixed_control_accepted_steps\": "
        << summary.mixed_control_accepted_steps << ",\n"
        << "    \"mixed_control_failed_solver_attempts\": "
        << summary.mixed_control_failed_solver_attempts << ",\n"
        << "    \"mixed_control_rejected_arc_attempts\": "
        << summary.mixed_control_rejected_arc_attempts << ",\n"
        << "    \"mixed_control_total_cutbacks\": "
        << summary.mixed_control_total_cutbacks << ",\n"
        << "    \"bordered_hybrid_attempted_steps\": "
        << summary.bordered_hybrid_attempted_steps << ",\n"
        << "    \"bordered_hybrid_successful_steps\": "
        << summary.bordered_hybrid_successful_steps << ",\n"
        << "    \"bordered_hybrid_skipped_steps\": "
        << summary.bordered_hybrid_skipped_steps << ",\n"
        << "    \"mixed_control_max_arc_length\": "
        << summary.mixed_control_max_arc_length << ",\n"
        << "    \"mixed_control_mean_arc_length\": "
        << summary.mixed_control_mean_arc_length << ",\n"
        << "    \"total_accepted_substeps\": "
        << summary.total_accepted_substeps << ",\n"
        << "    \"total_nonlinear_iterations\": "
        << summary.total_nonlinear_iterations << ",\n"
        << "    \"total_failed_attempts\": "
        << summary.total_failed_attempts << ",\n"
        << "    \"total_solver_profile_attempts\": "
        << summary.total_solver_profile_attempts << ",\n"
        << "    \"max_requested_step_nonlinear_iterations\": "
        << summary.max_requested_step_nonlinear_iterations << ",\n"
        << "    \"max_bisection_level\": "
        << summary.max_bisection_level << ",\n"
        << "    \"hard_step_count\": "
        << summary.hard_step_count << "\n"
        << "  },\n"
        << "  \"observables\": {\n"
        << "    \"peak_abs_base_shear_mn\": "
        << summary.peak_abs_base_shear_mn << ",\n"
        << "    \"peak_abs_steel_stress_mpa\": "
        << summary.peak_abs_steel_stress_mpa << ",\n"
        << "    \"max_host_damage\": " << summary.max_host_damage << ",\n"
        << "    \"max_damaged_host_points\": "
        << summary.max_damaged_host_points << "\n"
        << "  },\n"
        << "  \"timing\": {\n"
        << "    \"total_wall_seconds\": "
        << summary.elapsed_seconds << "\n"
        << "  },\n"
        << "  \"artifacts\": {\n"
        << "    \"hysteresis_csv\": \"global_xfem_newton_hysteresis.csv\",\n"
        << "    \"progress_csv\": \"global_xfem_newton_progress.csv\",\n"
        << "    \"mixed_control_arc_length_csv\": "
           "\"global_xfem_mixed_control_arc_length.csv\"\n"
        << "  }\n"
        << "}\n";
}

void write_global_xfem_mixed_control_arc_csv(
    const std::filesystem::path& path,
    const fall_n::MixedControlArcLengthResult& result)
{
    std::ofstream out(path);
    out << std::setprecision(12)
        << "record,step,p_start,p_target,p_accepted,increment,arc_length,"
           "normalized_control_increment,normalized_reaction_increment,"
           "normalized_internal_increment,cutbacks_before_acceptance,"
           "accepted,rejected_by_arc_length\n";
    int record_index = 0;
    for (const auto& row : result.records) {
        out << record_index++ << ','
            << row.step << ','
            << row.p_start << ','
            << row.p_target << ','
            << row.p_accepted << ','
            << row.increment << ','
            << row.arc_length << ','
            << row.normalized_control_increment << ','
            << row.normalized_reaction_increment << ','
            << row.normalized_internal_increment << ','
            << row.cutbacks_before_acceptance << ','
            << (row.accepted ? 1 : 0) << ','
            << (row.rejected_by_arc_length ? 1 : 0) << '\n';
    }
}

void write_reduced_rc_visualization_contract_json(
    const std::filesystem::path& path)
{
    std::ofstream out(path);
    out << "{\n"
        << "  \"schema\": \"reduced_rc_validation_visualization_contract_v1\",\n"
        << "  \"vtk_collection_policy\": {\n"
        << "    \"preferred_collection\": \"pvd_or_vtm_time_series\",\n"
        << "    \"pseudo_time_field\": \"pseudo_time\",\n"
        << "    \"physical_time_field\": \"physical_time\",\n"
        << "    \"paraview_goal\": \"animate structural/global and XFEM local fields on the same replay timeline\"\n"
        << "  },\n"
        << "  \"fields\": [\n";

    const auto fields = fall_n::canonical_reduced_rc_vtk_field_table_v;
    for (std::size_t i = 0; i < fields.size(); ++i) {
        const auto& field = fields[i];
        out << "    {\n"
            << "      \"name\": \"" << json_escape(std::string(field.name))
            << "\",\n"
            << "      \"scale\": \""
            << json_escape(std::string(fall_n::to_string(field.scale_kind)))
            << "\",\n"
            << "      \"location\": \""
            << json_escape(std::string(fall_n::to_string(field.location_kind)))
            << "\",\n"
            << "      \"components\": " << field.components << ",\n"
            << "      \"required_for_pseudo_time\": "
            << (field.required_for_pseudo_time ? "true" : "false")
            << ",\n"
            << "      \"required_for_physical_time\": "
            << (field.required_for_physical_time ? "true" : "false")
            << ",\n"
            << "      \"required_for_multiscale_replay\": "
            << (field.required_for_multiscale_replay ? "true" : "false")
            << ",\n"
            << "      \"interpretation\": \""
            << json_escape(std::string(field.interpretation)) << "\"\n"
            << "    }" << (i + 1 == fields.size() ? "\n" : ",\n");
    }

    out << "  ],\n"
        << "  \"multiscale_start_stages\": [\n";
    const auto stages =
        fall_n::canonical_reduced_rc_multiscale_start_stage_table_v;
    for (std::size_t i = 0; i < stages.size(); ++i) {
        const auto& stage = stages[i];
        out << "    {\n"
            << "      \"stage\": \""
            << json_escape(std::string(fall_n::to_string(stage.stage_kind)))
            << "\",\n"
            << "      \"key\": \"" << json_escape(std::string(stage.key))
            << "\",\n"
            << "      \"driver_hint\": \""
            << json_escape(std::string(stage.driver_hint)) << "\",\n"
            << "      \"prerequisite_gate\": \""
            << json_escape(std::string(stage.prerequisite_gate)) << "\",\n"
            << "      \"expected_artifact\": \""
            << json_escape(std::string(stage.expected_artifact)) << "\",\n"
            << "      \"may_run_before_two_way_fe2\": "
            << (stage.may_run_before_two_way_fe2 ? "true" : "false")
            << ",\n"
            << "      \"requires_xfem_enriched_dofs\": "
            << (stage.requires_xfem_enriched_dofs ? "true" : "false")
            << ",\n"
            << "      \"writes_vtk_time_series\": "
            << (stage.writes_vtk_time_series ? "true" : "false") << "\n"
            << "    }" << (i + 1 == stages.size() ? "\n" : ",\n");
    }
    out << "  ]\n"
        << "}\n";
}

[[nodiscard]] std::vector<fall_n::ReducedRCStructuralReplaySample>
make_reduced_rc_replay_samples_from_hinge_history(
    const std::vector<HistoryRow>& rows,
    const Options& options)
{
    std::vector<fall_n::ReducedRCStructuralReplaySample> samples;
    samples.reserve(rows.size());
    const double characteristic_length_m =
        std::max(options.characteristic_length_mm / 1000.0, 1.0e-12);
    const double denom =
        rows.size() > 1 ? static_cast<double>(rows.size() - 1) : 1.0;

    double previous_drift_mm = rows.empty() ? 0.0 : rows.front().drift_mm;
    double previous_base_shear_mn =
        rows.empty() ? 0.0 : rows.front().base_shear_mn;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const auto& row = rows[i];
        const double drift_increment_mm =
            i == 0 ? 0.0 : row.drift_mm - previous_drift_mm;
        const double work_increment =
            i == 0 ? 0.0
                   : 0.5 * (row.base_shear_mn + previous_base_shear_mn) *
                         drift_increment_mm;
        samples.push_back(fall_n::ReducedRCStructuralReplaySample{
            .site_index = 0,
            .pseudo_time = static_cast<double>(i) / denom,
            .physical_time = static_cast<double>(i) / denom,
            .z_over_l = 0.02,
            .drift_mm = row.drift_mm,
            .curvature_y = row.theta_y_rad / characteristic_length_m,
            .moment_y_mn_m = row.moment_mn_m,
            .base_shear_mn = row.base_shear_mn,
            .steel_stress_mpa = row.max_abs_steel_stress_mpa,
            .damage_indicator =
                std::max(row.max_damage, row.cracked_area_fraction),
            .work_increment_mn_mm = work_increment});
        previous_drift_mm = row.drift_mm;
        previous_base_shear_mn = row.base_shear_mn;
    }
    return samples;
}

[[nodiscard]] fall_n::ReducedRCMultiscaleReplayPlan
make_reduced_rc_multiscale_replay_plan_from_hinge_history(
    const std::vector<HistoryRow>& rows,
    const Options& options,
    const ReducedRCColumnReferenceSpec& spec)
{
    fall_n::ReducedRCMultiscaleReplayPlanSettings settings;
    settings.max_replay_sites = 3;
    settings.local_mesh = make_global_xfem_local_mesh_scale_input(options, spec);
    settings.curvature_activation_threshold = 0.010;
    settings.moment_activation_threshold_mn_m = 0.015;
    settings.steel_activation_threshold_mpa = 0.70 * spec.steel_fy_mpa;
    settings.damage_activation_threshold = 0.20;
    settings.work_activation_threshold_mn_mm = 2.0;
    settings.guarded_two_way_score_threshold = 2.50;
    const auto samples =
        make_reduced_rc_replay_samples_from_hinge_history(rows, options);
    return fall_n::make_reduced_rc_multiscale_replay_plan(samples, settings);
}

void write_reduced_rc_multiscale_replay_plan_csv(
    const std::filesystem::path& path,
    const fall_n::ReducedRCMultiscaleReplayPlan& plan)
{
    std::ofstream out(path);
    out << std::setprecision(12)
        << "rank,site_index,z_over_l,sample_count,selected,activation_kind,"
           "activation_score,peak_abs_curvature_y,peak_abs_moment_y_mn_m,"
           "peak_abs_base_shear_mn,peak_abs_steel_stress_mpa,"
           "max_damage_indicator,accumulated_abs_work_mn_mm,"
           "estimated_hot_state_mib,direct_factorization_risk_mib,"
           "solver_advice,selection_reason\n";
    for (std::size_t i = 0; i < plan.sites.size(); ++i) {
        const auto& site = plan.sites[i];
        out << i << ','
            << site.site_index << ','
            << site.z_over_l << ','
            << site.sample_count << ','
            << (site.selected_for_replay ? 1 : 0) << ','
            << fall_n::to_string(site.activation_kind) << ','
            << site.activation_score << ','
            << site.peak_abs_curvature_y << ','
            << site.peak_abs_moment_y_mn_m << ','
            << site.peak_abs_base_shear_mn << ','
            << site.peak_abs_steel_stress_mpa << ','
            << site.max_damage_indicator << ','
            << site.accumulated_abs_work_mn_mm << ','
            << site.local_cost.estimated_hot_state_mib << ','
            << site.local_cost.direct_factorization_risk_mib << ','
            << fall_n::to_string(site.local_cost.solver_advice) << ','
            << '"' << json_escape(std::string(site.selection_reason)) << '"'
            << '\n';
    }
}

void write_reduced_rc_multiscale_replay_plan_json(
    const std::filesystem::path& path,
    const fall_n::ReducedRCMultiscaleReplayPlan& plan)
{
    std::ofstream out(path);
    out << std::setprecision(12)
        << "{\n"
        << "  \"schema\": \"reduced_rc_multiscale_one_way_replay_plan_v1\",\n"
        << "  \"scientific_status\": \"one_way_replay_before_two_way_fe2\",\n"
        << "  \"history_sample_count\": " << plan.history_sample_count << ",\n"
        << "  \"candidate_site_count\": " << plan.candidate_site_count << ",\n"
        << "  \"selected_site_count\": " << plan.selected_site_count << ",\n"
        << "  \"ready_for_one_way_replay\": "
        << (plan.ready_for_one_way_replay ? "true" : "false") << ",\n"
        << "  \"ready_for_two_way_fe2\": "
        << (plan.ready_for_two_way_fe2 ? "true" : "false") << ",\n"
        << "  \"vtk_contract_satisfied\": "
        << (plan.vtk_contract_satisfied ? "true" : "false") << ",\n"
        << "  \"execution_policy\": {\n"
        << "    \"seed_state_cache_recommended\": "
        << (plan.seed_state_cache_recommended ? "true" : "false") << ",\n"
        << "    \"newton_warm_start_recommended\": "
        << (plan.newton_warm_start_recommended ? "true" : "false") << ",\n"
        << "    \"site_level_openmp_recommended\": "
        << (plan.site_level_openmp_recommended ? "true" : "false") << ",\n"
        << "    \"avoid_direct_lu_for_batch\": "
        << (plan.avoid_direct_lu_for_batch ? "true" : "false") << ",\n"
        << "    \"selected_hot_state_mib\": "
        << plan.selected_hot_state_mib << ",\n"
        << "    \"selected_direct_factorization_risk_mib\": "
        << plan.selected_direct_factorization_risk_mib << "\n"
        << "  },\n"
        << "  \"sites\": [\n";
    for (std::size_t i = 0; i < plan.sites.size(); ++i) {
        const auto& site = plan.sites[i];
        out << "    {\n"
            << "      \"rank\": " << i << ",\n"
            << "      \"site_index\": " << site.site_index << ",\n"
            << "      \"z_over_l\": " << site.z_over_l << ",\n"
            << "      \"sample_count\": " << site.sample_count << ",\n"
            << "      \"selected_for_replay\": "
            << (site.selected_for_replay ? "true" : "false") << ",\n"
            << "      \"activation_kind\": \""
            << fall_n::to_string(site.activation_kind) << "\",\n"
            << "      \"selection_reason\": \""
            << json_escape(std::string(site.selection_reason)) << "\",\n"
            << "      \"activation_score\": "
            << site.activation_score << ",\n"
            << "      \"demand\": {\n"
            << "        \"peak_abs_curvature_y\": "
            << site.peak_abs_curvature_y << ",\n"
            << "        \"peak_abs_moment_y_mn_m\": "
            << site.peak_abs_moment_y_mn_m << ",\n"
            << "        \"peak_abs_base_shear_mn\": "
            << site.peak_abs_base_shear_mn << ",\n"
            << "        \"peak_abs_steel_stress_mpa\": "
            << site.peak_abs_steel_stress_mpa << ",\n"
            << "        \"max_damage_indicator\": "
            << site.max_damage_indicator << ",\n"
            << "        \"accumulated_abs_work_mn_mm\": "
            << site.accumulated_abs_work_mn_mm << "\n"
            << "      },\n"
            << "      \"local_cost\": {\n"
            << "        \"estimated_total_state_dofs\": "
            << site.local_cost.estimated_total_state_dofs << ",\n"
            << "        \"host_material_point_count\": "
            << site.local_cost.host_material_point_count << ",\n"
            << "        \"estimated_hot_state_mib\": "
            << site.local_cost.estimated_hot_state_mib << ",\n"
            << "        \"direct_factorization_risk_mib\": "
            << site.local_cost.direct_factorization_risk_mib << ",\n"
            << "        \"solver_advice\": \""
            << fall_n::to_string(site.local_cost.solver_advice) << "\"\n"
            << "      }\n"
            << "    }" << (i + 1 == plan.sites.size() ? "\n" : ",\n");
    }
    out << "  ],\n"
        << "  \"artifacts\": {\n"
        << "    \"site_catalog_csv\": \"multiscale_replay_site_catalog.csv\",\n"
        << "    \"visualization_contract_json\": \"visualization_contract.json\"\n"
        << "  }\n"
        << "}\n";
}

void write_reduced_rc_multiscale_runtime_policy_json(
    const std::filesystem::path& path,
    const fall_n::ReducedRCMultiscaleRuntimePolicy& policy)
{
    const auto& runtime = policy.local_runtime_settings;
    std::ofstream out(path);
    out << std::setprecision(12)
        << "{\n"
        << "  \"schema\": \"reduced_rc_multiscale_runtime_policy_v1\",\n"
        << "  \"ready_for_local_site_batch\": "
        << (policy.ready_for_local_site_batch ? "true" : "false") << ",\n"
        << "  \"executor_kind\": \""
        << fall_n::to_string(policy.executor_kind) << "\",\n"
        << "  \"recommended_site_threads\": "
        << policy.recommended_site_threads << ",\n"
        << "  \"local_sites_run_in_parallel\": "
        << (policy.local_sites_run_in_parallel ? "true" : "false") << ",\n"
        << "  \"cache_budget_is_bounded\": "
        << (policy.cache_budget_is_bounded ? "true" : "false") << ",\n"
        << "  \"direct_lu_kept_as_reference_only\": "
        << (policy.direct_lu_kept_as_reference_only ? "true" : "false")
        << ",\n"
        << "  \"iterative_preconditioner_expected\": "
        << (policy.iterative_preconditioner_expected ? "true" : "false")
        << ",\n"
        << "  \"rationale\": \""
        << json_escape(std::string(policy.rationale)) << "\",\n"
        << "  \"local_runtime_settings\": {\n"
        << "    \"profiling_enabled\": "
        << (runtime.profiling_enabled ? "true" : "false") << ",\n"
        << "    \"seed_state_reuse_enabled\": "
        << (runtime.seed_state_reuse_enabled ? "true" : "false") << ",\n"
        << "    \"restore_seed_before_solve\": "
        << (runtime.restore_seed_before_solve ? "true" : "false") << ",\n"
        << "    \"max_cached_seed_states\": "
        << runtime.max_cached_seed_states << ",\n"
        << "    \"adaptive_activation_enabled\": "
        << (runtime.adaptive_activation_enabled ? "true" : "false") << ",\n"
        << "    \"keep_active_once_triggered\": "
        << (runtime.keep_active_once_triggered ? "true" : "false") << ",\n"
        << "    \"deactivation_metric_threshold\": "
        << runtime.deactivation_metric_threshold << ",\n"
        << "    \"prefer_active_seed_retention\": "
        << (runtime.prefer_active_seed_retention ? "true" : "false")
        << "\n"
        << "  },\n"
        << "  \"artifacts\": {\n"
        << "    \"replay_plan_json\": \"multiscale_replay_plan.json\",\n"
        << "    \"site_catalog_csv\": \"multiscale_replay_site_catalog.csv\"\n"
        << "  }\n"
        << "}\n";
}

void write_reduced_rc_local_site_batch_plan_csv(
    const std::filesystem::path& path,
    const fall_n::ReducedRCLocalSiteBatchPlan& plan)
{
    std::ofstream out(path);
    out << std::setprecision(12)
        << "batch_index,slot_index,site_index,z_over_l,activation_score,"
           "estimated_hot_state_mib,direct_factorization_risk_mib,"
           "solver_kind,seed_restore_required,warm_start_required,"
           "vtk_time_series_required,rationale\n";
    for (const auto& row : plan.rows) {
        out << row.batch_index << ','
            << row.slot_index << ','
            << row.site_index << ','
            << row.z_over_l << ','
            << row.activation_score << ','
            << row.estimated_hot_state_mib << ','
            << row.direct_factorization_risk_mib << ','
            << fall_n::to_string(row.solver_kind) << ','
            << (row.seed_restore_required ? 1 : 0) << ','
            << (row.warm_start_required ? 1 : 0) << ','
            << (row.vtk_time_series_required ? 1 : 0) << ','
            << '"' << json_escape(std::string(row.rationale)) << '"'
            << '\n';
    }
}

void write_reduced_rc_local_site_batch_plan_json(
    const std::filesystem::path& path,
    const fall_n::ReducedRCLocalSiteBatchPlan& plan)
{
    std::ofstream out(path);
    out << std::setprecision(12)
        << "{\n"
        << "  \"schema\": \"reduced_rc_multiscale_local_site_batch_plan_v1\",\n"
        << "  \"ready_for_local_site_batch\": "
        << (plan.ready_for_local_site_batch ? "true" : "false") << ",\n"
        << "  \"ready_for_many_site_replay\": "
        << (plan.ready_for_many_site_replay ? "true" : "false") << ",\n"
        << "  \"executor_kind\": \"" << fall_n::to_string(plan.executor_kind)
        << "\",\n"
        << "  \"selected_site_count\": " << plan.selected_site_count << ",\n"
        << "  \"batch_count\": " << plan.batch_count << ",\n"
        << "  \"max_concurrent_sites\": " << plan.max_concurrent_sites << ",\n"
        << "  \"recommended_site_threads\": "
        << plan.recommended_site_threads << ",\n"
        << "  \"bounded_seed_cache_required\": "
        << (plan.bounded_seed_cache_required ? "true" : "false") << ",\n"
        << "  \"iterative_preconditioner_expected\": "
        << (plan.iterative_preconditioner_expected ? "true" : "false")
        << ",\n"
        << "  \"total_estimated_hot_state_mib\": "
        << plan.total_estimated_hot_state_mib << ",\n"
        << "  \"max_batch_hot_state_mib\": "
        << plan.max_batch_hot_state_mib << ",\n"
        << "  \"total_direct_factorization_risk_mib\": "
        << plan.total_direct_factorization_risk_mib << ",\n"
        << "  \"rationale\": \""
        << json_escape(std::string(plan.rationale)) << "\",\n"
        << "  \"batches\": [\n";
    for (std::size_t i = 0; i < plan.batches.size(); ++i) {
        const auto& batch = plan.batches[i];
        out << "    {\n"
            << "      \"batch_index\": " << batch.batch_index << ",\n"
            << "      \"site_count\": " << batch.site_count << ",\n"
            << "      \"estimated_hot_state_mib\": "
            << batch.estimated_hot_state_mib << ",\n"
            << "      \"direct_factorization_risk_mib\": "
            << batch.direct_factorization_risk_mib << ",\n"
            << "      \"recommended_threads\": "
            << batch.recommended_threads << ",\n"
            << "      \"dominant_solver_kind\": \""
            << fall_n::to_string(batch.dominant_solver_kind) << "\",\n"
            << "      \"uses_parallel_site_loop\": "
            << (batch.uses_parallel_site_loop ? "true" : "false") << ",\n"
            << "      \"within_hot_state_budget\": "
            << (batch.within_hot_state_budget ? "true" : "false") << ",\n"
            << "      \"direct_lu_within_budget\": "
            << (batch.direct_lu_within_budget ? "true" : "false")
            << "\n"
            << "    }" << (i + 1 == plan.batches.size() ? "\n" : ",\n");
    }
    out << "  ],\n"
        << "  \"sites\": [\n";
    for (std::size_t i = 0; i < plan.rows.size(); ++i) {
        const auto& row = plan.rows[i];
        out << "    {\n"
            << "      \"batch_index\": " << row.batch_index << ",\n"
            << "      \"slot_index\": " << row.slot_index << ",\n"
            << "      \"site_index\": " << row.site_index << ",\n"
            << "      \"z_over_l\": " << row.z_over_l << ",\n"
            << "      \"activation_score\": " << row.activation_score << ",\n"
            << "      \"estimated_hot_state_mib\": "
            << row.estimated_hot_state_mib << ",\n"
            << "      \"direct_factorization_risk_mib\": "
            << row.direct_factorization_risk_mib << ",\n"
            << "      \"solver_kind\": \""
            << fall_n::to_string(row.solver_kind) << "\",\n"
            << "      \"seed_restore_required\": "
            << (row.seed_restore_required ? "true" : "false") << ",\n"
            << "      \"warm_start_required\": "
            << (row.warm_start_required ? "true" : "false") << ",\n"
            << "      \"vtk_time_series_required\": "
            << (row.vtk_time_series_required ? "true" : "false") << ",\n"
            << "      \"rationale\": \""
            << json_escape(std::string(row.rationale)) << "\"\n"
            << "    }" << (i + 1 == plan.rows.size() ? "\n" : ",\n");
    }
    out << "  ],\n"
        << "  \"artifacts\": {\n"
        << "    \"runtime_policy_json\": \"multiscale_runtime_policy.json\",\n"
        << "    \"replay_plan_json\": \"multiscale_replay_plan.json\"\n"
        << "  }\n"
        << "}\n";
}

void write_reduced_rc_local_site_replay_result_csv(
    const std::filesystem::path& path,
    const fall_n::ReducedRCLocalSiteReplayRunResult& result)
{
    std::ofstream out(path);
    out << std::setprecision(12)
        << "batch_index,slot_index,site_index,status,input_sample_count,"
           "attempted_step_count,accepted_step_count,failed_step_count,"
           "generated_cutback_step_count,max_cutback_level,"
           "total_nonlinear_iterations,total_elapsed_seconds,"
           "accumulated_abs_work_mn_mm,peak_abs_steel_stress_mpa,"
           "max_damage_indicator,last_drift_mm,last_pseudo_time,"
           "failure_reason\n";
    for (const auto& site : result.sites) {
        out << site.batch_index << ','
            << site.slot_index << ','
            << site.site_index << ','
            << fall_n::to_string(site.status) << ','
            << site.input_sample_count << ','
            << site.attempted_step_count << ','
            << site.accepted_step_count << ','
            << site.failed_step_count << ','
            << site.generated_cutback_step_count << ','
            << site.max_cutback_level << ','
            << site.total_nonlinear_iterations << ','
            << site.total_elapsed_seconds << ','
            << site.accumulated_abs_work_mn_mm << ','
            << site.peak_abs_steel_stress_mpa << ','
            << site.max_damage_indicator << ','
            << site.last_drift_mm << ','
            << site.last_pseudo_time << ','
            << '"' << json_escape(std::string(site.failure_reason)) << '"'
            << '\n';
    }
}

void write_reduced_rc_local_site_replay_result_json(
    const std::filesystem::path& path,
    const fall_n::ReducedRCLocalSiteReplayRunResult& result,
    std::string_view local_solver_binding)
{
    std::ofstream out(path);
    out << std::setprecision(12)
        << "{\n"
        << "  \"schema\": \"reduced_rc_local_site_replay_result_v1\",\n"
        << "  \"scientific_status\": "
           "\"orchestration_smoke_not_physical_local_solve\",\n"
        << "  \"local_solver_binding\": \""
        << json_escape(std::string(local_solver_binding)) << "\",\n"
        << "  \"next_required_binding\": "
           "\"promoted_xfem_local_model_callback\",\n"
        << "  \"completed\": " << (result.completed ? "true" : "false")
        << ",\n"
        << "  \"ready_for_guarded_fe2_smoke\": "
        << (result.ready_for_guarded_fe2_smoke ? "true" : "false")
        << ",\n"
        << "  \"selected_site_count\": " << result.selected_site_count
        << ",\n"
        << "  \"completed_site_count\": " << result.completed_site_count
        << ",\n"
        << "  \"failed_site_count\": " << result.failed_site_count << ",\n"
        << "  \"batch_count\": " << result.batch_count << ",\n"
        << "  \"attempted_step_count\": " << result.attempted_step_count
        << ",\n"
        << "  \"accepted_step_count\": " << result.accepted_step_count
        << ",\n"
        << "  \"failed_step_count\": " << result.failed_step_count
        << ",\n"
        << "  \"generated_cutback_step_count\": "
        << result.generated_cutback_step_count << ",\n"
        << "  \"total_nonlinear_iterations\": "
        << result.total_nonlinear_iterations << ",\n"
        << "  \"total_elapsed_seconds\": "
        << result.total_elapsed_seconds << ",\n"
        << "  \"max_site_elapsed_seconds\": "
        << result.max_site_elapsed_seconds << ",\n"
        << "  \"accumulated_abs_work_mn_mm\": "
        << result.accumulated_abs_work_mn_mm << ",\n"
        << "  \"peak_abs_steel_stress_mpa\": "
        << result.peak_abs_steel_stress_mpa << ",\n"
        << "  \"max_damage_indicator\": "
        << result.max_damage_indicator << ",\n"
        << "  \"batches\": [\n";
    for (std::size_t i = 0; i < result.batches.size(); ++i) {
        const auto& batch = result.batches[i];
        out << "    {\n"
            << "      \"batch_index\": " << batch.batch_index << ",\n"
            << "      \"site_count\": " << batch.site_count << ",\n"
            << "      \"completed_site_count\": "
            << batch.completed_site_count << ",\n"
            << "      \"failed_site_count\": " << batch.failed_site_count
            << ",\n"
            << "      \"attempted_step_count\": "
            << batch.attempted_step_count << ",\n"
            << "      \"accepted_step_count\": "
            << batch.accepted_step_count << ",\n"
            << "      \"total_nonlinear_iterations\": "
            << batch.total_nonlinear_iterations << ",\n"
            << "      \"total_elapsed_seconds\": "
            << batch.total_elapsed_seconds << ",\n"
            << "      \"max_site_elapsed_seconds\": "
            << batch.max_site_elapsed_seconds << "\n"
            << "    }"
            << (i + 1 == result.batches.size() ? "\n" : ",\n");
    }
    out << "  ],\n"
        << "  \"sites\": [\n";
    for (std::size_t i = 0; i < result.sites.size(); ++i) {
        const auto& site = result.sites[i];
        out << "    {\n"
            << "      \"batch_index\": " << site.batch_index << ",\n"
            << "      \"slot_index\": " << site.slot_index << ",\n"
            << "      \"site_index\": " << site.site_index << ",\n"
            << "      \"status\": \"" << fall_n::to_string(site.status)
            << "\",\n"
            << "      \"input_sample_count\": " << site.input_sample_count
            << ",\n"
            << "      \"attempted_step_count\": "
            << site.attempted_step_count << ",\n"
            << "      \"accepted_step_count\": "
            << site.accepted_step_count << ",\n"
            << "      \"failed_step_count\": " << site.failed_step_count
            << ",\n"
            << "      \"generated_cutback_step_count\": "
            << site.generated_cutback_step_count << ",\n"
            << "      \"max_cutback_level\": " << site.max_cutback_level
            << ",\n"
            << "      \"total_nonlinear_iterations\": "
            << site.total_nonlinear_iterations << ",\n"
            << "      \"total_elapsed_seconds\": "
            << site.total_elapsed_seconds << ",\n"
            << "      \"accumulated_abs_work_mn_mm\": "
            << site.accumulated_abs_work_mn_mm << ",\n"
            << "      \"peak_abs_steel_stress_mpa\": "
            << site.peak_abs_steel_stress_mpa << ",\n"
            << "      \"max_damage_indicator\": "
            << site.max_damage_indicator << ",\n"
            << "      \"last_drift_mm\": " << site.last_drift_mm << ",\n"
            << "      \"last_pseudo_time\": " << site.last_pseudo_time
            << ",\n"
            << "      \"failure_reason\": \""
            << json_escape(std::string(site.failure_reason)) << "\"\n"
            << "    }" << (i + 1 == result.sites.size() ? "\n" : ",\n");
    }
    out << "  ]\n"
        << "}\n";
}

void write_reduced_rc_multiscale_readiness_gate_json(
    const std::filesystem::path& path,
    const fall_n::ReducedRCMultiscaleReadinessGate& gate)
{
    std::ofstream out(path);
    out << "{\n"
        << "  \"schema\": \"reduced_rc_multiscale_readiness_gate_v1\",\n"
        << "  \"ready_for_one_way_local_replay\": "
        << (gate.ready_for_one_way_local_replay ? "true" : "false")
        << ",\n"
        << "  \"ready_for_elastic_fe2_smoke\": "
        << (gate.ready_for_elastic_fe2_smoke ? "true" : "false")
        << ",\n"
        << "  \"ready_for_guarded_enriched_fe2_smoke\": "
        << (gate.ready_for_guarded_enriched_fe2_smoke ? "true" : "false")
        << ",\n"
        << "  \"physical_local_solver_bound\": "
        << (gate.physical_local_solver_bound ? "true" : "false") << ",\n"
        << "  \"local_solver_label\": \""
        << json_escape(std::string(gate.local_solver_label)) << "\",\n"
        << "  \"next_stage\": \"" << fall_n::to_string(gate.next_stage)
        << "\",\n"
        << "  \"blocking_reason\": \""
        << json_escape(std::string(gate.blocking_reason)) << "\"\n"
        << "}\n";
}

template <typename GlobalXFEMKinematicPolicy>
    requires fall_n::xfem::ShiftedHeavisideKinematicPolicy<
        GlobalXFEMKinematicPolicy>
[[nodiscard]] GlobalXFEMNewtonSummary run_global_xfem_newton_column_trial_impl(
    const Options& options,
    const ReducedRCColumnReferenceSpec& spec,
    const std::vector<double>& protocol,
    const fall_n::xfem::BilinearCohesiveLawParameters& cohesive)
{
    using fall_n::HexOrder;
    using fall_n::LongitudinalBiasLocation;
    using fall_n::PrismFace;
    using fall_n::PrismaticSpec;
    using fall_n::RebarLineInterpolation;
    using fall_n::make_reinforced_prismatic_domain;
    using fall_n::xfem::PlaneCrackLevelSet;

    GlobalXFEMNewtonSummary summary{
        .attempted = options.run_global_xfem_newton,
        .status = options.run_global_xfem_newton ? "attempted" : "skipped"};
    summary.requested_kinematic_formulation =
        normalized_xfem_kinematic_formulation(
            options.global_xfem_kinematic_formulation);
    summary.effective_kinematic_formulation =
        summary.requested_kinematic_formulation;
    summary.kinematic_support_status = global_xfem_kinematic_support_status(
        summary.effective_kinematic_formulation);
    summary.large_amplitude_kinematic_recommendation =
        global_xfem_large_amplitude_recommendation(
            summary.effective_kinematic_formulation);
    summary.continuation_kind =
        normalized_material_token(options.global_xfem_continuation);
    std::vector<GlobalXFEMNewtonRow> rows;
    fall_n::MixedControlArcLengthResult mixed_control_result{};

    const auto tic = std::chrono::steady_clock::now();
    try {
        if (!options.run_global_xfem_newton) {
            write_global_xfem_newton_csv(
                options.output_dir / "global_xfem_newton_hysteresis.csv",
                rows);
            write_global_xfem_mixed_control_arc_csv(
                options.output_dir /
                    "global_xfem_mixed_control_arc_length.csv",
                mixed_control_result);
            write_global_xfem_newton_manifest(
                options.output_dir / "global_xfem_newton_manifest.json",
                options,
                spec,
                summary,
                cohesive);
            return summary;
        }

        const auto rebar = make_global_xfem_rebar_spec(spec);
        summary.rebar_bar_count = static_cast<int>(rebar.bars.size());
        summary.total_rebar_area_m2 = total_rebar_area(rebar);

        auto reinforced = make_reinforced_prismatic_domain(PrismaticSpec{
            .width = spec.section_b_m,
            .height = spec.section_h_m,
            .length = spec.column_height_m,
            .nx = std::max(options.global_xfem_nx, 1),
            .ny = std::max(options.global_xfem_ny, 1),
            .nz = std::max(options.global_xfem_nz, 2),
            .hex_order = HexOrder::Linear,
            .longitudinal_bias_power =
                std::max(options.global_xfem_bias_power, 1.0e-6),
            .longitudinal_bias_location =
                parse_longitudinal_bias_location(
                    options.global_xfem_bias_location),
            .physical_group = "GlobalXFEMConcrete"},
            rebar,
            RebarLineInterpolation::two_node_linear);
        auto& domain = reinforced.domain;
        auto& grid = reinforced.grid;

        const double crack_z =
            choose_global_xfem_crack_z(options, grid, summary);
        summary.crack_z_m = crack_z;
        const PlaneCrackLevelSet base_crack{
            Eigen::Vector3d{0.0, 0.0, crack_z},
            Eigen::Vector3d::UnitZ()};

        const double ec_mpa = 4700.0 * std::sqrt(spec.concrete_fpc_mpa);
        Material<ThreeDimensionalMaterial> material =
            make_global_xfem_concrete_material(options, spec);

        using XFEMElement =
            fall_n::xfem::ShiftedHeavisideSolidElement<
                ThreeDimensionalMaterial,
                GlobalXFEMKinematicPolicy>;
        fall_n::xfem::ShiftedHeavisideSolidOptions xfem_solid_options;
        xfem_solid_options.cohesive_surface_tangent_mode =
            parse_global_xfem_cohesive_surface_tangent_mode(
                options.global_xfem_cohesive_surface_tangent);
        xfem_solid_options.cohesive_traction_measure_kind =
            parse_global_xfem_cohesive_traction_measure(
                options.global_xfem_cohesive_traction_measure);
        Material<UniaxialMaterial> rebar_material{
            InelasticMaterial<MenegottoPintoSteel>{
                MenegottoPintoSteel{
                    spec.steel_E_mpa,
                    spec.steel_fy_mpa,
                    spec.steel_b}},
            InelasticUpdate{}};

        const auto crack_crossing_rebar_sites =
            make_crack_crossing_rebar_sites(
                reinforced,
                rebar,
                crack_z,
                std::max(
                    options.global_xfem_crack_crossing_rebar_area_scale,
                    0.0));
        const auto crack_crossing_rebar_axes =
            crack_crossing_rebar_bridge_axes(
                options.global_xfem_crack_crossing_rebar_mode);
        const bool use_bounded_crack_crossing_bridge =
            crack_crossing_bridge_uses_bounded_slip(
                options.global_xfem_crack_crossing_bridge_law);
        const auto bounded_crack_crossing_bridge_parameters =
            make_bounded_slip_bridge_parameters(
                options,
                spec,
                crack_crossing_rebar_sites.size() *
                    crack_crossing_rebar_axes.size(),
                std::max(
                    options.global_xfem_crack_crossing_rebar_area_scale,
                    0.0));
        const fall_n::xfem::ShiftedHeavisideCrackCrossingRebarElement::Options
            crack_crossing_bridge_options{
                .axis_frame_kind = parse_crack_crossing_axis_frame(
                    options.global_xfem_crack_crossing_axis_frame),
                .include_corotational_host_axis_tangent =
                    parse_crack_crossing_host_axis_tangent(
                        options
                            .global_xfem_crack_crossing_host_axis_tangent)};
        const bool global_xfem_has_history_material =
            !rebar.bars.empty() ||
            (!crack_crossing_rebar_sites.empty() &&
             !crack_crossing_rebar_axes.empty()) ||
            global_xfem_concrete_material_is_nonlinear(options);
        const double crack_crossing_gauge_length_m =
            std::max(
                options.global_xfem_crack_crossing_gauge_length_mm,
                1.0e-9) / 1000.0;

        std::vector<FEM_Element> elements;
        elements.reserve(
            domain.num_elements() +
            crack_crossing_rebar_sites.size() *
                std::max<std::size_t>(
                    crack_crossing_rebar_axes.size(),
                    1));
        for (std::size_t element_index = 0;
             element_index < reinforced.rebar_range.first;
             ++element_index) {
            elements.emplace_back(
                XFEMElement{
                    &domain.element(element_index),
                    material,
                    base_crack,
                    cohesive,
                    xfem_solid_options});
        }
        for (std::size_t element_index = reinforced.rebar_range.first;
             element_index < reinforced.rebar_range.last;
             ++element_index) {
            const auto bar_index =
                (element_index - reinforced.rebar_range.first) /
                static_cast<std::size_t>(grid.nz);
            elements.emplace_back(
                TrussElement<3, 2>{
                    &domain.element(element_index),
                    rebar_material,
                    rebar.bars.at(bar_index).area});
        }
        summary.rebar_element_count = static_cast<int>(
            reinforced.rebar_range.last - reinforced.rebar_range.first);
        const std::size_t crack_crossing_rebar_first = elements.size();
        for (const auto& site : crack_crossing_rebar_sites) {
            for (const auto& axis : crack_crossing_rebar_axes) {
                if (use_bounded_crack_crossing_bridge) {
                    elements.emplace_back(
                        fall_n::xfem::ShiftedHeavisideCrackCrossingRebarElement{
                            &domain.element(site.host_element_index),
                            site.local_coordinates,
                            axis,
                            site.area,
                            crack_crossing_gauge_length_m,
                            bounded_crack_crossing_bridge_parameters,
                            crack_crossing_bridge_options});
                } else {
                    elements.emplace_back(
                        fall_n::xfem::ShiftedHeavisideCrackCrossingRebarElement{
                            &domain.element(site.host_element_index),
                            rebar_material,
                            site.local_coordinates,
                            axis,
                            site.area,
                            crack_crossing_gauge_length_m,
                            crack_crossing_bridge_options});
                }
            }
        }
        const std::size_t crack_crossing_rebar_last = elements.size();
        summary.crack_crossing_rebar_element_count = static_cast<int>(
            crack_crossing_rebar_last - crack_crossing_rebar_first);

        using XFEMModel = Model<
            ThreeDimensionalMaterial,
            GlobalXFEMKinematicPolicy,
            3,
            MultiElementPolicy>;
        XFEMModel model{domain, std::move(elements)};

        auto to_node_ids = [](const std::vector<PetscInt>& ids) {
            std::vector<std::size_t> out;
            out.reserve(ids.size());
            for (const auto id : ids) {
                if (id >= 0) {
                    out.push_back(static_cast<std::size_t>(id));
                }
            }
            return out;
        };
        const auto base_nodes = to_node_ids(
            grid.nodes_on_face(PrismFace::MinZ));
        const auto top_nodes = to_node_ids(
            grid.nodes_on_face(PrismFace::MaxZ));
        const auto base_rebar_nodes =
            global_xfem_rebar_end_nodes(reinforced, false);
        const auto top_rebar_nodes =
            global_xfem_rebar_end_nodes(reinforced, true);
        auto support_nodes = base_nodes;
        support_nodes.insert(
            support_nodes.end(),
            base_rebar_nodes.begin(),
            base_rebar_nodes.end());
        if (base_nodes.empty() || top_nodes.empty()) {
            throw std::runtime_error(
                "Global XFEM Newton trial generated an empty support or control face.");
        }

        for (const auto node_id : base_nodes) {
            model.fix_node(node_id);
        }
        for (const auto node_id : base_rebar_nodes) {
            model.fix_node(node_id);
        }
        for (const auto node_id : top_nodes) {
            model.constrain_dof(node_id, 0, 0.0);
            model.constrain_dof(node_id, 1, 0.0);
        }
        for (const auto node_id : top_rebar_nodes) {
            model.constrain_dof(node_id, 0, 0.0);
            model.constrain_dof(node_id, 1, 0.0);
        }

        model.setup();

        fall_n::xfem::ShiftedHeavisideRebarCoupling rebar_coupling;
        rebar_coupling.setup_longitudinal_bars(
            domain,
            grid,
            reinforced.embeddings,
            rebar.bars.size(),
            options.global_xfem_rebar_coupling_alpha_scale_over_ec *
                ec_mpa,
            base_crack,
            true);
        summary.rebar_coupling_count =
            static_cast<int>(rebar_coupling.num_couplings());

        const double concrete_stiffness_share =
            ec_mpa * std::max(
                spec.section_b_m * spec.section_h_m -
                    summary.total_rebar_area_m2,
                0.0);
        const double steel_stiffness_share =
            spec.steel_E_mpa * summary.total_rebar_area_m2;
        const double total_axial_stiffness =
            std::max(concrete_stiffness_share + steel_stiffness_share,
                     1.0e-12);
        const double concrete_axial_mn =
            options.axial_compression_mn *
            concrete_stiffness_share / total_axial_stiffness;
        const double steel_axial_mn =
            options.axial_compression_mn - concrete_axial_mn;
        const double nodal_axial_mn =
            -concrete_axial_mn / static_cast<double>(top_nodes.size());
        for (const auto node_id : top_nodes) {
            const auto dofs = model.get_domain().node(node_id).dof_index();
            if (dofs.size() > 2) {
                VecSetValueLocal(
                    model.force_vector(),
                    static_cast<PetscInt>(dofs[2]),
                    nodal_axial_mn,
                    ADD_VALUES);
            }
        }
        const double rebar_nodal_axial_mn = top_rebar_nodes.empty()
            ? 0.0
            : -steel_axial_mn / static_cast<double>(top_rebar_nodes.size());
        for (const auto node_id : top_rebar_nodes) {
            const auto dofs = model.get_domain().node(node_id).dof_index();
            if (dofs.size() > 2) {
                VecSetValueLocal(
                    model.force_vector(),
                    static_cast<PetscInt>(dofs[2]),
                    rebar_nodal_axial_mn,
                    ADD_VALUES);
            }
        }
        VecAssemblyBegin(model.force_vector());
        VecAssemblyEnd(model.force_vector());

        summary.element_count = static_cast<int>(domain.num_elements());
        summary.node_count = static_cast<int>(domain.num_nodes());
        for (const auto& node : domain.nodes()) {
            if (fall_n::xfem::node_has_shifted_heaviside_enrichment(node)) {
                ++summary.enriched_node_count;
            }
        }
        PetscInt local_state_dofs = 0;
        VecGetSize(model.state_vector(), &local_state_dofs);
        summary.local_state_dofs =
            static_cast<int>(std::max<PetscInt>(local_state_dofs, 0));
        petsc::OwnedVec solver_global{};
        DMCreateGlobalVector(model.get_plex(), solver_global.ptr());
        PetscInt solver_global_dofs = 0;
        VecGetSize(solver_global.get(), &solver_global_dofs);
        summary.solver_global_dofs =
            static_cast<int>(std::max<PetscInt>(solver_global_dofs, 0));
        const int steps = static_cast<int>(
            protocol.size() > 1 ? protocol.size() - 1 : 1);

        using XFEMAnalysis = NonlinearAnalysis<
            ThreeDimensionalMaterial,
            GlobalXFEMKinematicPolicy,
            3,
            MultiElementPolicy>;
        XFEMAnalysis analysis{&model};
        analysis.set_global_residual_hook(
            [&rebar_coupling](Vec u_local, Vec residual_global, DM dm) {
                rebar_coupling.add_to_global_residual(
                    u_local,
                    residual_global,
                    dm);
            });
        analysis.set_jacobian_hook(
            [&rebar_coupling](Vec, Mat jacobian, DM dm) {
                rebar_coupling.add_to_jacobian(jacobian, dm);
            });
        analysis.set_incremental_logging(options.global_xfem_incremental_logging);
        auto tune_profile = [&](fall_n::NonlinearSolveProfile profile) {
            profile.max_iterations = std::max(
                options.global_xfem_solver_max_iterations,
                20);
            profile.atol = 1.0e-9;
            profile.rtol = 1.0e-9;
            profile.divergence_tolerance =
                global_xfem_has_history_material
                    ? 1.0e12
                    : PETSC_DETERMINE;
            if (global_xfem_has_history_material) {
                profile.small_residual_acceptance.profile_atol_multiplier =
                    10.0;
                profile.small_residual_acceptance.accept_diverged_dtol = true;
                profile.small_residual_acceptance.accept_diverged_line_search =
                    true;
                profile.small_residual_acceptance.accept_diverged_max_it = true;
                profile.small_residual_acceptance.accept_diverged_tr_delta =
                    true;
            }
            profile.linear_tuning.factor_reuse_ordering = true;
            profile.linear_tuning.factor_reuse_fill = true;
            return profile;
        };
        const std::string solver_profile_token =
            normalized_material_token(
                options.global_xfem_solver_cascade &&
                        options.global_xfem_solver_profile == "backtracking"
                    ? std::string{"cascade"}
                    : options.global_xfem_solver_profile);
        auto make_solver_profiles = [&]() {
            std::vector<fall_n::NonlinearSolveProfile> profiles;
            auto add = [&](fall_n::NonlinearSolveProfile profile) {
                profiles.push_back(tune_profile(std::move(profile)));
            };

            if (solver_profile_token == "backtracking" ||
                solver_profile_token == "newton-backtracking") {
                add(fall_n::make_newton_backtracking_profile(
                    "global_xfem_newton_backtracking_lu"));
            } else if (solver_profile_token == "l2" ||
                       solver_profile_token == "newton-l2") {
                add(fall_n::make_newton_l2_profile(
                    "global_xfem_newton_l2_lu"));
            } else if (solver_profile_token == "l2-gmres-ilu" ||
                       solver_profile_token == "newton-l2-gmres-ilu" ||
                       solver_profile_token == "gmres-ilu") {
                auto profile = fall_n::make_newton_l2_profile(
                    "global_xfem_newton_l2_gmres_ilu");
                profile.ksp_type = KSPGMRES;
                profile.pc_type = PCILU;
                profile.linear_tuning.ksp_rtol = 1.0e-10;
                profile.linear_tuning.ksp_atol = 1.0e-12;
                profile.linear_tuning.ksp_max_iterations = 500;
                profile.linear_tuning.factor_levels = 1;
                profile.linear_tuning.ksp_reuse_preconditioner = true;
                add(profile);
            } else if (solver_profile_token == "l2-gmres-asm" ||
                       solver_profile_token == "l2-fgmres-asm" ||
                       solver_profile_token == "newton-l2-gmres-asm" ||
                       solver_profile_token == "gmres-asm" ||
                       solver_profile_token == "fgmres-asm") {
                auto profile = fall_n::make_newton_l2_gmres_asm_profile(
                    "global_xfem_newton_l2_fgmres_asm");
                profile.linear_tuning.ksp_rtol = 1.0e-8;
                profile.linear_tuning.ksp_atol = 1.0e-12;
                profile.linear_tuning.ksp_max_iterations = 1000;
                profile.linear_tuning.pc_asm_overlap = 1;
                add(profile);
            } else if (solver_profile_token == "trust-region" ||
                       solver_profile_token == "newton-trust-region") {
                add(fall_n::make_newton_trust_region_profile(
                    "global_xfem_newton_trust_region_lu"));
            } else if (solver_profile_token == "dogleg" ||
                       solver_profile_token == "trust-region-dogleg" ||
                       solver_profile_token == "newton-trust-region-dogleg") {
                add(fall_n::make_newton_trust_region_dogleg_profile(
                    "global_xfem_newton_trust_region_dogleg_lu"));
            } else if (solver_profile_token == "quasi-newton" ||
                       solver_profile_token == "qn") {
                add(fall_n::make_quasi_newton_profile(
                    "global_xfem_quasi_newton_lu"));
            } else if (solver_profile_token == "ngmres" ||
                       solver_profile_token == "nonlinear-gmres") {
                add(fall_n::make_nonlinear_gmres_profile(
                    "global_xfem_nonlinear_gmres_lu"));
            } else if (solver_profile_token == "ncg" ||
                       solver_profile_token == "nonlinear-conjugate-gradient") {
                add(fall_n::make_nonlinear_conjugate_gradient_profile(
                    "global_xfem_nonlinear_conjugate_gradient_lu"));
            } else if (solver_profile_token == "anderson" ||
                       solver_profile_token == "anderson-acceleration") {
                add(fall_n::make_anderson_profile(
                    "global_xfem_anderson_acceleration_lu"));
            } else if (solver_profile_token == "richardson" ||
                       solver_profile_token == "nonlinear-richardson") {
                add(fall_n::make_nonlinear_richardson_profile(
                    "global_xfem_nonlinear_richardson_lu"));
            } else if (solver_profile_token == "cascade" ||
                       solver_profile_token == "newton-cascade") {
                add(fall_n::make_newton_backtracking_profile(
                    "global_xfem_newton_backtracking_lu"));
                add(fall_n::make_newton_l2_profile(
                    "global_xfem_newton_l2_lu"));
                add(fall_n::make_newton_trust_region_profile(
                    "global_xfem_newton_trust_region_lu"));
            } else if (solver_profile_token == "robust-cascade" ||
                       solver_profile_token == "full-cascade") {
                add(fall_n::make_newton_backtracking_profile(
                    "global_xfem_newton_backtracking_lu"));
                add(fall_n::make_newton_l2_profile(
                    "global_xfem_newton_l2_lu"));
                add(fall_n::make_newton_trust_region_profile(
                    "global_xfem_newton_trust_region_lu"));
                add(fall_n::make_newton_trust_region_dogleg_profile(
                    "global_xfem_newton_trust_region_dogleg_lu"));
                add(fall_n::make_quasi_newton_profile(
                    "global_xfem_quasi_newton_lu"));
                add(fall_n::make_nonlinear_gmres_profile(
                    "global_xfem_nonlinear_gmres_lu"));
            } else if (solver_profile_token == "non-newton-cascade") {
                add(fall_n::make_quasi_newton_profile(
                    "global_xfem_quasi_newton_lu"));
                add(fall_n::make_nonlinear_gmres_profile(
                    "global_xfem_nonlinear_gmres_lu"));
                add(fall_n::make_nonlinear_conjugate_gradient_profile(
                    "global_xfem_nonlinear_conjugate_gradient_lu"));
                add(fall_n::make_anderson_profile(
                    "global_xfem_anderson_acceleration_lu"));
            } else {
                throw std::invalid_argument(
                    "Unsupported global XFEM solver profile: " +
                    solver_profile_token);
            }
            return profiles;
        };
        analysis.set_solve_profiles(make_solver_profiles());
        if (global_xfem_has_history_material &&
            options.global_xfem_adaptive_increments) {
            typename XFEMAnalysis::IncrementAdaptationSettings adaptation{};
            adaptation.enabled = true;
            adaptation.min_increment_size =
                1.0 / static_cast<double>(std::max(steps * 2048, 1));
            adaptation.max_increment_size =
                1.0 / static_cast<double>(std::max(steps, 1));
            adaptation.max_cutbacks_per_step = 10;
            adaptation.easy_newton_iterations = 8;
            adaptation.difficult_newton_iterations = 24;
            adaptation.easy_steps_before_growth = 3;
            analysis.set_increment_adaptation(adaptation);
        }
        if (global_xfem_has_history_material) {
            typename XFEMAnalysis::IncrementPredictorSettings predictor{};
            predictor.enabled = true;
            predictor.kind =
                XFEMAnalysis::IncrementPredictorKind::
                    secant_with_linearized_fallback;
            predictor.max_scale_factor = 1.25;
            predictor.max_relative_increment_norm = 1.25;
            predictor.difficult_newton_iterations = 16;
            predictor.disable_during_bisection = true;
            predictor.disable_after_cutback = true;
            analysis.set_increment_predictor(predictor);
        }

        std::ofstream progress_csv(
            options.output_dir / "global_xfem_newton_progress.csv");
        progress_csv << std::setprecision(10);
        write_global_xfem_newton_csv_header(progress_csv);
        auto record_global_xfem_row = [&](GlobalXFEMNewtonRow row) {
            rows.push_back(std::move(row));
            write_global_xfem_newton_csv_row(progress_csv, rows.back());
            progress_csv.flush();
        };

        record_global_xfem_row(GlobalXFEMNewtonRow{});
        auto make_global_xfem_row =
            [&](int step, double p, const XFEMModel& solved_model) {
                const double drift_mm = interpolate_protocol(protocol, p);
                const double base_shear =
                    -sum_support_internal_force_component(
                        solved_model,
                        support_nodes,
                        0);
                const double axial_reaction =
                    -sum_support_internal_force_component(
                        solved_model,
                        support_nodes,
                        2);
                double max_abs_steel_stress = 0.0;
                double max_host_damage = 0.0;
                int damaged_host_points = 0;
                for (std::size_t e = 0;
                     e < reinforced.rebar_range.first;
                     ++e) {
                    for (const auto& snapshot :
                         solved_model.elements()[e].material_point_snapshots()) {
                        if (snapshot.damage.has_value()) {
                            const double damage = *snapshot.damage;
                            max_host_damage =
                                std::max(max_host_damage, damage);
                            if (damage > 1.0e-6) {
                                ++damaged_host_points;
                            }
                        }
                    }
                }
                for (std::size_t e = reinforced.rebar_range.first;
                     e < reinforced.rebar_range.last;
                     ++e) {
                    for (const auto& gp :
                         solved_model.elements()[e].collect_gauss_fields(
                             solved_model.state_vector())) {
                        max_abs_steel_stress = std::max(
                            max_abs_steel_stress,
                            std::abs(gp.stress[0]));
                    }
                }
                for (std::size_t e = crack_crossing_rebar_first;
                     e < crack_crossing_rebar_last;
                     ++e) {
                    for (const auto& gp :
                         solved_model.elements()[e].collect_gauss_fields(
                             solved_model.state_vector())) {
                        max_abs_steel_stress = std::max(
                            max_abs_steel_stress,
                            std::abs(gp.stress[0]));
                    }
                }
                const auto& diagnostics =
                    analysis.last_increment_step_diagnostics();
                return GlobalXFEMNewtonRow{
                    .step = step,
                    .p = p,
                    .drift_mm = drift_mm,
                    .base_shear_mn = base_shear,
                    .axial_reaction_mn = axial_reaction,
                    .max_abs_steel_stress_mpa = max_abs_steel_stress,
                    .max_host_damage = max_host_damage,
                    .damaged_host_points = damaged_host_points,
                    .accepted_substeps =
                        diagnostics.accepted_substep_count,
                    .total_newton_iterations =
                        diagnostics.total_newton_iterations,
                    .failed_attempts =
                        diagnostics.failed_attempt_count,
                    .solver_profile_attempts =
                        diagnostics.solver_profile_attempt_count,
                    .max_bisection_level =
                        diagnostics.max_bisection_level,
                    .last_snes_reason =
                        diagnostics.last_snes_reason,
                    .last_ksp_reason =
                        diagnostics.last_solver_ksp_reason,
                    .accepted_by_small_residual_policy =
                        diagnostics.accepted_by_small_residual_policy,
                    .residual_norm = diagnostics.last_function_norm,
                    .solver_profile_label =
                        diagnostics.last_solver_profile_label};
            };

        GlobalXFEMNewtonRow mixed_control_candidate_row{};
        const auto continuation_token =
            normalized_material_token(options.global_xfem_continuation);
        if (continuation_token == "mixed-arc-length" ||
            continuation_token == "mixed-control" ||
            continuation_token == "mixed-control-arc-length")
        {
            analysis.set_step_callback(
                [&](int step, double p, const XFEMModel& solved_model) {
                    mixed_control_candidate_row =
                        make_global_xfem_row(step, p, solved_model);
                });
        } else {
            analysis.set_step_callback(
                [&](int step, double p, const XFEMModel& solved_model) {
                    record_global_xfem_row(
                        make_global_xfem_row(step, p, solved_model));
                });
        }

        auto scheme = make_control(
            [&top_nodes, &top_rebar_nodes, &protocol](
                double p,
                Vec f_full,
                Vec f_ext,
                XFEMModel* model_ptr) {
                VecCopy(f_full, f_ext);
                const double drift_m =
                    interpolate_protocol(protocol, p) / 1000.0;
                for (const auto node_id : top_nodes) {
                    model_ptr->set_imposed_value_unassembled(
                        node_id,
                        0,
                        drift_m);
                    model_ptr->set_imposed_value_unassembled(
                        node_id,
                        1,
                        0.0);
                }
                for (const auto node_id : top_rebar_nodes) {
                    model_ptr->set_imposed_value_unassembled(
                        node_id,
                        0,
                        drift_m);
                    model_ptr->set_imposed_value_unassembled(
                        node_id,
                        1,
                        0.0);
                }
                model_ptr->finalize_imposed_solution();
            });

        bool ok = false;
        if (continuation_token == "fixed-increment" ||
            continuation_token == "fixed" ||
            continuation_token == "incremental-displacement")
        {
            ok = analysis.solve_incremental(
                steps,
                std::max(options.global_xfem_max_bisections, 0),
                scheme);
            summary.mixed_control_status = "not_used";
        } else if (continuation_token == "bordered-fixed-control" ||
                   continuation_token == "bordered-fixed" ||
                   continuation_token == "petsc-bordered-fixed-control" ||
                   continuation_token == "bordered-fixed-control-hybrid" ||
                   continuation_token == "bordered-fixed-hybrid" ||
                   continuation_token ==
                       "petsc-bordered-fixed-control-hybrid") {
            const bool use_snes_fallback =
                continuation_token == "bordered-fixed-control-hybrid" ||
                continuation_token == "bordered-fixed-hybrid" ||
                continuation_token ==
                    "petsc-bordered-fixed-control-hybrid";
            const double nominal_increment =
                1.0 / static_cast<double>(std::max(steps, 1));
            analysis.begin_incremental(
                steps,
                std::max(options.global_xfem_max_bisections, 0),
                std::move(scheme));
            analysis.set_increment_size(nominal_increment);
            summary.mixed_control_status = use_snes_fallback
                ? "bordered_fixed_control_with_snes_fallback"
                : "bordered_fixed_control";

            ok = true;
            int bordered_failure_streak = 0;
            int bordered_retry_countdown = 0;
            for (int i = 1; i <= steps; ++i) {
                const double target_p =
                    static_cast<double>(i) / static_cast<double>(steps);
                const double p_start = analysis.current_time();
                auto seed = analysis.clone_solution_vector();
                const auto run_snes_fallback = [&]() {
                    analysis.revert_trial_state();
                    analysis.set_solution_vector(seed.get());
                    const auto fallback_verdict = analysis.step_to(target_p);
                    return fallback_verdict == fall_n::StepVerdict::Continue;
                };
                if (use_snes_fallback && bordered_retry_countdown > 0) {
                    --bordered_retry_countdown;
                    ++summary.bordered_hybrid_skipped_steps;
                    if (run_snes_fallback()) {
                        continue;
                    }
                    ok = false;
                    std::ostringstream failure;
                    failure
                        << "SNES fallback failed during bordered hybrid skip "
                        << "at p_target=" << target_p;
                    summary.failure_reason = failure.str();
                    break;
                }
                ++summary.bordered_hybrid_attempted_steps;
                const auto bordered_result =
                    fall_n::solve_petsc_bordered_mixed_control_newton(
                        fall_n::PetscBorderedMixedControlState{
                            .unknowns = seed.get(),
                            .load_parameter = target_p},
                        [&](const fall_n::PetscBorderedMixedControlState& state) {
                            return fall_n::
                                make_fixed_control_petsc_bordered_evaluation(
                                    analysis,
                                    state,
                                    target_p,
                                    fall_n::
                                        PetscNonlinearAnalysisBorderedAdapterSettings{
                                            .control_column_step = 1.0e-6,
                                            .use_central_difference = false});
                        },
                        fall_n::PetscBorderedMixedControlNewtonSettings{
                            .max_iterations = std::max(
                                options.global_xfem_solver_max_iterations,
                                20),
                            .residual_tolerance = 1.0e-8,
                            .constraint_tolerance = 1.0e-12,
                            .line_search_enabled = true,
                            .max_line_search_cutbacks =
                                std::max(
                                    options.global_xfem_max_bisections,
                                    16),
                            .line_search_min_alpha = 1.0e-8,
                            .line_search_extra_trial_count = 24,
                            .line_search_merit =
                                fall_n::PetscBorderedLineSearchMeritKind::
                                    residual_only,
                            .accept_best_line_search_trial = false,
                            .ksp_type = KSPPREONLY,
                            .pc_type = PCLU,
                            .reuse_preconditioner = true});
                if (!bordered_result.converged()) {
                    ++summary.mixed_control_failed_solver_attempts;
                    if (use_snes_fallback) {
                        ++bordered_failure_streak;
                        if (options.global_xfem_bordered_hybrid_disable_streak >
                                0 &&
                            bordered_failure_streak >=
                                options
                                    .global_xfem_bordered_hybrid_disable_streak)
                        {
                            bordered_retry_countdown = std::max(
                                options
                                    .global_xfem_bordered_hybrid_retry_interval,
                                0);
                            bordered_failure_streak = 0;
                        }
                        if (run_snes_fallback()) {
                            continue;
                        }
                    }
                    ok = false;
                    std::ostringstream failure;
                    failure
                        << "bordered fixed-control solve aborted at p_target="
                        << target_p
                        << ", status="
                        << fall_n::to_string(bordered_result.status)
                        << ", iterations=" << bordered_result.iterations
                        << ", residual_norm="
                        << bordered_result.residual_norm
                        << ", constraint_abs="
                        << bordered_result.constraint_abs
                        << ", correction_norm="
                        << bordered_result.correction_norm;
                    if (!bordered_result.records.empty()) {
                        failure
                            << ", last_alpha="
                            << bordered_result.records.back().line_search_alpha
                            << ", last_load_correction_abs="
                            << bordered_result.records.back()
                                   .load_correction_abs;
                    }
                    failure
                        << ", ksp_reason="
                        << static_cast<int>(
                               bordered_result.last_ksp_reason);
                    if (use_snes_fallback) {
                        failure
                            << "; SNES fallback also failed before reaching "
                            << "the same target";
                    }
                    summary.failure_reason = failure.str();
                    break;
                }
                ++summary.bordered_hybrid_successful_steps;
                bordered_failure_streak = 0;
                bordered_retry_countdown = 0;

                analysis.apply_incremental_control_parameter(
                    bordered_result.load_parameter);
                analysis.accept_external_solution_step(
                    bordered_result.unknowns.get(),
                    bordered_result.load_parameter,
                    typename XFEMAnalysis::IncrementStepDiagnostics{
                        .p_start = p_start,
                        .p_target = target_p,
                        .last_attempt_p_start = p_start,
                        .last_attempt_p_target = target_p,
                        .accepted_substep_count = 1,
                        .total_newton_iterations =
                            bordered_result.iterations,
                        .solver_profile_attempt_count =
                            bordered_result.iterations,
                        .last_snes_reason =
                            static_cast<int>(SNES_CONVERGED_FNORM_ABS),
                        .last_function_norm =
                            bordered_result.residual_norm,
                        .last_solver_profile_label =
                            "global_xfem_petsc_bordered_fixed_control_fd_lu",
                        .last_solver_ksp_type = KSPPREONLY,
                        .last_solver_pc_type = PCLU,
                        .last_solver_ksp_reason =
                            static_cast<int>(
                                bordered_result.last_ksp_reason),
                        .last_solver_ksp_iterations =
                            bordered_result.total_ksp_iterations,
                        .last_solver_factor_reuse_ordering = true,
                        .last_solver_factor_reuse_fill = true,
                        .last_solver_ksp_reuse_preconditioner = true,
                        .converged = true});
            }
        } else if (continuation_token == "mixed-arc-length" ||
                   continuation_token == "mixed-control" ||
                   continuation_token == "mixed-control-arc-length") {
            const double nominal_increment =
                1.0 / static_cast<double>(std::max(steps, 1));
            const double min_increment =
                std::isfinite(options.global_xfem_mixed_arc_min_increment)
                    ? options.global_xfem_mixed_arc_min_increment
                    : nominal_increment / 2048.0;
            const double max_increment =
                std::isfinite(options.global_xfem_mixed_arc_max_increment)
                    ? options.global_xfem_mixed_arc_max_increment
                    : nominal_increment;
            const double peak_protocol_mm =
                std::max(
                    1.0,
                    std::ranges::max(
                        protocol |
                        std::views::transform([](double drift_mm) {
                            return std::abs(drift_mm);
                        })));

            analysis.begin_incremental(
                std::max(
                    static_cast<int>(std::ceil(1.0 / max_increment)),
                    1),
                std::max(options.global_xfem_max_bisections, 0),
                std::move(scheme));
            analysis.set_increment_size(max_increment);

            auto sample_mixed_control_state = [&]() {
                const auto& row =
                    mixed_control_candidate_row.step > 0
                        ? mixed_control_candidate_row
                        : rows.back();
                return fall_n::MixedControlArcLengthObservation{
                    .control = row.drift_mm,
                    .reaction = row.base_shear_mn,
                    .internal = row.max_host_damage};
            };

            mixed_control_result =
                fall_n::run_mixed_control_arc_length_continuation(
                    analysis,
                    sample_mixed_control_state,
                    fall_n::MixedControlArcLengthSettings{
                        .target_p = 1.0,
                        .initial_increment = max_increment,
                        .min_increment = min_increment,
                        .max_increment = max_increment,
                        .target_arc_length =
                            options.global_xfem_mixed_arc_target,
                        .reject_arc_length_factor =
                            options.global_xfem_mixed_arc_reject_factor,
                        .max_cutbacks_per_step =
                            std::max(options.global_xfem_max_bisections, 0),
                        .max_total_steps =
                            std::max(steps * 4096, 1),
                        .guard_points = [&]() {
                            std::vector<double> points;
                            points.reserve(static_cast<std::size_t>(steps));
                            for (int i = 1; i <= steps; ++i) {
                                points.push_back(
                                    static_cast<double>(i) /
                                    static_cast<double>(steps));
                            }
                            return points;
                        }(),
                        .scales = {
                            .control = peak_protocol_mm,
                            .reaction =
                                options.global_xfem_mixed_arc_reaction_scale_mn,
                            .internal = 1.0},
                        .weights = {
                            .control = 1.0,
                            .reaction = 1.0,
                            .internal =
                                options.global_xfem_mixed_arc_damage_weight}},
                    [&](const fall_n::MixedControlArcLengthStepRecord&) {
                        record_global_xfem_row(mixed_control_candidate_row);
                    });
            ok = mixed_control_result.completed();
            summary.mixed_control_status =
                std::string{
                    fall_n::to_string(mixed_control_result.status)};
            summary.mixed_control_accepted_steps =
                mixed_control_result.accepted_steps;
            summary.mixed_control_failed_solver_attempts =
                mixed_control_result.failed_solver_attempts;
            summary.mixed_control_rejected_arc_attempts =
                mixed_control_result.rejected_arc_attempts;
            summary.mixed_control_total_cutbacks =
                mixed_control_result.total_cutbacks;
            summary.mixed_control_max_arc_length =
                mixed_control_result.max_arc_length;
            summary.mixed_control_mean_arc_length =
                mixed_control_result.mean_arc_length;
        } else {
            throw std::invalid_argument(
                "Unsupported global XFEM continuation kind: " +
                continuation_token);
        }

        summary.completed = ok;
        summary.status = ok ? "completed" : "aborted";
        if (!ok && summary.failure_reason.empty()) {
            const auto& diagnostics =
                analysis.last_increment_step_diagnostics();
            std::ostringstream failure;
            failure
                << "incremental solve aborted at p_target="
                << diagnostics.p_target
                << ", last_attempt=[" << diagnostics.last_attempt_p_start
                << "," << diagnostics.last_attempt_p_target << "]"
                << ", snes_reason=" << diagnostics.last_snes_reason
                << ", ksp_reason=" << diagnostics.last_solver_ksp_reason
                << ", iterations="
                << diagnostics.total_newton_iterations
                << ", failed_attempts="
                << diagnostics.failed_attempt_count
                << ", residual_norm="
                << diagnostics.last_function_norm
                << ", profile="
                << diagnostics.last_solver_profile_label;
            summary.failure_reason = failure.str();
        }
        summary.point_count = static_cast<int>(rows.size());
        for (const auto& row : rows) {
            summary.peak_abs_drift_mm = std::max(
                summary.peak_abs_drift_mm,
                std::abs(row.drift_mm));
            summary.peak_abs_base_shear_mn = std::max(
                summary.peak_abs_base_shear_mn,
                std::abs(row.base_shear_mn));
            summary.peak_abs_steel_stress_mpa = std::max(
                summary.peak_abs_steel_stress_mpa,
                std::abs(row.max_abs_steel_stress_mpa));
            summary.max_host_damage = std::max(
                summary.max_host_damage,
                row.max_host_damage);
            summary.max_damaged_host_points = std::max(
                summary.max_damaged_host_points,
                row.damaged_host_points);
            summary.total_accepted_substeps += row.accepted_substeps;
            summary.total_nonlinear_iterations +=
                row.total_newton_iterations;
            summary.total_failed_attempts += row.failed_attempts;
            summary.total_solver_profile_attempts +=
                row.solver_profile_attempts;
            summary.max_requested_step_nonlinear_iterations = std::max(
                summary.max_requested_step_nonlinear_iterations,
                row.total_newton_iterations);
            summary.max_bisection_level = std::max(
                summary.max_bisection_level,
                row.max_bisection_level);
            if (row.total_newton_iterations >=
                options.global_xfem_solver_max_iterations)
            {
                summary.hard_step_count += 1;
            }
        }
    } catch (const std::exception& e) {
        summary.completed = false;
        summary.status = "failed";
        summary.failure_reason = e.what();
    }

    const auto toc = std::chrono::steady_clock::now();
    summary.elapsed_seconds =
        std::chrono::duration<double>(toc - tic).count();
    if (summary.point_count == 0) {
        summary.point_count = static_cast<int>(rows.size());
    }
    write_global_xfem_newton_csv(
        options.output_dir / "global_xfem_newton_hysteresis.csv",
        rows);
    write_global_xfem_mixed_control_arc_csv(
        options.output_dir / "global_xfem_mixed_control_arc_length.csv",
        mixed_control_result);
    write_global_xfem_newton_manifest(
        options.output_dir / "global_xfem_newton_manifest.json",
        options,
        spec,
        summary,
        cohesive);
    return summary;
}

[[nodiscard]] GlobalXFEMNewtonSummary run_global_xfem_newton_column_trial(
    const Options& options,
    const ReducedRCColumnReferenceSpec& spec,
    const std::vector<double>& protocol,
    const fall_n::xfem::BilinearCohesiveLawParameters& cohesive)
{
    const auto formulation = normalized_xfem_kinematic_formulation(
        options.global_xfem_kinematic_formulation);
    if (formulation == "corotational") {
        return run_global_xfem_newton_column_trial_impl<
            continuum::Corotational>(options, spec, protocol, cohesive);
    }
    if (formulation == "total-lagrangian") {
        return run_global_xfem_newton_column_trial_impl<
            continuum::TotalLagrangian>(options, spec, protocol, cohesive);
    }
    if (formulation == "updated-lagrangian") {
        return run_global_xfem_newton_column_trial_impl<
            continuum::UpdatedLagrangian>(options, spec, protocol, cohesive);
    }
    return run_global_xfem_newton_column_trial_impl<
        continuum::SmallStrain>(options, spec, protocol, cohesive);
}

void write_hysteresis_csv(
    const std::filesystem::path& path,
    const std::vector<HistoryRow>& rows)
{
    std::ofstream out(path);
    out << std::setprecision(10)
        << "step,drift_mm,theta_y_rad,tangential_slip_mm,base_shear_MN,"
           "flexural_shear_MN,cohesive_shear_MN,steel_shear_MN,moment_MN_m,"
           "concrete_moment_MN_m,steel_moment_MN_m,axial_reaction_MN,"
           "max_opening_mm,min_opening_mm,max_damage,cracked_area_fraction,"
           "max_abs_steel_stress_MPa\n";
    for (const auto& row : rows) {
        out << row.step << ','
            << row.drift_mm << ','
            << row.theta_y_rad << ','
            << row.tangential_slip_mm << ','
            << row.base_shear_mn << ','
            << row.flexural_shear_mn << ','
            << row.cohesive_shear_mn << ','
            << row.steel_shear_mn << ','
            << row.moment_mn_m << ','
            << row.concrete_moment_mn_m << ','
            << row.steel_moment_mn_m << ','
            << row.axial_reaction_mn << ','
            << row.max_opening_mm << ','
            << row.min_opening_mm << ','
            << row.max_damage << ','
            << row.cracked_area_fraction << ','
            << row.max_abs_steel_stress_mpa << '\n';
    }
}

void write_manifest(
    const std::filesystem::path& path,
    const Options& options,
    const ReducedRCColumnReferenceSpec& spec,
    const PetscLayoutAudit& petsc_audit,
    const GlobalXFEMNewtonSummary& global_xfem_newton,
    const std::vector<double>& protocol,
    const std::vector<HistoryRow>& rows,
    double elapsed_seconds)
{
    const auto peak_shear = std::ranges::max(
        rows | std::views::transform([](const HistoryRow& row) {
            return std::abs(row.base_shear_mn);
        }));
    const auto peak_drift = std::ranges::max(
        protocol | std::views::transform([](double drift) {
            return std::abs(drift);
        }));
    const auto max_damage = std::ranges::max(
        rows | std::views::transform([](const HistoryRow& row) {
            return row.max_damage;
        }));

    std::ofstream out(path);
    out << std::setprecision(12)
        << "{\n"
        << "  \"tool\": \"fall_n\",\n"
        << "  \"status\": \"completed\",\n"
        << "  \"driver_kind\": \"xfem_local_cohesive_hinge_reference_benchmark\",\n"
        << "  \"completed_successfully\": true,\n"
        << "  \"local_model_taxonomy\": {\n"
        << "    \"discretization_kind\": \"xfem_local_cohesive_hinge_surrogate\",\n"
        << "    \"fracture_representation_kind\": \"explicit_cohesive_crack_plane\",\n"
        << "    \"reinforcement_representation_kind\": \"section_crossing_menegotto_pinto_bars\",\n"
        << "    \"maturity_kind\": \"future_extension_benchmark_candidate\",\n"
        << "    \"supports_discrete_crack_geometry\": true,\n"
        << "    \"requires_enriched_dofs\": true,\n"
        << "    \"notes\": \"The main plotted hysteresis uses the local base-hinge surrogate. A separate global_xfem_newton_hysteresis.csv artifact now exercises the PETSc/SNES shifted-Heaviside solid element as a full global solve with Menegotto-Pinto truss bars coupled through the enriched host interpolation.\"\n"
        << "  },\n"
        << "  \"global_petsc_xfem_dof_layout_audit\": {\n"
        << "    \"completed\": " << (petsc_audit.completed ? "true" : "false") << ",\n"
        << "    \"global_xfem_solid_kernel_completed\": "
        << (petsc_audit.global_xfem_solid_kernel_completed ? "true" : "false") << ",\n"
        << "    \"solid_elements\": " << petsc_audit.solid_elements << ",\n"
        << "    \"total_nodes\": " << petsc_audit.total_nodes << ",\n"
        << "    \"enriched_nodes\": " << petsc_audit.enriched_nodes << ",\n"
        << "    \"standard_node_dofs\": " << petsc_audit.standard_node_dofs << ",\n"
        << "    \"enriched_node_dofs\": " << petsc_audit.enriched_node_dofs << ",\n"
        << "    \"first_cut_element_gather_dofs\": "
        << petsc_audit.first_cut_element_gather_dofs << ",\n"
        << "    \"first_cut_element_expected_xfem_dofs\": "
        << petsc_audit.first_cut_element_expected_xfem_dofs << ",\n"
        << "    \"enriched_x_local_index\": "
        << petsc_audit.enriched_x_local_index << ",\n"
        << "    \"enriched_x_global_index\": "
        << petsc_audit.enriched_x_global_index << ",\n"
        << "    \"trial_enriched_residual_norm\": "
        << petsc_audit.trial_enriched_residual_norm << ",\n"
        << "    \"trial_enriched_tangent_norm\": "
        << petsc_audit.trial_enriched_tangent_norm << ",\n"
        << "    \"interpretation\": \"The enriched host-node unknowns are real PETSc section DOFs and the shifted-Heaviside solid element now assembles volumetric plus cohesive residual/tangent contributions on those DOFs.\"\n"
        << "  },\n"
        << "  \"global_xfem_newton_column_trial\": {\n"
        << "    \"attempted\": "
        << (global_xfem_newton.attempted ? "true" : "false") << ",\n"
        << "    \"completed_successfully\": "
        << (global_xfem_newton.completed ? "true" : "false") << ",\n"
        << "    \"status\": \""
        << json_escape(global_xfem_newton.status) << "\",\n"
        << "    \"requested_kinematic_formulation\": \""
        << json_escape(global_xfem_newton.requested_kinematic_formulation)
        << "\",\n"
        << "    \"effective_kinematic_formulation\": \""
        << json_escape(global_xfem_newton.effective_kinematic_formulation)
        << "\",\n"
        << "    \"guarded_finite_kinematics_allowed\": "
        << (options.allow_guarded_xfem_finite_kinematics ? "true" : "false")
        << ",\n"
        << "    \"large_amplitude_kinematic_recommendation\": \""
        << json_escape(
               global_xfem_newton.large_amplitude_kinematic_recommendation)
        << "\",\n"
        << "    \"mesh_elements\": "
        << global_xfem_newton.element_count << ",\n"
        << "    \"mesh_nodes\": " << global_xfem_newton.node_count << ",\n"
        << "    \"enriched_nodes\": "
        << global_xfem_newton.enriched_node_count << ",\n"
        << "    \"solver_global_dofs\": "
        << global_xfem_newton.solver_global_dofs << ",\n"
        << "    \"rebar_bar_count\": "
        << global_xfem_newton.rebar_bar_count << ",\n"
        << "    \"rebar_element_count\": "
        << global_xfem_newton.rebar_element_count << ",\n"
        << "    \"rebar_coupling_count\": "
        << global_xfem_newton.rebar_coupling_count << ",\n"
        << "    \"crack_crossing_rebar_element_count\": "
        << global_xfem_newton.crack_crossing_rebar_element_count << ",\n"
        << "    \"point_count\": "
        << global_xfem_newton.point_count << ",\n"
        << "    \"peak_abs_base_shear_mn\": "
        << global_xfem_newton.peak_abs_base_shear_mn << ",\n"
        << "    \"peak_abs_steel_stress_mpa\": "
        << global_xfem_newton.peak_abs_steel_stress_mpa << ",\n"
        << "    \"max_host_damage\": "
        << global_xfem_newton.max_host_damage << ",\n"
        << "    \"max_damaged_host_points\": "
        << global_xfem_newton.max_damaged_host_points << ",\n"
        << "    \"wall_seconds\": "
        << global_xfem_newton.elapsed_seconds << ",\n"
        << "    \"manifest\": \"global_xfem_newton_manifest.json\",\n"
        << "    \"hysteresis_csv\": \"global_xfem_newton_hysteresis.csv\"\n"
        << "  },\n"
        << "  \"protocol\": {\n"
        << "    \"amplitudes_mm\": \"" << options.amplitudes_mm << "\",\n"
        << "    \"steps_per_segment\": " << options.steps_per_segment << ",\n"
        << "    \"point_count\": " << protocol.size() << ",\n"
        << "    \"peak_drift_mm\": " << peak_drift << "\n"
        << "  },\n"
        << "  \"reference_spec\": {\n"
        << "    \"column_height_m\": " << spec.column_height_m << ",\n"
        << "    \"section_b_m\": " << spec.section_b_m << ",\n"
        << "    \"section_h_m\": " << spec.section_h_m << ",\n"
        << "    \"concrete_fpc_mpa\": " << spec.concrete_fpc_mpa << ",\n"
        << "    \"steel_fy_mpa\": " << spec.steel_fy_mpa << "\n"
        << "  },\n"
        << "  \"xfem_surrogate\": {\n"
        << "    \"section_cells_x\": " << options.section_cells_x << ",\n"
        << "    \"section_cells_y\": " << options.section_cells_y << ",\n"
        << "    \"top_rotation_drift_ratio\": " << options.top_rotation_drift_ratio << ",\n"
        << "    \"tangential_slip_drift_ratio\": " << options.tangential_slip_drift_ratio << ",\n"
        << "    \"characteristic_length_mm\": " << options.characteristic_length_mm << ",\n"
        << "    \"residual_shear_fraction\": " << options.residual_shear_fraction << ",\n"
        << "    \"shear_transfer_law\": \"" << options.shear_transfer_law << "\",\n"
        << "    \"shear_cap_mpa\": " << options.shear_cap_mpa << ",\n"
        << "    \"compression_cap_mpa\": " << options.compression_cap_mpa << ",\n"
        << "    \"steel_gauge_length_mm\": " << options.steel_gauge_length_mm << "\n"
        << "  },\n"
        << "  \"observables\": {\n"
        << "    \"max_abs_base_shear_mn\": " << peak_shear << ",\n"
        << "    \"max_damage\": " << max_damage << ",\n"
        << "    \"last_base_shear_mn\": " << rows.back().base_shear_mn << ",\n"
        << "    \"last_drift_mm\": " << rows.back().drift_mm << "\n"
        << "  },\n"
        << "  \"timing\": {\n"
        << "    \"total_wall_seconds\": " << elapsed_seconds << "\n"
        << "  },\n"
        << "  \"artifacts\": {\n"
        << "    \"hysteresis_csv\": \"hysteresis.csv\",\n"
        << "    \"global_xfem_newton_hysteresis_csv\": \"global_xfem_newton_hysteresis.csv\",\n"
        << "    \"global_xfem_newton_manifest\": \"global_xfem_newton_manifest.json\",\n"
        << "    \"visualization_contract_json\": \"visualization_contract.json\",\n"
        << "    \"multiscale_replay_plan_json\": \"multiscale_replay_plan.json\",\n"
        << "    \"multiscale_replay_site_catalog_csv\": \"multiscale_replay_site_catalog.csv\",\n"
        << "    \"multiscale_runtime_policy_json\": \"multiscale_runtime_policy.json\",\n"
        << "    \"multiscale_local_site_batch_plan_json\": \"multiscale_local_site_batch_plan.json\",\n"
        << "    \"multiscale_local_site_batch_plan_csv\": \"multiscale_local_site_batch_plan.csv\",\n"
        << "    \"multiscale_local_site_replay_result_json\": \"multiscale_local_site_replay_result.json\",\n"
        << "    \"multiscale_local_site_replay_result_csv\": \"multiscale_local_site_replay_result.csv\",\n"
        << "    \"multiscale_readiness_gate_json\": \"multiscale_readiness_gate.json\",\n"
        << "    \"cohesive_section_state_csv\": \"cohesive_section_state.csv\",\n"
        << "    \"steel_history_csv\": \"steel_history.csv\"\n"
        << "  }\n"
        << "}\n";
}

} // namespace

int main(int argc, char** argv)
{
    const auto tic = std::chrono::steady_clock::now();
    bool petsc_initialized = false;
    try {
        const Options options = parse_args(argc, argv);
        validate_supported_global_xfem_kinematics(options);
        const ReducedRCColumnReferenceSpec spec =
            default_reduced_rc_column_reference_spec_v;
        if (options.global_xfem_scale_audit_only) {
            std::filesystem::create_directories(options.output_dir);
            const auto audit = make_global_xfem_scale_audit(options, spec);
            write_global_xfem_scale_audit_json(
                options.output_dir / "global_xfem_scale_audit.json",
                options,
                spec,
                audit);
            std::cout
                << "Global XFEM scale audit completed | mesh="
                << std::max(options.global_xfem_nx, 1) << "x"
                << std::max(options.global_xfem_ny, 1) << "x"
                << std::max(options.global_xfem_nz, 2)
                << " | dofs=" << audit.estimated_total_state_dofs
                << " | material points="
                << audit.host_material_point_count
                << " | direct factor risk MiB="
                << audit.direct_factorization_risk_mib
                << " | advice=" << fall_n::to_string(audit.solver_advice)
                << " | output=" << options.output_dir.string() << '\n';
            return 0;
        }
        PetscInitialize(nullptr, nullptr, nullptr, nullptr);
        petsc_initialized = true;
        const auto amplitudes = parse_csv_doubles(options.amplitudes_mm);
        const auto protocol = cyclic_protocol_mm(
            amplitudes,
            options.steps_per_segment);
        const PetscLayoutAudit petsc_audit =
            run_global_petsc_xfem_layout_audit(spec);

        std::filesystem::create_directories(options.output_dir);

        const auto section_spec = to_rc_column_section_spec(spec);
        const auto bar_positions = rc_column_longitudinal_bar_positions(section_spec);
        const double bar_area_m2 = rc_column_longitudinal_bar_area(section_spec);
        std::vector<MenegottoPintoSteel> steel_models;
        steel_models.reserve(bar_positions.size());
        for (std::size_t i = 0; i < bar_positions.size(); ++i) {
            steel_models.emplace_back(spec.steel_E_mpa, spec.steel_fy_mpa, spec.steel_b);
        }

        const double ec_mpa = 4700.0 * std::sqrt(spec.concrete_fpc_mpa);
        const double gc_mpa = ec_mpa / (2.0 * (1.0 + spec.concrete_nu));
        const double ft_mpa = spec.concrete_ft_ratio * spec.concrete_fpc_mpa;
        const double gf_n_per_mm = 0.06;
        const double gross_area_m2 = spec.section_b_m * spec.section_h_m;
        const double axial_stress_mpa =
            gross_area_m2 > 0.0
                ? options.axial_compression_mn / gross_area_m2
                : 0.0;
        const double axial_closure_mm =
            axial_stress_mpa / std::max(ec_mpa, 1.0e-12) *
            options.characteristic_length_mm;

        auto configure_shear_transfer =
            [&](fall_n::xfem::BilinearCohesiveLawParameters& law,
                double opening_decay,
                double shear_cap_mpa) {
                law.shear_transfer_law.kind =
                    parse_shear_law(options.shear_transfer_law);
                law.shear_transfer_law.residual_ratio =
                    options.residual_shear_fraction;
                law.shear_transfer_law.large_opening_ratio =
                    options.residual_shear_fraction;
                law.shear_transfer_law.opening_decay_strain =
                    std::max(opening_decay, 1.0e-12);
                law.shear_transfer_law.max_closed_ratio = 1.0;
                law.shear_traction_cap =
                    std::max(shear_cap_mpa, 1.0e-12);
            };

        auto local_cohesive_mm =
            fall_n::xfem::make_crack_band_consistent_cohesive_law(
                ec_mpa,
                gc_mpa,
                ft_mpa,
                gf_n_per_mm,
                options.characteristic_length_mm,
                options.cohesive_penalty_scale,
                1.0,
                1.0,
                options.residual_shear_fraction);
        configure_shear_transfer(
            local_cohesive_mm,
            options.characteristic_length_mm,
            options.shear_cap_mpa);

        auto global_cohesive_m = fall_n::xfem::
            make_metre_jump_crack_band_consistent_cohesive_law_from_mpa_n_per_mm(
                ec_mpa,
                gc_mpa,
                ft_mpa,
                gf_n_per_mm,
                options.characteristic_length_mm / 1000.0,
                options.cohesive_penalty_scale,
                1.0,
                1.0,
                options.residual_shear_fraction);
        configure_shear_transfer(
            global_cohesive_m,
            options.characteristic_length_mm / 1000.0,
            options.global_xfem_shear_cap_mpa);
        global_cohesive_m.tangent_mode =
            parse_global_xfem_cohesive_tangent_mode(
                options.global_xfem_cohesive_tangent);

        const GlobalXFEMNewtonSummary global_xfem_newton =
            run_global_xfem_newton_column_trial(
                options,
                spec,
                protocol,
                global_cohesive_m);

        const int nx = std::max(options.section_cells_x, 1);
        const int ny = std::max(options.section_cells_y, 1);
        const double dx = spec.section_b_m / static_cast<double>(nx);
        const double dy = spec.section_h_m / static_cast<double>(ny);
        const double dA = dx * dy;
        std::vector<CellState> cells(static_cast<std::size_t>(nx * ny));

        std::vector<HistoryRow> rows;
        rows.reserve(protocol.size());
        std::ofstream section_state(options.output_dir / "cohesive_section_state.csv");
        section_state
            << std::setprecision(10)
            << "step,drift_mm,cell_ix,cell_iy,x_m,y_m,opening_mm,"
               "tangential_slip_mm,traction_x_MPa,traction_z_MPa,damage\n";
        std::ofstream steel_history(options.output_dir / "steel_history.csv");
        steel_history
            << std::setprecision(10)
            << "step,drift_mm,bar_index,x_m,y_m,steel_strain,steel_stress_MPa,"
               "bar_force_MN\n";

        for (std::size_t step = 0; step < protocol.size(); ++step) {
            const double drift_mm = protocol[step];
            const double drift_m = drift_mm / 1000.0;
            const double theta_y =
                options.top_rotation_drift_ratio * drift_m /
                std::max(spec.column_height_m, 1.0e-12);
            const double tangential_slip_mm =
                options.tangential_slip_drift_ratio * drift_mm;

            double concrete_moment = 0.0;
            double cohesive_shear = 0.0;
            double axial_reaction = 0.0;
            double max_opening = -1.0e100;
            double min_opening = 1.0e100;
            double max_damage = 0.0;
            int cracked_cells = 0;

            for (int ix = 0; ix < nx; ++ix) {
                const double x =
                    -0.5 * spec.section_b_m +
                    (static_cast<double>(ix) + 0.5) * dx;
                for (int iy = 0; iy < ny; ++iy) {
                    const double y =
                        -0.5 * spec.section_h_m +
                        (static_cast<double>(iy) + 0.5) * dy;
                    auto& cell = cells[static_cast<std::size_t>(iy * nx + ix)];
                    const double opening_mm =
                        theta_y * x * 1000.0 - axial_closure_mm;
                    const Eigen::Vector3d tangential_jump{
                        tangential_slip_mm, 0.0, 0.0};
                    auto response = fall_n::xfem::evaluate_bilinear_cohesive_law(
                        local_cohesive_mm,
                        cell.cohesive,
                        Eigen::Vector3d::UnitZ(),
                        opening_mm,
                        tangential_jump);
                    cell.cohesive = fall_n::xfem::advance_bilinear_cohesive_state(
                        response);

                    double tx = clamp(
                        response.traction.x(),
                        -options.shear_cap_mpa,
                        options.shear_cap_mpa);
                    double tz = clamp(
                        response.traction.z(),
                        -options.compression_cap_mpa,
                        ft_mpa);
                    cohesive_shear += tx * dA;
                    axial_reaction += tz * dA;
                    concrete_moment += tz * dA * x;
                    max_opening = std::max(max_opening, opening_mm);
                    min_opening = std::min(min_opening, opening_mm);
                    max_damage = std::max(max_damage, response.damage);
                    if (cell.cohesive.max_effective_separation >
                        ft_mpa / std::max(
                            local_cohesive_mm.normal_stiffness,
                            1.0e-12)) {
                        ++cracked_cells;
                    }

                    section_state
                        << step << ','
                        << drift_mm << ','
                        << ix << ','
                        << iy << ','
                        << x << ','
                        << y << ','
                        << opening_mm << ','
                        << tangential_slip_mm << ','
                        << tx << ','
                        << tz << ','
                        << response.damage << '\n';
                }
            }

            double steel_moment = 0.0;
            double max_abs_steel_stress = 0.0;
            for (std::size_t bar = 0; bar < bar_positions.size(); ++bar) {
                const auto& [x, y] = bar_positions[bar];
                const double opening_mm =
                    theta_y * x * 1000.0 - axial_closure_mm;
                const double steel_strain =
                    opening_mm / std::max(options.steel_gauge_length_mm, 1.0e-12);
                Strain<1> strain;
                strain.set_components(steel_strain);
                const double stress_mpa =
                    steel_models[bar].compute_response(strain).components();
                steel_models[bar].update(strain);
                const double force_mn = stress_mpa * bar_area_m2;
                steel_moment += force_mn * x;
                max_abs_steel_stress =
                    std::max(max_abs_steel_stress, std::abs(stress_mpa));
                steel_history
                    << step << ','
                    << drift_mm << ','
                    << bar << ','
                    << x << ','
                    << y << ','
                    << steel_strain << ','
                    << stress_mpa << ','
                    << force_mn << '\n';
            }

            const double total_moment = concrete_moment + steel_moment;
            const double flexural_shear =
                total_moment / std::max(spec.column_height_m, 1.0e-12);
            const double base_shear = flexural_shear + cohesive_shear;

            rows.push_back(HistoryRow{
                .step = static_cast<int>(step),
                .drift_mm = drift_mm,
                .theta_y_rad = theta_y,
                .tangential_slip_mm = tangential_slip_mm,
                .base_shear_mn = base_shear,
                .flexural_shear_mn = flexural_shear,
                .cohesive_shear_mn = cohesive_shear,
                .steel_shear_mn =
                    steel_moment / std::max(spec.column_height_m, 1.0e-12),
                .moment_mn_m = total_moment,
                .concrete_moment_mn_m = concrete_moment,
                .steel_moment_mn_m = steel_moment,
                .axial_reaction_mn = axial_reaction,
                .max_opening_mm = max_opening,
                .min_opening_mm = min_opening,
                .max_damage = max_damage,
                .cracked_area_fraction =
                    static_cast<double>(cracked_cells) /
                    static_cast<double>(nx * ny),
                .max_abs_steel_stress_mpa = max_abs_steel_stress});
        }

        write_hysteresis_csv(options.output_dir / "hysteresis.csv", rows);
        const auto toc = std::chrono::steady_clock::now();
        const double elapsed =
            std::chrono::duration<double>(toc - tic).count();
        write_manifest(
            options.output_dir / "runtime_manifest.json",
            options,
            spec,
            petsc_audit,
            global_xfem_newton,
            protocol,
            rows,
            elapsed);
        write_reduced_rc_visualization_contract_json(
            options.output_dir / "visualization_contract.json");
        const auto multiscale_replay_plan =
            make_reduced_rc_multiscale_replay_plan_from_hinge_history(
                rows,
                options,
                spec);
        write_reduced_rc_multiscale_replay_plan_json(
            options.output_dir / "multiscale_replay_plan.json",
            multiscale_replay_plan);
        write_reduced_rc_multiscale_replay_plan_csv(
            options.output_dir / "multiscale_replay_site_catalog.csv",
            multiscale_replay_plan);
        const auto multiscale_runtime_policy =
            fall_n::make_reduced_rc_multiscale_runtime_policy(
                multiscale_replay_plan);
        write_reduced_rc_multiscale_runtime_policy_json(
            options.output_dir / "multiscale_runtime_policy.json",
            multiscale_runtime_policy);
        const auto local_site_batch_plan =
            fall_n::make_reduced_rc_local_site_batch_plan(
                multiscale_replay_plan,
                multiscale_runtime_policy);
        write_reduced_rc_local_site_batch_plan_json(
            options.output_dir / "multiscale_local_site_batch_plan.json",
            local_site_batch_plan);
        write_reduced_rc_local_site_batch_plan_csv(
            options.output_dir / "multiscale_local_site_batch_plan.csv",
            local_site_batch_plan);
        const auto multiscale_replay_samples =
            make_reduced_rc_replay_samples_from_hinge_history(rows, options);
        constexpr std::string_view replay_oracle_label{
            "structural_history_replay_oracle"};
        auto replay_oracle =
            [replay_oracle_label](
                const fall_n::ReducedRCLocalSiteReplayStepContext& context) {
                return fall_n::ReducedRCLocalSiteReplayStepResult{
                    .converged = true,
                    .nonlinear_iterations = 0,
                    .elapsed_seconds = 0.0,
                    .damage_indicator =
                        context.target_sample.damage_indicator,
                    .steel_stress_mpa =
                        context.target_sample.steel_stress_mpa,
                    .local_work_increment_mn_mm =
                        context.target_sample.work_increment_mn_mm,
                    .status_label = replay_oracle_label};
            };
        fall_n::ReducedRCLocalSiteReplaySettings replay_settings;
        replay_settings.continue_after_site_failure = true;
        const auto local_site_replay_result =
            local_site_batch_plan.executor_kind ==
                    fall_n::ReducedRCMultiscaleExecutorKind::
                        openmp_site_parallel
                ? fall_n::run_reduced_rc_local_site_replay_batch(
                      multiscale_replay_samples,
                      local_site_batch_plan,
                      replay_oracle,
                      replay_settings,
                      fall_n::ReducedRCOpenMPSiteReplayExecutor{
                          local_site_batch_plan.recommended_site_threads})
                : fall_n::run_reduced_rc_local_site_replay_batch(
                      multiscale_replay_samples,
                      local_site_batch_plan,
                      replay_oracle,
                      replay_settings);
        write_reduced_rc_local_site_replay_result_json(
            options.output_dir / "multiscale_local_site_replay_result.json",
            local_site_replay_result,
            replay_oracle_label);
        write_reduced_rc_local_site_replay_result_csv(
            options.output_dir / "multiscale_local_site_replay_result.csv",
            local_site_replay_result);
        const auto multiscale_readiness_gate =
            fall_n::make_reduced_rc_multiscale_readiness_gate(
                multiscale_replay_plan,
                multiscale_runtime_policy,
                local_site_batch_plan,
                local_site_replay_result,
                fall_n::ReducedRCMultiscaleReadinessGateSettings{
                    .physical_local_solver_bound = false,
                    .local_solver_label = replay_oracle_label});
        write_reduced_rc_multiscale_readiness_gate_json(
            options.output_dir / "multiscale_readiness_gate.json",
            multiscale_readiness_gate);

        const auto peak = std::ranges::max(
            rows | std::views::transform([](const HistoryRow& row) {
                return std::abs(row.base_shear_mn);
            }));
        std::cout
            << "XFEM local cohesive hinge benchmark completed | points="
            << rows.size()
            << " | peak drift="
            << *std::ranges::max_element(
                   protocol,
                   {},
                   [](double value) { return std::abs(value); })
            << " mm | max |Vb|=" << peak
            << " MN | global_xfem_status="
            << global_xfem_newton.status
            << " | global_xfem_completed="
            << (global_xfem_newton.completed ? "true" : "false")
            << " | multiscale_next="
            << fall_n::to_string(multiscale_readiness_gate.next_stage)
            << " | output=" << options.output_dir.string() << '\n';
        PetscFinalize();
        petsc_initialized = false;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "XFEM benchmark failed: " << e.what() << '\n';
        if (petsc_initialized) {
            PetscFinalize();
        }
        return 1;
    }
}
