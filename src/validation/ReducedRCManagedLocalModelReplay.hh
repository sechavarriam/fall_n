#ifndef FALL_N_REDUCED_RC_MANAGED_LOCAL_MODEL_REPLAY_HH
#define FALL_N_REDUCED_RC_MANAGED_LOCAL_MODEL_REPLAY_HH

// =============================================================================
//  ReducedRCManagedLocalModelReplay.hh
// =============================================================================
//
//  Managed local-model replay contract for the reduced-RC FE2 validation path.
//
//  A selected multiscale site is not interpreted as "one XFEM model per failed
//  integration point".  It is a managed local boundary-value problem with its
//  own domain, mesh, material states, checkpoint cache and output stream.  The
//  macro structural element only supplies a time-ordered history of generalized
//  kinematics and resultants; the local model receives those fields as imposed
//  boundary data and evolves its own state across pseudo-time.
//
//  This header keeps that protocol independent from PETSc and from the concrete
//  XFEM implementation.  A real adapter can wrap `Model<...>` and NLAnalysis;
//  tests can use a tiny stateful mock while preserving the same semantics.
//
// =============================================================================

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <string_view>
#include <vector>

#include <Eigen/Dense>

#include "src/analysis/MultiscaleMaterialHistoryTransfer.hh"
#include "src/analysis/MultiscaleTypes.hh"
#include "src/validation/ReducedRCMultiscaleReplayPlan.hh"

namespace fall_n {

enum class ReducedRCManagedLocalBoundaryMode {
    top_face_dirichlet,
    affine_section_dirichlet,
    mixed_dirichlet_neumann
};

enum class ReducedRCLocalLongitudinalBiasLocation {
    fixed_end,
    loaded_end,
    both_ends
};

[[nodiscard]] constexpr std::string_view to_string(
    ReducedRCManagedLocalBoundaryMode mode) noexcept
{
    switch (mode) {
        case ReducedRCManagedLocalBoundaryMode::top_face_dirichlet:
            return "top_face_dirichlet";
        case ReducedRCManagedLocalBoundaryMode::affine_section_dirichlet:
            return "affine_section_dirichlet";
        case ReducedRCManagedLocalBoundaryMode::mixed_dirichlet_neumann:
            return "mixed_dirichlet_neumann";
    }
    return "unknown_reduced_rc_managed_local_boundary_mode";
}

[[nodiscard]] constexpr std::string_view to_string(
    ReducedRCLocalLongitudinalBiasLocation location) noexcept
{
    switch (location) {
        case ReducedRCLocalLongitudinalBiasLocation::fixed_end:
            return "fixed_end";
        case ReducedRCLocalLongitudinalBiasLocation::loaded_end:
            return "loaded_end";
        case ReducedRCLocalLongitudinalBiasLocation::both_ends:
            return "both_ends";
    }
    return "unknown_reduced_rc_local_longitudinal_bias_location";
}

enum class ReducedRCManagedLocalReplayStatus {
    not_run,
    completed,
    empty_history,
    adapter_initialization_failed,
    local_solve_failed
};

[[nodiscard]] constexpr std::string_view to_string(
    ReducedRCManagedLocalReplayStatus status) noexcept
{
    switch (status) {
        case ReducedRCManagedLocalReplayStatus::not_run:
            return "not_run";
        case ReducedRCManagedLocalReplayStatus::completed:
            return "completed";
        case ReducedRCManagedLocalReplayStatus::empty_history:
            return "empty_history";
        case ReducedRCManagedLocalReplayStatus::adapter_initialization_failed:
            return "adapter_initialization_failed";
        case ReducedRCManagedLocalReplayStatus::local_solve_failed:
            return "local_solve_failed";
    }
    return "unknown_reduced_rc_managed_local_replay_status";
}

struct ReducedRCManagedLocalPatchSpec {
    std::size_t site_index{0};
    double z_over_l{0.0};
    double characteristic_length_m{0.10};
    double section_width_m{0.20};
    double section_depth_m{0.20};
    std::size_t nx{2};
    std::size_t ny{2};
    std::size_t nz{4};
    double crack_z_over_l{0.40};
    double longitudinal_bias_power{1.0};
    ReducedRCLocalLongitudinalBiasLocation longitudinal_bias_location{
        ReducedRCLocalLongitudinalBiasLocation::fixed_end};
    bool crack_position_inferred_from_macro{false};
    bool double_hinge_bias_inferred_from_macro{false};
    ReducedRCManagedLocalBoundaryMode boundary_mode{
        ReducedRCManagedLocalBoundaryMode::affine_section_dirichlet};
    bool independent_domain_and_mesh{true};
    bool vtk_time_series_required{true};
    bool warm_start_required{true};
    bool vtk_global_placement{false};
    std::array<double, 3> vtk_origin{0.0, 0.0, 0.0};
    std::array<double, 3> vtk_e_x{1.0, 0.0, 0.0};
    std::array<double, 3> vtk_e_y{0.0, 1.0, 0.0};
    std::array<double, 3> vtk_e_z{0.0, 0.0, 1.0};
    std::size_t vtk_parent_element_id{0};
    std::size_t vtk_section_gp{0};
    double vtk_xi{0.0};
};

struct ReducedRCManagedLocalBoundarySample {
    std::size_t site_index{0};
    std::size_t sample_index{0};
    double pseudo_time{0.0};
    double physical_time{0.0};
    double z_over_l{0.0};
    double tip_drift_m{0.0};
    double curvature_y{0.0};
    double curvature_z{0.0};
    double imposed_rotation_y_rad{0.0};
    double imposed_rotation_z_rad{0.0};
    double axial_strain{0.0};
    double macro_moment_y_mn_m{0.0};
    double macro_moment_z_mn_m{0.0};
    double macro_base_shear_mn{0.0};
    double macro_steel_stress_mpa{0.0};
    double macro_damage_indicator{0.0};
    double macro_work_increment_mn_mm{0.0};
    Eigen::Vector3d imposed_top_translation_m{Eigen::Vector3d::Zero()};
    Eigen::Vector3d imposed_top_rotation_rad{Eigen::Vector3d::Zero()};
};

struct ReducedRCManagedLocalStepResult {
    bool converged{true};
    bool hard_failure{false};
    int nonlinear_iterations{0};
    double elapsed_seconds{0.0};
    double residual_norm{0.0};
    double local_work_increment_mn_mm{
        std::numeric_limits<double>::quiet_NaN()};
    double max_damage_indicator{0.0};
    double peak_abs_steel_stress_mpa{0.0};
    std::string_view status_label{"converged"};
};

struct ReducedRCManagedLocalReplaySettings {
    bool stop_on_first_failure{true};
    double default_axial_strain{0.0};
    MaterialHistorySeedPolicy material_history_seed_policy{
        MaterialHistorySeedPolicy::SeedThenReplayIncrement};
    bool build_macro_material_history_packet{true};
};

struct ReducedRCManagedLocalReplayResult {
    ReducedRCManagedLocalReplayStatus status{
        ReducedRCManagedLocalReplayStatus::not_run};
    std::size_t site_index{0};
    std::size_t input_sample_count{0};
    std::size_t attempted_step_count{0};
    std::size_t accepted_step_count{0};
    std::size_t model_instance_count{0};
    int total_nonlinear_iterations{0};
    double total_elapsed_seconds{0.0};
    double accumulated_abs_work_mn_mm{0.0};
    double max_damage_indicator{0.0};
    double peak_abs_steel_stress_mpa{0.0};
    MaterialHistoryTransferPacket macro_material_history{};
    UpscalingResult homogenized_response{};
    std::string_view failure_reason{};

    [[nodiscard]] bool completed() const noexcept
    {
        return status == ReducedRCManagedLocalReplayStatus::completed;
    }
};

template <typename Adapter>
concept ReducedRCManagedLocalModelAdapter =
    requires(Adapter adapter,
             const ReducedRCManagedLocalPatchSpec& patch,
             const ReducedRCManagedLocalBoundarySample& sample) {
        { adapter.initialize_managed_local_model(patch) } ->
            std::convertible_to<bool>;
        { adapter.apply_macro_boundary_sample(sample) } ->
            std::convertible_to<bool>;
        { adapter.solve_current_pseudo_time_step(sample) } ->
            std::same_as<ReducedRCManagedLocalStepResult>;
        { adapter.homogenized_section_response() } ->
            std::same_as<UpscalingResult>;
    };

[[nodiscard]] inline ReducedRCManagedLocalBoundarySample
make_reduced_rc_managed_local_boundary_sample(
    const ReducedRCStructuralReplaySample& sample,
    const ReducedRCManagedLocalPatchSpec& patch,
    std::size_t sample_index,
    double axial_strain = 0.0)
{
    ReducedRCManagedLocalBoundarySample out{};
    out.site_index = sample.site_index;
    out.sample_index = sample_index;
    out.pseudo_time = sample.pseudo_time;
    out.physical_time = sample.physical_time;
    out.z_over_l = sample.z_over_l;
    out.tip_drift_m = sample.drift_mm / 1000.0;
    out.curvature_y = sample.curvature_y;
    out.curvature_z = sample.curvature_z;
    out.imposed_rotation_y_rad = sample.curvature_y *
        std::max(patch.characteristic_length_m, 0.0);
    out.imposed_rotation_z_rad = sample.curvature_z *
        std::max(patch.characteristic_length_m, 0.0);
    out.axial_strain = axial_strain;
    out.macro_moment_y_mn_m = sample.moment_y_mn_m;
    out.macro_moment_z_mn_m = sample.moment_z_mn_m;
    out.macro_base_shear_mn = sample.base_shear_mn;
    out.macro_steel_stress_mpa = sample.steel_stress_mpa;
    out.macro_damage_indicator = sample.damage_indicator;
    out.macro_work_increment_mn_mm = sample.work_increment_mn_mm;

    // First reduced-column closure: one lateral direction, top-face Dirichlet.
    // Richer adapters may reinterpret the same generalized fields as an affine
    // boundary map over all local boundary nodes.
    out.imposed_top_translation_m =
        Eigen::Vector3d{out.tip_drift_m, 0.0,
                        axial_strain * patch.characteristic_length_m};
    out.imposed_top_rotation_rad =
        Eigen::Vector3d{0.0,
                        out.imposed_rotation_y_rad,
                        out.imposed_rotation_z_rad};
    return out;
}

[[nodiscard]] inline MaterialHistorySiteKey
make_reduced_rc_managed_section_history_key(
    const ReducedRCStructuralReplaySample& sample,
    const ReducedRCManagedLocalPatchSpec& patch) noexcept
{
    MaterialHistorySiteKey key{};
    key.site.macro_element_id = sample.site_index;
    key.site.section_gp = 0;
    key.role = MaterialHistorySiteRole::SectionResultant;
    key.local_site_index = patch.site_index;
    key.xi = sample.z_over_l;
    key.y = 0.0;
    key.z = sample.z_over_l;
    return key;
}

[[nodiscard]] inline MaterialHistoryTransferPacket
make_reduced_rc_managed_section_history_packet(
    const std::vector<ReducedRCStructuralReplaySample>& site_history,
    const ReducedRCManagedLocalPatchSpec& patch,
    MaterialHistorySeedPolicy seed_policy =
        MaterialHistorySeedPolicy::SeedThenReplayIncrement,
    double axial_strain = 0.0)
{
    MaterialHistoryTransferPacket packet{};
    packet.direction = MaterialHistoryTransferDirection::MacroToLocal;
    packet.seed_policy = seed_policy;
    packet.source_label = "macro_structural_section_history";
    packet.target_label = "managed_local_section_boundary_history";
    packet.samples.reserve(site_history.size());

    for (const auto& sample : site_history) {
        auto key = make_reduced_rc_managed_section_history_key(sample, patch);
        Eigen::VectorXd eta(3);
        eta << axial_strain, sample.curvature_y, sample.curvature_z;
        Eigen::VectorXd q(3);
        q << 0.0, sample.moment_y_mn_m, sample.moment_z_mn_m;
        packet.samples.push_back(make_section_generalized_material_history_sample(
            key, std::move(eta), std::move(q),
            sample.pseudo_time, sample.physical_time));
    }
    return packet;
}

template <ReducedRCManagedLocalModelAdapter Adapter>
[[nodiscard]] ReducedRCManagedLocalReplayResult
run_reduced_rc_managed_local_model_replay(
    const std::vector<ReducedRCStructuralReplaySample>& site_history,
    const ReducedRCManagedLocalPatchSpec& patch,
    Adapter& adapter,
    ReducedRCManagedLocalReplaySettings settings = {})
{
    ReducedRCManagedLocalReplayResult result{};
    result.site_index = patch.site_index;
    result.input_sample_count = site_history.size();
    if (settings.build_macro_material_history_packet) {
        result.macro_material_history =
            make_reduced_rc_managed_section_history_packet(
                site_history,
                patch,
                settings.material_history_seed_policy,
                settings.default_axial_strain);
    }

    if (site_history.empty()) {
        result.status = ReducedRCManagedLocalReplayStatus::empty_history;
        result.failure_reason = "no structural samples for managed local model";
        return result;
    }

    if (!adapter.initialize_managed_local_model(patch)) {
        result.status =
            ReducedRCManagedLocalReplayStatus::adapter_initialization_failed;
        result.failure_reason = "managed local model adapter rejected patch";
        return result;
    }
    result.model_instance_count = 1;

    for (std::size_t i = 0; i < site_history.size(); ++i) {
        const auto boundary = make_reduced_rc_managed_local_boundary_sample(
            site_history[i], patch, i, settings.default_axial_strain);
        ++result.attempted_step_count;
        if (!adapter.apply_macro_boundary_sample(boundary)) {
            result.status = ReducedRCManagedLocalReplayStatus::local_solve_failed;
            result.failure_reason = "managed local model rejected boundary sample";
            return result;
        }

        const auto step = adapter.solve_current_pseudo_time_step(boundary);
        if (!step.converged || step.hard_failure) {
            result.status = ReducedRCManagedLocalReplayStatus::local_solve_failed;
            result.failure_reason = step.status_label;
            if (settings.stop_on_first_failure) {
                return result;
            }
            continue;
        }

        ++result.accepted_step_count;
        result.total_nonlinear_iterations += step.nonlinear_iterations;
        result.total_elapsed_seconds +=
            std::isfinite(step.elapsed_seconds) ? step.elapsed_seconds : 0.0;
        const double work = std::isfinite(step.local_work_increment_mn_mm)
            ? step.local_work_increment_mn_mm
            : site_history[i].work_increment_mn_mm;
        result.accumulated_abs_work_mn_mm += std::abs(work);
        result.max_damage_indicator = std::max(
            result.max_damage_indicator,
            std::clamp(step.max_damage_indicator, 0.0, 1.0));
        result.peak_abs_steel_stress_mpa = std::max(
            result.peak_abs_steel_stress_mpa,
            std::abs(step.peak_abs_steel_stress_mpa));
    }

    result.homogenized_response = adapter.homogenized_section_response();
    result.status = ReducedRCManagedLocalReplayStatus::completed;
    return result;
}

} // namespace fall_n

#endif // FALL_N_REDUCED_RC_MANAGED_LOCAL_MODEL_REPLAY_HH
