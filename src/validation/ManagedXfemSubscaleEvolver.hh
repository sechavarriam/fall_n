#ifndef FALL_N_MANAGED_XFEM_SUBSCALE_EVOLVER_HH
#define FALL_N_MANAGED_XFEM_SUBSCALE_EVOLVER_HH

// =============================================================================
//  ManagedXfemSubscaleEvolver.hh
// =============================================================================
//
//  FE2-facing adapter around the managed XFEM local model.  It satisfies the
//  section-local model contract consumed by MultiscaleAnalysis while keeping the
//  local XFEM problem as one persistent, independent Model per promoted macro
//  site.  The macro element supplies two end-section kinematic snapshots; this
//  evolver reduces them to the crack-band station selected by the macro-inferred
//  local-site policy and delegates the actual local solve to
//  ReducedRCManagedXfemLocalModelAdapter.
//
// =============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <format>
#include <optional>
#include <string>
#include <utility>

#include "src/analysis/HomogenizedTangentFiniteDifference.hh"
#include "src/analysis/MultiscaleTypes.hh"
#include "src/post-processing/VTK/PVDWriter.hh"
#include "src/reconstruction/FieldTransfer.hh"
#include "src/reconstruction/LocalModelAdapter.hh"
#include "src/reconstruction/LocalCrackData.hh"
#include "src/reconstruction/SubModelSolver.hh"
#include "src/validation/ReducedRCManagedXfemLocalModelAdapter.hh"

namespace fall_n {

struct ManagedXfemAdaptiveTransitionPolicy {
    bool enabled{false};
    int min_transition_steps{1};
    int base_transition_steps{2};
    int max_transition_steps{8};
    int min_bisections{4};
    int base_bisections{6};
    int max_bisections{10};
    double strain_scale{2.0e-4};
    double curvature_scale{2.5e-4};
    double shear_scale{2.0e-4};
    double twist_scale{2.0e-4};
    double warning_increment_severity{0.65};
    double critical_increment_severity{1.50};
    int high_iteration_threshold{10};
};

struct ManagedXfemTransitionControl {
    int transition_steps{1};
    int max_bisections{0};
    double increment_severity{0.0};
    bool adaptive{false};
    std::string reason{"fixed"};
};

[[nodiscard]] inline double managed_xfem_control_increment_severity(
    const Eigen::Vector<double, 6>& delta,
    const ManagedXfemAdaptiveTransitionPolicy& policy) noexcept
{
    const double strain_scale = std::max(policy.strain_scale, 1.0e-12);
    const double curvature_scale =
        std::max(policy.curvature_scale, 1.0e-12);
    const double shear_scale = std::max(policy.shear_scale, 1.0e-12);
    const double twist_scale = std::max(policy.twist_scale, 1.0e-12);
    return std::max({std::abs(delta[0]) / strain_scale,
                     std::abs(delta[1]) / curvature_scale,
                     std::abs(delta[2]) / curvature_scale,
                     std::abs(delta[3]) / shear_scale,
                     std::abs(delta[4]) / shear_scale,
                     std::abs(delta[5]) / twist_scale});
}

[[nodiscard]] inline ManagedXfemTransitionControl
select_managed_xfem_transition_control(
    const ManagedXfemAdaptiveTransitionPolicy& policy,
    int base_transition_steps,
    int base_max_bisections,
    const Eigen::Vector<double, 6>& control_increment,
    int previous_local_iterations,
    bool previous_local_failed)
{
    const int base_steps = std::max(1, base_transition_steps);
    const int base_bis = std::max(0, base_max_bisections);
    ManagedXfemTransitionControl control{
        .transition_steps = base_steps,
        .max_bisections = base_bis,
        .increment_severity =
            managed_xfem_control_increment_severity(control_increment, policy),
        .adaptive = policy.enabled,
        .reason = policy.enabled ? "adaptive_base" : "fixed"};
    if (!policy.enabled) {
        return control;
    }

    const int min_steps = std::clamp(policy.min_transition_steps, 1,
                                     std::max(1, policy.max_transition_steps));
    const int max_steps = std::max(min_steps, policy.max_transition_steps);
    const int min_bis = std::clamp(policy.min_bisections, 0,
                                   std::max(0, policy.max_bisections));
    const int max_bis = std::max(min_bis, policy.max_bisections);
    const int nominal_steps =
        std::clamp(policy.base_transition_steps > 0
                       ? policy.base_transition_steps
                       : base_steps,
                   min_steps,
                   max_steps);
    const int nominal_bis =
        std::clamp(policy.base_bisections >= 0 ? policy.base_bisections
                                               : base_bis,
                   min_bis,
                   max_bis);
    control.transition_steps = nominal_steps;
    control.max_bisections = nominal_bis;

    if (previous_local_failed) {
        control.transition_steps = max_steps;
        control.max_bisections = max_bis;
        control.reason = "previous_local_failure";
        return control;
    }

    const bool solver_pressure =
        policy.high_iteration_threshold > 0 &&
        previous_local_iterations >= policy.high_iteration_threshold;
    const double warning =
        std::max(0.0, policy.warning_increment_severity);
    const double critical =
        std::max(warning + 1.0e-12, policy.critical_increment_severity);

    if (solver_pressure || control.increment_severity >= critical) {
        control.transition_steps = max_steps;
        control.max_bisections = max_bis;
        control.reason = solver_pressure
            ? "previous_solver_pressure"
            : "critical_control_increment";
        return control;
    }

    if (control.increment_severity >= warning) {
        const double r =
            std::clamp((control.increment_severity - warning) /
                           (critical - warning),
                       0.0,
                       1.0);
        control.transition_steps = std::clamp(
            static_cast<int>(std::lround(nominal_steps +
                                         r * (max_steps - nominal_steps))),
            min_steps,
            max_steps);
        control.max_bisections = std::clamp(
            static_cast<int>(std::lround(nominal_bis +
                                         r * (max_bis - nominal_bis))),
            min_bis,
            max_bis);
        control.reason = "warning_control_increment";
        return control;
    }

    if (control.increment_severity < 0.25 && previous_local_iterations <= 3) {
        control.transition_steps = min_steps;
        control.max_bisections = min_bis;
        control.reason = "low_increment_fast_path";
        return control;
    }

    control.reason = "nominal_increment";
    return control;
}

class ManagedXfemSubscaleEvolver {
public:
    struct checkpoint_type {
        SectionKinematics kin_A{};
        SectionKinematics kin_B{};
        MacroSectionState macro_state{};
        ReducedRCManagedLocalBoundarySample trial_sample{};
        ReducedRCManagedLocalBoundarySample committed_sample{};
        SectionHomogenizedResponse last_response{};
        SectionHomogenizedResponse committed_response{};
        ReducedRCManagedLocalStepResult last_step{};
        SubModelSolverResult last_result{};
        CrackSummary last_crack_summary{};
        MaterialHistoryTransferPacket trial_material_history{};
        MaterialHistoryTransferPacket committed_material_history{};
        Eigen::Vector<double, 6> committed_control{
            Eigen::Vector<double, 6>::Zero()};
        ManagedXfemTransitionControl last_transition_control{};
        ManagedXfemAdaptiveTransitionPolicy adaptive_transition_policy{};
        bool has_macro_state{false};
        bool has_trial_sample{false};
        bool has_committed_sample{false};
        bool has_trial_material_history{false};
        bool has_committed_material_history{false};
        bool has_committed_control{false};
        bool auto_commit{true};
        int step_count{0};
    };

    ManagedXfemSubscaleEvolver(
        std::size_t parent_element_id,
        ReducedRCManagedLocalPatchSpec patch,
        ReducedRCManagedXfemLocalModelAdapterOptions options = {})
        : parent_element_id_{parent_element_id}
        , patch_{std::move(patch)}
        , options_{options}
    {
        patch_.site_index = parent_element_id_;
    }

    ManagedXfemSubscaleEvolver(ManagedXfemSubscaleEvolver&&) noexcept =
        default;
    ManagedXfemSubscaleEvolver& operator=(
        ManagedXfemSubscaleEvolver&&) noexcept = default;
    ManagedXfemSubscaleEvolver(const ManagedXfemSubscaleEvolver&) = delete;
    ManagedXfemSubscaleEvolver& operator=(
        const ManagedXfemSubscaleEvolver&) = delete;

    void update_kinematics(const SectionKinematics& kin_A,
                           const SectionKinematics& kin_B)
    {
        kin_A_ = kin_A;
        kin_B_ = kin_B;
        has_cached_fd_response_ = false;
    }

    void update_macro_section_state(const MacroSectionState& state)
    {
        macro_state_ = state;
        has_macro_state_ = true;
        has_cached_fd_response_ = false;
    }

    void set_tangent_computation_mode(TangentComputationMode mode) noexcept
    {
        tangent_mode_ = mode;
    }

    [[nodiscard]] TangentComputationMode tangent_computation_mode()
        const noexcept
    {
        return tangent_mode_;
    }

    void set_finite_difference_tangent_settings(
        HomogenizedTangentFiniteDifferenceSettings settings) noexcept
    {
        fd_settings_ = settings;
    }

    void set_adaptive_transition_policy(
        ManagedXfemAdaptiveTransitionPolicy policy) noexcept
    {
        adaptive_transition_policy_ = policy;
    }

    [[nodiscard]] ManagedXfemAdaptiveTransitionPolicy
    adaptive_transition_policy() const noexcept
    {
        return adaptive_transition_policy_;
    }

    [[nodiscard]] const ManagedXfemTransitionControl&
    last_transition_control() const noexcept
    {
        return last_transition_control_;
    }

    void set_vtk_output_profile(LocalVTKOutputProfile profile) noexcept
    {
        vtk_output_profile_ = profile;
        if (adapter_) {
            adapter_->set_vtk_output_profile(profile);
        }
    }

    [[nodiscard]] LocalVTKOutputProfile vtk_output_profile() const noexcept
    {
        return vtk_output_profile_;
    }

    [[nodiscard]] SubModelSolverResult solve_step(double time)
    {
        SubModelSolverResult result{};
        result.stage = has_committed_sample_
            ? SubModelSolveStage::SubsequentFullStep
            : SubModelSolveStage::FirstSolveSingleStep;
        has_cached_fd_response_ = false;

        if (!ensure_initialized_()) {
            result.converged = false;
            result.failure_cause = SubModelFailureCause::FunctionDomain;
            last_result_ = result;
            return result;
        }

        trial_sample_ = make_sample_(time);
        has_trial_sample_ = true;
        const auto trial_control = sample_control_vector_(trial_sample_);
        const auto control_increment = has_committed_sample_
            ? trial_control - sample_control_vector_(committed_sample_)
            : trial_control;
        last_transition_control_ =
            select_managed_xfem_transition_control(
                adaptive_transition_policy_,
                options_.local_transition_steps,
                options_.local_max_bisections,
                control_increment,
                last_step_.nonlinear_iterations,
                !last_result_.converged &&
                    last_result_.failure_cause !=
                        SubModelFailureCause::None);
        adapter_->set_local_transition_controls(
            last_transition_control_.transition_steps,
            last_transition_control_.max_bisections);
        const bool boundary_ok =
            adapter_->apply_macro_boundary_sample(trial_sample_);
        if (!boundary_ok) {
            result.converged = false;
            result.failure_cause = SubModelFailureCause::FunctionDomain;
            last_result_ = result;
            return result;
        }

        const auto step =
            adapter_->solve_current_pseudo_time_step(trial_sample_);
        result.converged = step.converged && !step.hard_failure;
        result.failure_cause = result.converged
            ? SubModelFailureCause::None
            : SubModelFailureCause::NewtonDiverged;
        result.snes_iterations =
            static_cast<PetscInt>(std::max(0, step.nonlinear_iterations));
        result.function_norm = step.residual_norm;
        result.max_displacement =
            trial_sample_.imposed_top_translation_m.cwiseAbs().maxCoeff();

        last_step_ = step;
        last_response_ = make_section_response_from_upscaling_(
            adapter_->homogenized_section_response());
        trial_material_history_ =
            make_section_material_history_packet_(trial_sample_, last_response_);
        has_trial_material_history_ = true;
        const auto crack_state = adapter_->local_crack_state();
        if (crack_state.summary.total_cracks > 0 ||
            crack_state.summary.fracture_history_available) {
            last_crack_summary_ = crack_state.summary;
        } else {
            last_crack_summary_.num_cracked_gps =
                step.max_damage_indicator > 0.0 ? 1 : 0;
            last_crack_summary_.total_cracks =
                step.max_damage_indicator > 0.0 ? 1 : 0;
            last_crack_summary_.damage_scalar_available = true;
            last_crack_summary_.max_damage_scalar = step.max_damage_indicator;
            last_crack_summary_.max_opening = 0.0;
        }

        if (result.converged && auto_commit_) {
            commit_trial_state();
        }
        last_result_ = result;
        return result;
    }

    [[nodiscard]] SectionHomogenizedResponse
    section_response(double width, double height, double h_pert = 1.0e-6)
    {
        auto response = last_response_;
        if (response.status == ResponseStatus::NotReady) {
            response = make_section_response_from_upscaling_(
                adapter_ ? adapter_->homogenized_section_response()
                         : UpscalingResult{});
        }
        response.tangent_mode_requested = tangent_mode_;

        if (tangent_mode_ == TangentComputationMode::ForceAdaptiveFiniteDifference ||
            tangent_mode_ ==
                TangentComputationMode::
                    ValidateCondensationAgainstAdaptiveFiniteDifference) {
            const auto control = current_control_vector_();
            auto fd = can_reuse_finite_difference_tangent_(control)
                ? cached_fd_response_
                : finite_difference_tangent_response_(response,
                                                       control,
                                                       h_pert);
            if (!can_reuse_finite_difference_tangent_(control)) {
                cached_fd_response_ = fd;
                cached_fd_control_ = control;
                cached_fd_step_count_ = step_count_;
                has_cached_fd_response_ = true;
            }
            if (tangent_mode_ ==
                TangentComputationMode::ForceAdaptiveFiniteDifference) {
                response.tangent = fd.tangent;
                response.tangent_scheme =
                    TangentLinearizationScheme::AdaptiveFiniteDifference;
                response.condensed_tangent_status =
                    CondensedTangentStatus::ForcedAdaptiveFiniteDifference;
                response.tangent_validation_status =
                    fd.tangent_validation_status;
                response.tangent_validation_relative_gap =
                    fd.tangent_validation_relative_gap;
                response.tangent_validation_max_column_gap =
                    fd.tangent_validation_max_column_gap;
                response.tangent_validation_column_gaps =
                    fd.tangent_validation_column_gaps;
            } else {
                populate_tangent_validation_diagnostics(
                    response,
                    fd.tangent,
                    finite_difference_settings_(h_pert));
                if (response.tangent_validation_status ==
                    TangentValidationStatus::Rejected) {
                    response.condensed_tangent_status =
                        CondensedTangentStatus::ValidationRejected;
                    response.tangent = fd.tangent;
                    response.tangent_scheme =
                        TangentLinearizationScheme::AdaptiveFiniteDifference;
                }
            }
            response.perturbation_sizes = fd.perturbation_sizes;
            response.tangent_column_valid = fd.tangent_column_valid;
            response.tangent_column_central = fd.tangent_column_central;
            response.failed_perturbations = fd.failed_perturbations;
        }
        complete_elastic_section_floor_(response, width, height);
        return response;
    }

    [[nodiscard]] Eigen::Matrix<double, 6, 6>
    section_tangent(double width, double height, double h_pert = 1.0e-6)
    {
        return section_response(width, height, h_pert).tangent;
    }

    [[nodiscard]] Eigen::Vector<double, 6>
    section_forces(double width, double height)
    {
        return section_response(width, height).forces;
    }

    void commit_state() { commit_trial_state(); }
    void revert_state() { restore_checkpoint(capture_checkpoint()); }

    void commit_trial_state()
    {
        committed_sample_ = trial_sample_;
        committed_response_ = last_response_;
        committed_material_history_ = trial_material_history_;
        committed_control_ = sample_control_vector_(committed_sample_);
        has_committed_sample_ = true;
        has_committed_material_history_ = has_trial_material_history_;
        has_committed_control_ = true;
    }

    void end_of_step(double /*time*/) { ++step_count_; }

    void set_auto_commit(bool enabled) noexcept { auto_commit_ = enabled; }

    [[nodiscard]] checkpoint_type capture_checkpoint() const
    {
        return checkpoint_type{
            .kin_A = kin_A_,
            .kin_B = kin_B_,
            .macro_state = macro_state_,
            .trial_sample = trial_sample_,
            .committed_sample = committed_sample_,
            .last_response = last_response_,
            .committed_response = committed_response_,
            .last_step = last_step_,
            .last_result = last_result_,
            .last_crack_summary = last_crack_summary_,
            .trial_material_history = trial_material_history_,
            .committed_material_history = committed_material_history_,
            .committed_control = committed_control_,
            .last_transition_control = last_transition_control_,
            .adaptive_transition_policy = adaptive_transition_policy_,
            .has_macro_state = has_macro_state_,
            .has_trial_sample = has_trial_sample_,
            .has_committed_sample = has_committed_sample_,
            .has_trial_material_history = has_trial_material_history_,
            .has_committed_material_history =
                has_committed_material_history_,
            .has_committed_control = has_committed_control_,
            .auto_commit = auto_commit_,
            .step_count = step_count_};
    }

    void restore_checkpoint(const checkpoint_type& checkpoint)
    {
        kin_A_ = checkpoint.kin_A;
        kin_B_ = checkpoint.kin_B;
        macro_state_ = checkpoint.macro_state;
        trial_sample_ = checkpoint.trial_sample;
        committed_sample_ = checkpoint.committed_sample;
        committed_response_ = checkpoint.committed_response;
        last_response_ = checkpoint.last_response;
        last_step_ = checkpoint.last_step;
        last_result_ = checkpoint.last_result;
        last_crack_summary_ = checkpoint.last_crack_summary;
        trial_material_history_ = checkpoint.trial_material_history;
        committed_material_history_ = checkpoint.committed_material_history;
        committed_control_ = checkpoint.committed_control;
        last_transition_control_ = checkpoint.last_transition_control;
        adaptive_transition_policy_ = checkpoint.adaptive_transition_policy;
        has_macro_state_ = checkpoint.has_macro_state;
        has_trial_sample_ = checkpoint.has_trial_sample;
        has_committed_sample_ = checkpoint.has_committed_sample;
        has_trial_material_history_ = checkpoint.has_trial_material_history;
        has_committed_material_history_ =
            checkpoint.has_committed_material_history;
        has_committed_control_ = checkpoint.has_committed_control;
        auto_commit_ = checkpoint.auto_commit;
        step_count_ = checkpoint.step_count;
        adapter_.reset();
        initialized_ = false;
        has_cached_fd_response_ = false;
        if ((has_trial_sample_ || has_committed_sample_) &&
            ensure_initialized_()) {
            const auto& sample =
                has_trial_sample_ ? trial_sample_ : committed_sample_;
            (void)adapter_->apply_macro_boundary_sample(sample);
            (void)adapter_->solve_current_pseudo_time_step(sample);
        }
    }

    [[nodiscard]] std::size_t parent_element_id() const noexcept
    {
        return parent_element_id_;
    }

    [[nodiscard]] int step_count() const noexcept { return step_count_; }

    [[nodiscard]] CrackSummary crack_summary() const noexcept
    {
        return last_crack_summary_;
    }

    [[nodiscard]] const SubModelSolverResult& last_solve_result() const noexcept
    {
        return last_result_;
    }

    [[nodiscard]] const ReducedRCManagedLocalBoundarySample&
    last_boundary_sample() const noexcept
    {
        return trial_sample_;
    }

    [[nodiscard]] const ReducedRCManagedLocalBoundarySample&
    committed_boundary_sample() const noexcept
    {
        return committed_sample_;
    }

    [[nodiscard]] const SectionHomogenizedResponse&
    last_section_response() const noexcept
    {
        return last_response_;
    }

    [[nodiscard]] bool has_committed_sample() const noexcept
    {
        return has_committed_sample_;
    }

    [[nodiscard]] const ReducedRCManagedLocalPatchSpec& patch() const noexcept
    {
        return patch_;
    }

    [[nodiscard]] const MaterialHistoryTransferPacket&
    trial_material_history_packet() const noexcept
    {
        return trial_material_history_;
    }

    [[nodiscard]] const MaterialHistoryTransferPacket&
    committed_material_history_packet() const noexcept
    {
        return committed_material_history_;
    }

    void configure_vtk_output(const std::filesystem::path& output_root)
    {
        vtk_output_dir_ =
            output_root /
            std::format("site_{:05}_element_{:05}",
                        patch_.site_index,
                        parent_element_id_);
        std::filesystem::create_directories(vtk_output_dir_);
        pvd_mesh_.emplace(
            (vtk_output_dir_ / "managed_xfem_mesh").string());
        pvd_gauss_.emplace(
            (vtk_output_dir_ / "managed_xfem_gauss").string());
        pvd_cracks_.emplace(
            (vtk_output_dir_ / "managed_xfem_cracks").string());
        vtk_output_configured_ = true;
    }

    [[nodiscard]] ReducedRCManagedXfemLocalVTKSnapshot write_vtk_snapshot(
        double time,
        int step_count,
        double min_abs_crack_opening = 0.0)
    {
        ReducedRCManagedXfemLocalVTKSnapshot snapshot{};
        if (!vtk_output_configured_) {
            snapshot.status_label = "managed_xfem_vtk_output_not_configured";
            return snapshot;
        }
        if (!ensure_initialized_()) {
            snapshot.status_label = "managed_xfem_model_not_initialized";
            return snapshot;
        }

        snapshot = adapter_->write_vtk_snapshot(
            vtk_output_dir_,
            time,
            step_count,
            min_abs_crack_opening);
        if (snapshot.written) {
            if (pvd_mesh_) {
                pvd_mesh_->add_timestep(time, snapshot.mesh_path);
            }
            if (pvd_gauss_ && !snapshot.gauss_path.empty()) {
                pvd_gauss_->add_timestep(time, snapshot.gauss_path);
            }
            if (pvd_cracks_ && !snapshot.cracks_path.empty()) {
                pvd_cracks_->add_timestep(time, snapshot.cracks_path);
            }
            write_vtk_collections_();
        }
        return snapshot;
    }

    [[nodiscard]] const std::filesystem::path& vtk_output_dir() const noexcept
    {
        return vtk_output_dir_;
    }

    void finalize() noexcept
    {
        write_vtk_collections_();
    }

    [[nodiscard]] static constexpr LocalModelTaxonomy local_model_taxonomy()
        noexcept
    {
        return {
            .discretization_kind =
                LocalModelDiscretizationKind::xfem_enriched_continuum,
            .fracture_representation_kind =
                LocalFractureRepresentationKind::strong_discontinuity_enrichment,
            .reinforcement_representation_kind =
                LocalReinforcementRepresentationKind::embedded_truss_line,
            .maturity_kind = LocalModelMaturityKind::promoted_baseline,
            .supports_discrete_crack_geometry = true,
            .requires_enriched_dofs = true,
            .requires_skeleton_trace_unknowns = false,
            .suitable_for_future_multiscale_local_model = true,
            .notes = "Managed XFEM local model promoted for seismic FE2."};
    }

private:
    [[nodiscard]] bool ensure_initialized_()
    {
        if (initialized_ && adapter_) {
            return true;
        }
        adapter_.emplace(options_);
        adapter_->set_vtk_output_profile(vtk_output_profile_);
        initialized_ = adapter_->initialize_managed_local_model(patch_);
        return initialized_;
    }

    void write_vtk_collections_() noexcept
    {
        try {
            if (pvd_mesh_) {
                pvd_mesh_->write();
            }
            if (pvd_gauss_) {
                pvd_gauss_->write();
            }
            if (pvd_cracks_) {
                pvd_cracks_->write();
            }
        } catch (...) {
            // Visualization must never invalidate a converged FE2 state.
        }
    }

    [[nodiscard]] static double lerp_(double a, double b, double z) noexcept
    {
        return (1.0 - z) * a + z * b;
    }

    [[nodiscard]] ReducedRCManagedLocalBoundarySample
    make_sample_(double time) const
    {
        return make_sample_from_control_(time, current_control_vector_());
    }

    [[nodiscard]] Eigen::Vector<double, 6> current_control_vector_() const
    {
        const double z = std::clamp(patch_.crack_z_over_l, 0.0, 1.0);
        Eigen::Vector<double, 6> control = Eigen::Vector<double, 6>::Zero();
        control[0] = lerp_(kin_A_.eps_0, kin_B_.eps_0, z);
        control[1] = lerp_(kin_A_.kappa_y, kin_B_.kappa_y, z);
        control[2] = lerp_(kin_A_.kappa_z, kin_B_.kappa_z, z);
        control[3] = lerp_(kin_A_.gamma_y, kin_B_.gamma_y, z);
        control[4] = lerp_(kin_A_.gamma_z, kin_B_.gamma_z, z);
        control[5] = lerp_(kin_A_.twist, kin_B_.twist, z);
        if (has_macro_state_ && macro_state_.strain.allFinite()) {
            // The macro bridge is the authoritative generalized state in the
            // two-way path.  The endpoint kinematics remain available for the
            // affine boundary reconstruction, but the tangent audit must be
            // centered at the actual section state injected into the local.
            control = macro_state_.strain;
        }
        return control;
    }

    [[nodiscard]] ReducedRCManagedLocalBoundarySample
    make_sample_from_control_(double time,
                              const Eigen::Vector<double, 6>& control) const
    {
        const double z = std::clamp(patch_.crack_z_over_l, 0.0, 1.0);
        ReducedRCStructuralReplaySample sample{};
        sample.site_index = parent_element_id_;
        sample.pseudo_time = time;
        sample.physical_time = time;
        sample.z_over_l = z;
        const double axial_strain = control[0];
        sample.curvature_y = control[1];
        sample.curvature_z = control[2];
        if (has_macro_state_) {
            sample.moment_y_mn_m = macro_state_.forces[1];
            sample.moment_z_mn_m = macro_state_.forces[2];
            sample.base_shear_mn =
                std::hypot(macro_state_.forces[3], macro_state_.forces[4]);
        }
        sample.damage_indicator =
            std::min(1.0, std::max(std::abs(sample.curvature_y),
                                   std::abs(sample.curvature_z)) / 0.010);
        return make_reduced_rc_managed_local_boundary_sample(
            sample, patch_, static_cast<std::size_t>(step_count_),
            axial_strain);
    }

    [[nodiscard]] static Eigen::Vector<double, 6> sample_control_vector_(
        const ReducedRCManagedLocalBoundarySample& sample) noexcept
    {
        Eigen::Vector<double, 6> control = Eigen::Vector<double, 6>::Zero();
        control[0] = sample.axial_strain;
        control[1] = sample.curvature_y;
        control[2] = sample.curvature_z;
        return control;
    }

    [[nodiscard]] HomogenizedTangentFiniteDifferenceSettings
    finite_difference_settings_(double h_pert) const noexcept
    {
        auto settings = fd_settings_;
        if (h_pert > 0.0 && std::isfinite(h_pert)) {
            settings.absolute_perturbation_floor =
                std::max(settings.absolute_perturbation_floor, h_pert);
        }
        return settings;
    }

    [[nodiscard]] SectionHomogenizedResponse evaluate_response_at_control_(
        double time,
        const Eigen::Vector<double, 6>& control)
    {
        SectionHomogenizedResponse response{};
        if (has_macro_state_) {
            macro_state_.strain = control;
        }
        trial_sample_ = make_sample_from_control_(time, control);
        has_trial_sample_ = true;
        if (!ensure_initialized_() ||
            !adapter_->apply_macro_boundary_sample(trial_sample_)) {
            response.status = ResponseStatus::SolveFailed;
            return response;
        }
        const auto step =
            adapter_->solve_current_pseudo_time_step(trial_sample_);
        response = make_section_response_from_upscaling_(
            adapter_->homogenized_section_response());
        if (!step.converged || step.hard_failure) {
            response.status = ResponseStatus::SolveFailed;
        }
        return response;
    }

    [[nodiscard]] bool can_reuse_finite_difference_tangent_(
        const Eigen::Vector<double, 6>& control) const noexcept
    {
        if (!has_cached_fd_response_) {
            return false;
        }
        if (cached_fd_step_count_ != step_count_) {
            return false;
        }
        const auto settings = finite_difference_settings_(0.0);
        const double denom = std::max({1.0,
                                       cached_fd_control_.norm(),
                                       control.norm()});
        const double gap = (control - cached_fd_control_).norm() / denom;
        return gap <= settings.reuse_control_relative_tolerance;
    }

    [[nodiscard]] SectionHomogenizedResponse finite_difference_tangent_response_(
        const SectionHomogenizedResponse& base,
        const Eigen::Vector<double, 6>& q0,
        double h_pert)
    {
        auto fd = base;
        fd.tangent.setZero();
        fd.tangent_scheme = TangentLinearizationScheme::AdaptiveFiniteDifference;
        fd.tangent_mode_requested = tangent_mode_;
        fd.condensed_tangent_status =
            CondensedTangentStatus::ForcedAdaptiveFiniteDifference;
        fd.tangent_validation_status = TangentValidationStatus::Accepted;

        const auto settings = finite_difference_settings_(h_pert);
        const auto perturbations =
            homogenized_tangent_perturbation_sizes(q0, settings);
        fd.perturbation_sizes = perturbations;

        const auto base_checkpoint = capture_checkpoint();
        const double time = has_trial_sample_
            ? trial_sample_.physical_time
            : static_cast<double>(step_count_);
        const auto f0 = base.forces;

        // The present managed XFEM boundary map controls axial strain and the
        // two bending curvatures directly.  Shear/twist remain explicit audit
        // gaps and are completed by the elastic floor after this routine.
        constexpr int supported_controls = 3;
        for (int j = 0; j < supported_controls; ++j) {
            const double h = perturbations[static_cast<std::size_t>(j)];
            Eigen::Vector<double, 6> qp = q0;
            qp[j] += h;

            restore_checkpoint(base_checkpoint);
            const auto plus = evaluate_response_at_control_(time, qp);
            if (plus.status == ResponseStatus::SolveFailed) {
                ++fd.failed_perturbations;
                continue;
            }

            bool used_central = false;
            Eigen::Vector<double, 6> column =
                (plus.forces - f0) / h;
            if (settings.scheme == HomogenizedFiniteDifferenceScheme::Central) {
                Eigen::Vector<double, 6> qm = q0;
                qm[j] -= h;
                restore_checkpoint(base_checkpoint);
                const auto minus = evaluate_response_at_control_(time, qm);
                if (minus.status != ResponseStatus::SolveFailed) {
                    column = (plus.forces - minus.forces) / (2.0 * h);
                    used_central = true;
                }
            }

            fd.tangent.col(j) = column;
            fd.tangent_column_valid[static_cast<std::size_t>(j)] = true;
            fd.tangent_column_central[static_cast<std::size_t>(j)] =
                used_central;
        }

        restore_checkpoint(base_checkpoint);
        populate_tangent_validation_diagnostics(fd, fd.tangent, settings);
        refresh_section_operator_diagnostics(fd);
        return fd;
    }

    [[nodiscard]] SectionHomogenizedResponse
    make_section_response_from_upscaling_(const UpscalingResult& up) const
    {
        SectionHomogenizedResponse response{};
        response.site.macro_element_id = parent_element_id_;
        response.site.section_gp = 0;
        response.status = up.converged ? ResponseStatus::Ok
                                       : ResponseStatus::NotReady;
        response.operator_used = HomogenizationOperator::BoundaryReaction;
        response.tangent_scheme = TangentLinearizationScheme::Unknown;
        response.condensed_tangent_status =
            CondensedTangentStatus::NotAttempted;
        response.forces_consistent_with_tangent = true;

        const auto copy_vec = [](const Eigen::VectorXd& src,
                                 Eigen::Vector<double, 6>& dst) {
            const auto n = std::min<std::size_t>(
                6, static_cast<std::size_t>(src.size()));
            for (std::size_t i = 0; i < n; ++i) {
                dst[static_cast<Eigen::Index>(i)] = src[static_cast<Eigen::Index>(i)];
            }
        };
        copy_vec(up.f_hom, response.forces);
        copy_vec(up.eps_ref, response.strain_ref);

        const auto rows = std::min<std::size_t>(
            6, static_cast<std::size_t>(up.D_hom.rows()));
        const auto cols = std::min<std::size_t>(
            6, static_cast<std::size_t>(up.D_hom.cols()));
        for (std::size_t i = 0; i < rows; ++i) {
            for (std::size_t j = 0; j < cols; ++j) {
                response.tangent(static_cast<Eigen::Index>(i),
                                 static_cast<Eigen::Index>(j)) =
                    up.D_hom(static_cast<Eigen::Index>(i),
                             static_cast<Eigen::Index>(j));
            }
        }
        refresh_section_operator_diagnostics(response);
        return response;
    }

    [[nodiscard]] MaterialHistoryTransferPacket
    make_section_material_history_packet_(
        const ReducedRCManagedLocalBoundarySample& sample,
        const SectionHomogenizedResponse& response) const
    {
        MaterialHistoryTransferPacket packet = committed_material_history_;
        if (packet.samples.empty()) {
            packet.direction = MaterialHistoryTransferDirection::MacroToLocal;
            packet.seed_policy =
                MaterialHistorySeedPolicy::SeedThenReplayIncrement;
            packet.source_label = "macro_structural_section_state";
            packet.target_label = "managed_xfem_local_section_state";
        }

        MaterialHistorySiteKey key{};
        key.site.macro_element_id = parent_element_id_;
        key.site.section_gp = response.site.section_gp;
        key.role = MaterialHistorySiteRole::SectionResultant;
        key.local_site_index = patch_.site_index;
        key.xi = sample.z_over_l;
        key.y = 0.0;
        key.z = sample.z_over_l;

        Eigen::VectorXd eta(6);
        eta << sample.axial_strain,
               sample.curvature_y,
               sample.curvature_z,
               0.0,
               0.0,
               0.0;
        Eigen::VectorXd q(6);
        for (int i = 0; i < 6; ++i) {
            q[i] = response.forces[i];
        }
        packet.samples.push_back(
            make_section_generalized_material_history_sample(
                key,
                std::move(eta),
                std::move(q),
                sample.pseudo_time,
                sample.physical_time));
        return packet;
    }

    void complete_elastic_section_floor_(SectionHomogenizedResponse& response,
                                         double width,
                                         double height) const
    {
        const double E = options_.concrete_elastic_modulus_mpa;
        const double nu = options_.concrete_poisson_ratio;
        const double G = E / (2.0 * (1.0 + nu));
        const double area = width * height;
        constexpr double kappa_s = 5.0 / 6.0;
        const double Iy = height * width * width * width / 12.0;
        const double Iz = width * height * height * height / 12.0;
        response.tangent(0, 0) =
            response.tangent(0, 0) != 0.0 ? response.tangent(0, 0)
                                          : E * area;
        response.tangent(1, 1) =
            response.tangent(1, 1) != 0.0 ? response.tangent(1, 1)
                                          : E * Iy;
        response.tangent(2, 2) =
            response.tangent(2, 2) != 0.0 ? response.tangent(2, 2)
                                          : E * Iz;
        response.tangent(3, 3) = kappa_s * G * area;
        response.tangent(4, 4) = kappa_s * G * area;
        response.tangent(5, 5) = G * (Iy + Iz);
        refresh_section_operator_diagnostics(response);
    }

    std::size_t parent_element_id_{0};
    ReducedRCManagedLocalPatchSpec patch_{};
    ReducedRCManagedXfemLocalModelAdapterOptions options_{};
    std::optional<ReducedRCManagedXfemLocalModelAdapter> adapter_{};
    SectionKinematics kin_A_{};
    SectionKinematics kin_B_{};
    MacroSectionState macro_state_{};
    ReducedRCManagedLocalBoundarySample trial_sample_{};
    ReducedRCManagedLocalBoundarySample committed_sample_{};
    SectionHomogenizedResponse last_response_{};
    SectionHomogenizedResponse committed_response_{};
    ReducedRCManagedLocalStepResult last_step_{};
    SubModelSolverResult last_result_{};
    CrackSummary last_crack_summary_{};
    bool initialized_{false};
    bool has_macro_state_{false};
    bool has_trial_sample_{false};
    bool has_committed_sample_{false};
    bool has_trial_material_history_{false};
    bool has_committed_material_history_{false};
    bool auto_commit_{true};
    TangentComputationMode tangent_mode_{
        TangentComputationMode::PreferLinearizedCondensation};
    HomogenizedTangentFiniteDifferenceSettings fd_settings_{};
    Eigen::Vector<double, 6> cached_fd_control_{
        Eigen::Vector<double, 6>::Zero()};
    Eigen::Vector<double, 6> committed_control_{
        Eigen::Vector<double, 6>::Zero()};
    SectionHomogenizedResponse cached_fd_response_{};
    int cached_fd_step_count_{-1};
    bool has_cached_fd_response_{false};
    MaterialHistoryTransferPacket trial_material_history_{};
    MaterialHistoryTransferPacket committed_material_history_{};
    ManagedXfemAdaptiveTransitionPolicy adaptive_transition_policy_{};
    ManagedXfemTransitionControl last_transition_control_{};
    int step_count_{0};
    bool has_committed_control_{false};
    bool vtk_output_configured_{false};
    LocalVTKOutputProfile vtk_output_profile_{LocalVTKOutputProfile::Debug};
    std::filesystem::path vtk_output_dir_{};
    std::optional<PVDWriter> pvd_mesh_{};
    std::optional<PVDWriter> pvd_gauss_{};
    std::optional<PVDWriter> pvd_cracks_{};
};

static_assert(LocalModelAdapter<ManagedXfemSubscaleEvolver>);

} // namespace fall_n

#endif // FALL_N_MANAGED_XFEM_SUBSCALE_EVOLVER_HH
