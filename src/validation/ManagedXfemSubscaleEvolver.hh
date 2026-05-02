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
#include <optional>
#include <utility>

#include "src/analysis/MultiscaleTypes.hh"
#include "src/reconstruction/FieldTransfer.hh"
#include "src/reconstruction/LocalModelAdapter.hh"
#include "src/reconstruction/LocalCrackData.hh"
#include "src/reconstruction/SubModelSolver.hh"
#include "src/validation/ReducedRCManagedXfemLocalModelAdapter.hh"

namespace fall_n {

class ManagedXfemSubscaleEvolver {
public:
    struct checkpoint_type {
        SectionKinematics kin_A{};
        SectionKinematics kin_B{};
        ReducedRCManagedLocalBoundarySample committed_sample{};
        SectionHomogenizedResponse committed_response{};
        bool has_committed_sample{false};
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
    }

    [[nodiscard]] SubModelSolverResult solve_step(double time)
    {
        SubModelSolverResult result{};
        result.stage = has_committed_sample_
            ? SubModelSolveStage::SubsequentFullStep
            : SubModelSolveStage::FirstSolveSingleStep;

        if (!ensure_initialized_()) {
            result.converged = false;
            result.failure_cause = SubModelFailureCause::FunctionDomain;
            last_result_ = result;
            return result;
        }

        trial_sample_ = make_sample_(time);
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
        last_crack_summary_.num_cracked_gps =
            step.max_damage_indicator > 0.0 ? 1 : 0;
        last_crack_summary_.total_cracks =
            step.max_damage_indicator > 0.0 ? 1 : 0;
        last_crack_summary_.damage_scalar_available = true;
        last_crack_summary_.max_damage_scalar = step.max_damage_indicator;

        if (result.converged && auto_commit_) {
            commit_trial_state();
        }
        last_result_ = result;
        return result;
    }

    [[nodiscard]] SectionHomogenizedResponse
    section_response(double width, double height, double h_pert = 1.0e-6)
    {
        (void)h_pert;
        auto response = last_response_;
        if (response.status == ResponseStatus::NotReady) {
            response = make_section_response_from_upscaling_(
                adapter_ ? adapter_->homogenized_section_response()
                         : UpscalingResult{});
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
        has_committed_sample_ = true;
    }

    void end_of_step(double /*time*/) { ++step_count_; }

    void set_auto_commit(bool enabled) noexcept { auto_commit_ = enabled; }

    [[nodiscard]] checkpoint_type capture_checkpoint() const
    {
        return checkpoint_type{
            .kin_A = kin_A_,
            .kin_B = kin_B_,
            .committed_sample = committed_sample_,
            .committed_response = committed_response_,
            .has_committed_sample = has_committed_sample_,
            .auto_commit = auto_commit_,
            .step_count = step_count_};
    }

    void restore_checkpoint(const checkpoint_type& checkpoint)
    {
        kin_A_ = checkpoint.kin_A;
        kin_B_ = checkpoint.kin_B;
        committed_sample_ = checkpoint.committed_sample;
        committed_response_ = checkpoint.committed_response;
        last_response_ = checkpoint.committed_response;
        has_committed_sample_ = checkpoint.has_committed_sample;
        auto_commit_ = checkpoint.auto_commit;
        step_count_ = checkpoint.step_count;
        adapter_.reset();
        initialized_ = false;
        if (has_committed_sample_ && ensure_initialized_()) {
            (void)adapter_->apply_macro_boundary_sample(committed_sample_);
            (void)adapter_->solve_current_pseudo_time_step(committed_sample_);
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

    void finalize() noexcept {}

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
        initialized_ = adapter_->initialize_managed_local_model(patch_);
        return initialized_;
    }

    [[nodiscard]] static double lerp_(double a, double b, double z) noexcept
    {
        return (1.0 - z) * a + z * b;
    }

    [[nodiscard]] ReducedRCManagedLocalBoundarySample
    make_sample_(double time) const
    {
        const double z = std::clamp(patch_.crack_z_over_l, 0.0, 1.0);
        ReducedRCStructuralReplaySample sample{};
        sample.site_index = parent_element_id_;
        sample.pseudo_time = time;
        sample.physical_time = time;
        sample.z_over_l = z;
        sample.curvature_y = lerp_(kin_A_.kappa_y, kin_B_.kappa_y, z);
        sample.curvature_z = lerp_(kin_A_.kappa_z, kin_B_.kappa_z, z);
        sample.damage_indicator =
            std::min(1.0, std::max(std::abs(sample.curvature_y),
                                   std::abs(sample.curvature_z)) / 0.010);
        return make_reduced_rc_managed_local_boundary_sample(
            sample, patch_, static_cast<std::size_t>(step_count_));
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
    ReducedRCManagedLocalBoundarySample trial_sample_{};
    ReducedRCManagedLocalBoundarySample committed_sample_{};
    SectionHomogenizedResponse last_response_{};
    SectionHomogenizedResponse committed_response_{};
    ReducedRCManagedLocalStepResult last_step_{};
    SubModelSolverResult last_result_{};
    CrackSummary last_crack_summary_{};
    bool initialized_{false};
    bool has_committed_sample_{false};
    bool auto_commit_{true};
    int step_count_{0};
};

static_assert(LocalModelAdapter<ManagedXfemSubscaleEvolver>);

} // namespace fall_n

#endif // FALL_N_MANAGED_XFEM_SUBSCALE_EVOLVER_HH
