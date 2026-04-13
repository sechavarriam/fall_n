#ifndef FALL_N_SRC_RECONSTRUCTION_AFFINE_SECTION_SUBSCALE_MODEL_HH
#define FALL_N_SRC_RECONSTRUCTION_AFFINE_SECTION_SUBSCALE_MODEL_HH

#include <cstddef>

#include <Eigen/Dense>

#include "LocalModelAdapter.hh"

namespace fall_n {

struct AffineSectionSubscaleSolveResult {
    bool converged{true};
};

class AffineSectionSubscaleModel {
public:
    using driving_state_type = SectionSubproblemDrivingState;
    using effective_operator_type = SectionHomogenizedResponse;

    struct checkpoint_type {
        SectionSubproblemDrivingState applied_driving{};
        SectionSubproblemDrivingState committed_driving{};
        SectionHomogenizedResponse trial_response{};
        SectionHomogenizedResponse committed_response{};
        bool has_trial{false};
        bool auto_commit{true};
        int solve_calls{0};
        int commit_trial_calls{0};
        int end_of_step_calls{0};
    };

private:
    std::size_t parent_element_id_{0};
    Eigen::Matrix<double, 6, 6> tangent_{
        Eigen::Matrix<double, 6, 6>::Identity()};
    Eigen::Vector<double, 6> bias_{Eigen::Vector<double, 6>::Zero()};

    SectionSubproblemDrivingState applied_driving_{};
    SectionSubproblemDrivingState committed_driving_{};
    SectionHomogenizedResponse trial_response_{};
    SectionHomogenizedResponse committed_response_{};

    bool auto_commit_{true};
    bool has_trial_{false};
    int solve_calls_{0};
    int commit_trial_calls_{0};
    int end_of_step_calls_{0};

    [[nodiscard]] static Eigen::Vector<double, 6>
    generalized_strain_(const SectionKinematics& kin)
    {
        Eigen::Vector<double, 6> e;
        e << kin.eps_0,
             kin.kappa_y,
             kin.kappa_z,
             kin.gamma_y,
             kin.gamma_z,
             kin.twist;
        return e;
    }

    [[nodiscard]] static Eigen::Vector<double, 6>
    reduced_driving_strain_(const SectionSubproblemDrivingState& driving)
    {
        return 0.5 * (
            generalized_strain_(driving.face_a)
          + generalized_strain_(driving.face_b));
    }

    [[nodiscard]] SectionHomogenizedResponse
    build_response_(const SectionSubproblemDrivingState& driving) const
    {
        SectionHomogenizedResponse response;
        response.strain_ref = reduced_driving_strain_(driving);
        response.tangent = tangent_;
        response.forces = bias_ + tangent_ * response.strain_ref;
        response.status = ResponseStatus::Ok;
        response.operator_used = HomogenizationOperator::VolumeAverage;
        response.regularization = RegularizationPolicyKind::None;
        response.forces_consistent_with_tangent = true;
        response.tangent_scheme = TangentLinearizationScheme::Unknown;
        response.tangent_mode_requested =
            TangentComputationMode::PreferLinearizedCondensation;
        response.condensed_tangent_status =
            CondensedTangentStatus::NotAttempted;
        response.tangent_validation_status =
            TangentValidationStatus::NotRequested;
        response.tangent_validation_norm =
            TangentValidationNormKind::StateWeightedFrobenius;
        response.tangent_column_valid = {true, true, true, true, true, true};
        response.tangent_column_central = {true, true, true, true, true, true};
        refresh_section_operator_diagnostics(response);
        return response;
    }

    [[nodiscard]] const SectionHomogenizedResponse& active_response_() const
    {
        return has_trial_ ? trial_response_ : committed_response_;
    }

public:
    AffineSectionSubscaleModel() = default;

    explicit AffineSectionSubscaleModel(
        std::size_t parent_element_id,
        const Eigen::Matrix<double, 6, 6>& tangent,
        const Eigen::Vector<double, 6>& bias =
            Eigen::Vector<double, 6>::Zero())
        : parent_element_id_{parent_element_id}
        , tangent_{tangent}
        , bias_{bias}
        , committed_response_{build_response_(committed_driving_)}
    {}

    void update_kinematics(const SectionKinematics& kin_A,
                           const SectionKinematics& kin_B)
    {
        applied_driving_.face_a = kin_A;
        applied_driving_.face_b = kin_B;
    }

    void apply_driving_state(const SectionSubproblemDrivingState& driving)
    {
        update_kinematics(driving.face_a, driving.face_b);
    }

    [[nodiscard]] AffineSectionSubscaleSolveResult solve_step(double)
    {
        ++solve_calls_;
        trial_response_ = build_response_(applied_driving_);
        has_trial_ = true;
        if (auto_commit_) {
            commit_trial_state();
        }
        return {.converged = true};
    }

    [[nodiscard]] Eigen::Matrix<double, 6, 6>
    section_tangent(double, double, double) const
    {
        return active_response_().tangent;
    }

    [[nodiscard]] Eigen::Vector<double, 6>
    section_forces(double, double) const
    {
        return active_response_().forces;
    }

    [[nodiscard]] SectionHomogenizedResponse
    section_response(double, double, double) const
    {
        return active_response_();
    }

    [[nodiscard]] SectionHomogenizedResponse
    effective_operator(const SectionEffectiveOperatorRequest&) const
    {
        return active_response_();
    }

    void commit_state()
    {
        committed_response_ = active_response_();
        committed_driving_ = applied_driving_;
        has_trial_ = false;
    }

    void revert_state()
    {
        applied_driving_ = committed_driving_;
        trial_response_ = committed_response_;
        has_trial_ = false;
    }

    void commit_trial_state()
    {
        committed_response_ = trial_response_;
        committed_driving_ = applied_driving_;
        has_trial_ = false;
        ++commit_trial_calls_;
    }

    void end_of_step(double)
    {
        ++end_of_step_calls_;
    }

    void set_auto_commit(bool enabled)
    {
        auto_commit_ = enabled;
    }

    [[nodiscard]] checkpoint_type capture_checkpoint() const
    {
        return {
            applied_driving_,
            committed_driving_,
            trial_response_,
            committed_response_,
            has_trial_,
            auto_commit_,
            solve_calls_,
            commit_trial_calls_,
            end_of_step_calls_
        };
    }

    void restore_checkpoint(const checkpoint_type& checkpoint)
    {
        applied_driving_ = checkpoint.applied_driving;
        committed_driving_ = checkpoint.committed_driving;
        trial_response_ = checkpoint.trial_response;
        committed_response_ = checkpoint.committed_response;
        has_trial_ = checkpoint.has_trial;
        auto_commit_ = checkpoint.auto_commit;
        solve_calls_ = checkpoint.solve_calls;
        commit_trial_calls_ = checkpoint.commit_trial_calls;
        end_of_step_calls_ = checkpoint.end_of_step_calls;
    }

    [[nodiscard]] std::size_t parent_element_id() const
    {
        return parent_element_id_;
    }

    [[nodiscard]] int solve_calls() const noexcept { return solve_calls_; }
    [[nodiscard]] int commit_trial_calls() const noexcept
    {
        return commit_trial_calls_;
    }
    [[nodiscard]] int end_of_step_calls() const noexcept
    {
        return end_of_step_calls_;
    }
};

static_assert(
    RequestedSubscaleModel<
        AffineSectionSubscaleModel,
        SectionSubproblemDrivingState,
        SectionEffectiveOperatorRequest,
        SectionHomogenizedResponse>,
    "AffineSectionSubscaleModel must satisfy the generic requested subscale "
    "contract");

static_assert(
    LocalModelAdapter<AffineSectionSubscaleModel>,
    "AffineSectionSubscaleModel must satisfy the current FE2 section-local "
    "contract");

} // namespace fall_n

#endif // FALL_N_SRC_RECONSTRUCTION_AFFINE_SECTION_SUBSCALE_MODEL_HH
