#ifndef FALL_N_SRC_VALIDATION_SEISMIC_FE2_LOCAL_MODEL_VARIANT_HH
#define FALL_N_SRC_VALIDATION_SEISMIC_FE2_LOCAL_MODEL_VARIANT_HH

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

#include "src/reconstruction/NonlinearSubModelEvolver.hh"
#include "src/validation/ManagedXfemSubscaleEvolver.hh"

namespace fall_n {

enum class SeismicFE2LocalFamily {
    managed_xfem,
    continuum_kobathe_hex20,
    continuum_kobathe_hex27
};

[[nodiscard]] inline std::string_view to_string(
    SeismicFE2LocalFamily family) noexcept
{
    switch (family) {
        case SeismicFE2LocalFamily::managed_xfem:
            return "managed-xfem";
        case SeismicFE2LocalFamily::continuum_kobathe_hex20:
            return "continuum-kobathe-hex20";
        case SeismicFE2LocalFamily::continuum_kobathe_hex27:
            return "continuum-kobathe-hex27";
    }
    return "unknown";
}

class SeismicFE2LocalModel {
public:
    using ManagedCheckpoint = ManagedXfemSubscaleEvolver::checkpoint_type;
    using ContinuumCheckpoint = NonlinearSubModelEvolver::checkpoint_type;

    struct checkpoint_type {
        std::variant<ManagedCheckpoint, ContinuumCheckpoint> local{};
        SectionHomogenizedResponse last_response{};
        ReducedRCManagedLocalBoundarySample audit_boundary_sample{};
        ReducedRCManagedLocalPatchSpec audit_patch{};
        ManagedXfemTransitionControl audit_transition_control{};
        bool has_last_response{false};
        bool has_audit_sample{false};
    };

private:
    std::variant<ManagedXfemSubscaleEvolver, NonlinearSubModelEvolver> model_;
    SeismicFE2LocalFamily family_{SeismicFE2LocalFamily::managed_xfem};
    SectionHomogenizedResponse last_response_{};
    bool has_last_response_{false};
    ReducedRCManagedLocalBoundarySample audit_boundary_sample_{};
    ReducedRCManagedLocalPatchSpec audit_patch_{};
    ManagedXfemTransitionControl audit_transition_control_{};
    bool has_audit_sample_{false};

public:
    explicit SeismicFE2LocalModel(ManagedXfemSubscaleEvolver model)
        : model_{std::move(model)}
        , family_{SeismicFE2LocalFamily::managed_xfem}
    {}

    SeismicFE2LocalModel(NonlinearSubModelEvolver model,
                         SeismicFE2LocalFamily family)
        : model_{std::move(model)}
        , family_{family}
    {}

    SeismicFE2LocalModel(SeismicFE2LocalModel&&) noexcept = default;
    SeismicFE2LocalModel& operator=(SeismicFE2LocalModel&&) noexcept =
        default;
    SeismicFE2LocalModel(const SeismicFE2LocalModel&) = delete;
    SeismicFE2LocalModel& operator=(const SeismicFE2LocalModel&) = delete;

    [[nodiscard]] SeismicFE2LocalFamily family() const noexcept
    {
        return family_;
    }

    void update_kinematics(const SectionKinematics& kin_A,
                           const SectionKinematics& kin_B)
    {
        audit_boundary_sample_.curvature_y =
            0.5 * (kin_A.kappa_y + kin_B.kappa_y);
        audit_boundary_sample_.curvature_z =
            0.5 * (kin_A.kappa_z + kin_B.kappa_z);
        audit_boundary_sample_.axial_strain =
            0.5 * (kin_A.eps_0 + kin_B.eps_0);
        audit_boundary_sample_.imposed_rotation_y_rad =
            kin_B.kappa_y - kin_A.kappa_y;
        audit_boundary_sample_.imposed_rotation_z_rad =
            kin_B.kappa_z - kin_A.kappa_z;
        has_audit_sample_ = true;
        std::visit([&](auto& model) {
            model.update_kinematics(kin_A, kin_B);
        }, model_);
    }

    void update_macro_section_state(const MacroSectionState& state)
    {
        audit_boundary_sample_.site_index = state.site.macro_element_id;
        audit_boundary_sample_.z_over_l = std::clamp(
            0.5 * (state.site.xi + 1.0), 0.0, 1.0);
        audit_boundary_sample_.axial_strain = state.strain[0];
        audit_boundary_sample_.curvature_y = state.strain[1];
        audit_boundary_sample_.curvature_z = state.strain[2];
        audit_boundary_sample_.imposed_rotation_y_rad = state.strain[1];
        audit_boundary_sample_.imposed_rotation_z_rad = state.strain[2];
        audit_boundary_sample_.macro_moment_y_mn_m = state.forces[1];
        audit_boundary_sample_.macro_moment_z_mn_m = state.forces[2];
        audit_boundary_sample_.macro_base_shear_mn =
            std::hypot(state.forces[3], state.forces[4]);
        has_audit_sample_ = true;
        std::visit([&](auto& model) {
            if constexpr (requires { model.update_macro_section_state(state); }) {
                model.update_macro_section_state(state);
            }
        }, model_);
    }

    void set_tangent_computation_mode(TangentComputationMode mode) noexcept
    {
        std::visit([&](auto& model) {
            model.set_tangent_computation_mode(mode);
        }, model_);
    }

    void set_finite_difference_tangent_settings(
        HomogenizedTangentFiniteDifferenceSettings settings) noexcept
    {
        std::visit([&](auto& model) {
            if constexpr (requires {
                              model.set_finite_difference_tangent_settings(
                                  settings);
                          }) {
                model.set_finite_difference_tangent_settings(settings);
            }
        }, model_);
    }

    [[nodiscard]] SubModelSolverResult solve_step(double time)
    {
        audit_boundary_sample_.pseudo_time = time;
        audit_boundary_sample_.physical_time = time;
        audit_boundary_sample_.sample_index =
            static_cast<std::size_t>(std::max(step_count(), 0));
        return std::visit([&](auto& model) {
            return model.solve_step(time);
        }, model_);
    }

    [[nodiscard]] SectionHomogenizedResponse
    section_response(double width, double height, double h_pert = 1.0e-6)
    {
        last_response_ = std::visit([&](auto& model) {
            return model.section_response(width, height, h_pert);
        }, model_);
        has_last_response_ = true;
        return last_response_;
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

    [[nodiscard]] SectionHomogenizedResponse last_section_response() const
    {
        return has_last_response_ ? last_response_
                                  : SectionHomogenizedResponse{};
    }

    [[nodiscard]] const SubModelSolverResult& last_solve_result() const noexcept
    {
        return std::visit([](const auto& model)
            -> const SubModelSolverResult& {
            return model.last_solve_result();
        }, model_);
    }

    [[nodiscard]] const ReducedRCManagedLocalBoundarySample&
    last_boundary_sample() const noexcept
    {
        return std::visit([&](const auto& model)
            -> const ReducedRCManagedLocalBoundarySample& {
            if constexpr (requires { model.last_boundary_sample(); }) {
                return model.last_boundary_sample();
            } else {
                return audit_boundary_sample_;
            }
        }, model_);
    }

    [[nodiscard]] bool has_committed_sample() const noexcept
    {
        return std::visit([&](const auto& model) {
            if constexpr (requires { model.has_committed_sample(); }) {
                return model.has_committed_sample();
            } else {
                return has_audit_sample_;
            }
        }, model_);
    }

    [[nodiscard]] const ReducedRCManagedLocalPatchSpec& patch() const noexcept
    {
        return std::visit([&](const auto& model)
            -> const ReducedRCManagedLocalPatchSpec& {
            if constexpr (requires { model.patch(); }) {
                return model.patch();
            } else {
                return audit_patch_;
            }
        }, model_);
    }

    [[nodiscard]] const ManagedXfemTransitionControl&
    last_transition_control() const noexcept
    {
        return std::visit([&](const auto& model)
            -> const ManagedXfemTransitionControl& {
            if constexpr (requires { model.last_transition_control(); }) {
                return model.last_transition_control();
            } else {
                return audit_transition_control_;
            }
        }, model_);
    }

    void commit_state()
    {
        std::visit([](auto& model) { model.commit_state(); }, model_);
    }

    void revert_state()
    {
        std::visit([](auto& model) { model.revert_state(); }, model_);
    }

    void commit_trial_state()
    {
        std::visit([](auto& model) { model.commit_trial_state(); }, model_);
    }

    void end_of_step(double time)
    {
        std::visit([&](auto& model) { model.end_of_step(time); }, model_);
    }

    void set_auto_commit(bool enabled) noexcept
    {
        std::visit([&](auto& model) { model.set_auto_commit(enabled); },
                   model_);
    }

    [[nodiscard]] checkpoint_type capture_checkpoint() const
    {
        checkpoint_type checkpoint{};
        checkpoint.last_response = last_response_;
        checkpoint.has_last_response = has_last_response_;
        checkpoint.audit_boundary_sample = audit_boundary_sample_;
        checkpoint.audit_patch = audit_patch_;
        checkpoint.audit_transition_control = audit_transition_control_;
        checkpoint.has_audit_sample = has_audit_sample_;
        std::visit([&](const auto& model) {
            checkpoint.local = model.capture_checkpoint();
        }, model_);
        return checkpoint;
    }

    void restore_checkpoint(const checkpoint_type& checkpoint)
    {
        last_response_ = checkpoint.last_response;
        has_last_response_ = checkpoint.has_last_response;
        audit_boundary_sample_ = checkpoint.audit_boundary_sample;
        audit_patch_ = checkpoint.audit_patch;
        audit_transition_control_ = checkpoint.audit_transition_control;
        has_audit_sample_ = checkpoint.has_audit_sample;
        std::visit([&](auto& model) {
            using ModelT = std::remove_cvref_t<decltype(model)>;
            using CheckpointT = typename ModelT::checkpoint_type;
            if (const auto* cp = std::get_if<CheckpointT>(
                    &checkpoint.local))
            {
                model.restore_checkpoint(*cp);
            }
        }, model_);
    }

    [[nodiscard]] std::size_t parent_element_id() const noexcept
    {
        return std::visit([](const auto& model) {
            return model.parent_element_id();
        }, model_);
    }

    [[nodiscard]] int step_count() const noexcept
    {
        return std::visit([](const auto& model) {
            return model.step_count();
        }, model_);
    }

    [[nodiscard]] CrackSummary crack_summary() const noexcept
    {
        return std::visit([](const auto& model) {
            return model.crack_summary();
        }, model_);
    }

    [[nodiscard]] CrackSummary last_attempted_crack_summary() const noexcept
    {
        return std::visit([](const auto& model) {
            if constexpr (requires {
                              model.last_attempted_crack_summary();
                          }) {
                return model.last_attempted_crack_summary();
            } else {
                return model.crack_summary();
            }
        }, model_);
    }

    void configure_vtk_output(const std::filesystem::path& root)
    {
        std::visit([&](auto& model) {
            if constexpr (requires { model.configure_vtk_output(root); }) {
                model.configure_vtk_output(root);
            }
        }, model_);
    }

    [[nodiscard]] ReducedRCManagedXfemLocalVTKSnapshot write_vtk_snapshot(
        double time,
        int step,
        double visual_scale = 0.0)
    {
        return std::visit([&](auto& model) {
            return model.write_vtk_snapshot(time, step, visual_scale);
        }, model_);
    }

    void finalize()
    {
        std::visit([](auto& model) { model.finalize(); }, model_);
    }
};

} // namespace fall_n

#endif // FALL_N_SRC_VALIDATION_SEISMIC_FE2_LOCAL_MODEL_VARIANT_HH
