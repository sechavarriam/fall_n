#ifndef FALL_N_SRC_RECONSTRUCTION_BOUNDARY_REACTION_HOMOGENIZER_HH
#define FALL_N_SRC_RECONSTRUCTION_BOUNDARY_REACTION_HOMOGENIZER_HH

#include <algorithm>
#include <array>
#include <limits>

#include <Eigen/Dense>
#include <petsc.h>

#include "../analysis/MultiscaleTypes.hh"
#include "LocalBoundaryConditionApplicator.hh"
#include "PersistentLocalStateOps.hh"

namespace fall_n {

template <typename ModelT, typename SubModelT>
class BoundaryReactionHomogenizer {
public:
    struct TangentDiagnostics {
        Eigen::Matrix<double, 6, 6> tangent{
            Eigen::Matrix<double, 6, 6>::Zero()};
        std::array<double, 6> perturbation_sizes{};
        std::array<bool, 6> column_valid{};
        std::array<bool, 6> column_central{};
        double symmetry_error_before_regularization{0.0};
        int failed_perturbations{0};
        bool tangent_regularized{false};
    };

private:
    ModelT* model_{nullptr};
    SubModelT* sub_{nullptr};
    Vec displacement_{nullptr};
    Vec displacement_work_{nullptr};
    Vec imposed_work_{nullptr};
    SNES snes_{nullptr};
    RegularizationPolicyKind regularization_policy_{
        RegularizationPolicyKind::DiagonalFloor};
    double diagonal_floor_{1.0};
    LocalBoundaryConditionApplicator<ModelT, SubModelT> bc_applicator_{};
    PersistentLocalStateOps<ModelT> state_ops_{};

    [[nodiscard]] static Eigen::Vector<double, 6>
    relative_generalized_dofs_(const SectionKinematics& kin_A,
                               const SectionKinematics& kin_B)
    {
        Eigen::Vector<double, 6> q = Eigen::Vector<double, 6>::Zero();
        q << kin_B.u_local[0] - kin_A.u_local[0],
             kin_B.theta_local[1] - kin_A.theta_local[1],
             kin_B.theta_local[2] - kin_A.theta_local[2],
             kin_B.u_local[1] - kin_A.u_local[1],
             kin_B.u_local[2] - kin_A.u_local[2],
             kin_B.theta_local[0] - kin_A.theta_local[0];
        return q;
    }

    [[nodiscard]] static double characteristic_length_(
        const SectionKinematics& kin_A,
        const SectionKinematics& kin_B)
    {
        return std::max(
            (kin_B.centroid - kin_A.centroid).norm(),
            std::numeric_limits<double>::epsilon());
    }

    [[nodiscard]] static double adaptive_generalized_step_(
        int component,
        const Eigen::Vector<double, 6>& reference_dofs,
        double characteristic_length,
        double requested_h_pert)
    {
        const double base_physical =
            std::max(requested_h_pert * characteristic_length,
                     1.0e-9 * std::max(1.0, characteristic_length));
        const double scaled_physical =
            1.0e-3 * std::max(std::abs(reference_dofs[component]),
                              base_physical);
        return std::max(base_physical, scaled_physical)
             / characteristic_length;
    }

    static void apply_generalized_perturbation_(SectionKinematics& kin_B,
                                                int component,
                                                double delta_dof)
    {
        switch (component) {
            case 0: kin_B.u_local[0] += delta_dof; break;
            case 1: kin_B.theta_local[1] += delta_dof; break;
            case 2: kin_B.theta_local[2] += delta_dof; break;
            case 3: kin_B.u_local[1] += delta_dof; break;
            case 4: kin_B.u_local[2] += delta_dof; break;
            case 5: kin_B.theta_local[0] += delta_dof; break;
            default: break;
        }
    }

    void restore_tangent_baseline_state_() const
    {
        if (!model_) {
            return;
        }

        state_ops_.revert_state();
        VecCopy(displacement_work_, displacement_);
        VecCopy(imposed_work_, model_->imposed_solution());
        state_ops_.sync_state_vector();
    }

public:
    BoundaryReactionHomogenizer() = default;

    BoundaryReactionHomogenizer(
        ModelT* model,
        SubModelT* sub,
        Vec displacement,
        Vec displacement_work,
        Vec imposed_work,
        SNES snes,
        RegularizationPolicyKind regularization_policy,
        double diagonal_floor,
        LocalBoundaryConditionApplicator<ModelT, SubModelT> bc_applicator,
        PersistentLocalStateOps<ModelT> state_ops) noexcept
        : model_{model}
        , sub_{sub}
        , displacement_{displacement}
        , displacement_work_{displacement_work}
        , imposed_work_{imposed_work}
        , snes_{snes}
        , regularization_policy_{regularization_policy}
        , diagonal_floor_{diagonal_floor}
        , bc_applicator_{bc_applicator}
        , state_ops_{state_ops}
    {}

    [[nodiscard]] Eigen::Vector<double, 6> section_forces_from_reactions() const
    {
        if (!model_ || !sub_ || !displacement_) {
            return Eigen::Vector<double, 6>::Zero();
        }

        DM dm = model_->get_plex();

        Vec u_loc;
        DMGetLocalVector(dm, &u_loc);
        VecSet(u_loc, 0.0);
        DMGlobalToLocal(dm, displacement_, INSERT_VALUES, u_loc);
        VecAXPY(u_loc, 1.0, model_->imposed_solution());

        Vec f_loc;
        DMGetLocalVector(dm, &f_loc);
        VecSet(f_loc, 0.0);
        for (auto& elem : model_->elements()) {
            elem.compute_internal_forces(u_loc, f_loc);
        }

        DMRestoreLocalVector(dm, &u_loc);

        PetscSection sec;
        DMGetLocalSection(dm, &sec);

        const Eigen::Matrix3d& rotation = sub_->kin_B.R;
        const Eigen::Vector3d& centroid = sub_->kin_B.centroid;

        Eigen::Vector3d F_sum = Eigen::Vector3d::Zero();
        Eigen::Vector3d M_sum = Eigen::Vector3d::Zero();

        const PetscScalar* f_arr;
        VecGetArrayRead(f_loc, &f_arr);

        for (const auto& [nid, ignored_u] : sub_->bc_max_z) {
            (void)ignored_u;
            const PetscInt plex_pt = sub_->domain.node(nid).sieve_id.value();
            PetscInt off;
            PetscSectionGetOffset(sec, plex_pt, &off);

            const Eigen::Vector3d f_g{f_arr[off], f_arr[off + 1], f_arr[off + 2]};

            const auto& nd = sub_->domain.node(nid);
            const Eigen::Vector3d pos{nd.coord(0), nd.coord(1), nd.coord(2)};
            const Eigen::Vector3d r = pos - centroid;

            F_sum += f_g;
            M_sum += r.cross(f_g);
        }

        VecRestoreArrayRead(f_loc, &f_arr);
        DMRestoreLocalVector(dm, &f_loc);

        const Eigen::Vector3d F_local = rotation * F_sum;
        const Eigen::Vector3d M_local = rotation * M_sum;

        Eigen::Vector<double, 6> section_forces;
        section_forces << F_local[0], M_local[1], M_local[2],
            F_local[1], F_local[2], M_local[0];
        return section_forces;
    }

    [[nodiscard]] TangentDiagnostics compute_tangent(double h_pert) const
    {
        TangentDiagnostics diagnostics;
        if (!model_ || !sub_ || !displacement_ || !displacement_work_
            || !imposed_work_ || !snes_)
        {
            return diagnostics;
        }

        const Eigen::Vector<double, 6> s0_vec = section_forces_from_reactions();

        VecCopy(displacement_, displacement_work_);
        VecCopy(model_->imposed_solution(), imposed_work_);

        const SectionKinematics kin_A_orig = sub_->kin_A;
        const SectionKinematics kin_B_orig = sub_->kin_B;
        const double beam_length = characteristic_length_(kin_A_orig, kin_B_orig);
        const Eigen::Vector<double, 6> reference_dofs =
            relative_generalized_dofs_(kin_A_orig, kin_B_orig);

        for (int j = 0; j < 6; ++j) {
            const double h_j =
                adaptive_generalized_step_(j, reference_dofs, beam_length, h_pert);
            const double delta_q = h_j * beam_length;
            diagnostics.perturbation_sizes[static_cast<std::size_t>(j)] = h_j;

            bool plus_ok = false;
            bool minus_ok = false;
            Eigen::Vector<double, 6> s_plus = Eigen::Vector<double, 6>::Zero();
            Eigen::Vector<double, 6> s_minus = Eigen::Vector<double, 6>::Zero();

            for (int sign : {+1, -1}) {
                SectionKinematics kin_B_p = kin_B_orig;
                apply_generalized_perturbation_(
                    kin_B_p, j, static_cast<double>(sign) * delta_q);

                bc_applicator_.update_kinematics(kin_A_orig, kin_B_p);
                bc_applicator_.write_imposed_values();

                SNESSolve(snes_, nullptr, displacement_);

                SNESConvergedReason reason;
                SNESGetConvergedReason(snes_, &reason);

                if (reason > 0) {
                    if (sign > 0) {
                        s_plus = section_forces_from_reactions();
                        plus_ok = true;
                    } else {
                        s_minus = section_forces_from_reactions();
                        minus_ok = true;
                    }
                } else {
                    ++diagnostics.failed_perturbations;
                }

                restore_tangent_baseline_state_();
            }

            if (plus_ok && minus_ok) {
                diagnostics.tangent.col(j) = (s_plus - s_minus) / (2.0 * h_j);
                diagnostics.column_valid[static_cast<std::size_t>(j)] = true;
                diagnostics.column_central[static_cast<std::size_t>(j)] = true;
            } else if (plus_ok) {
                diagnostics.tangent.col(j) = (s_plus - s0_vec) / h_j;
                diagnostics.column_valid[static_cast<std::size_t>(j)] = true;
            } else if (minus_ok) {
                diagnostics.tangent.col(j) = (s0_vec - s_minus) / h_j;
                diagnostics.column_valid[static_cast<std::size_t>(j)] = true;
            }
        }

        bc_applicator_.update_kinematics(kin_A_orig, kin_B_orig);
        VecCopy(displacement_work_, displacement_);
        bc_applicator_.write_imposed_values();
        state_ops_.sync_state_vector();

        diagnostics.symmetry_error_before_regularization =
            (diagnostics.tangent - diagnostics.tangent.transpose()).norm();

        diagnostics.tangent =
            0.5 * (diagnostics.tangent + diagnostics.tangent.transpose());

        const auto tangent_before_regularization = diagnostics.tangent;

        if (regularization_policy_ == RegularizationPolicyKind::SPDProjection) {
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> eig{
                diagnostics.tangent};
            if (eig.info() == Eigen::Success) {
                auto evals = eig.eigenvalues();
                for (int i = 0; i < evals.size(); ++i) {
                    evals[i] = std::max(evals[i], diagonal_floor_);
                }
                diagnostics.tangent =
                    eig.eigenvectors() * evals.asDiagonal()
                    * eig.eigenvectors().transpose();
            }
        } else if (regularization_policy_
                   == RegularizationPolicyKind::DiagonalFloor)
        {
            for (int j = 0; j < 6; ++j) {
                if (diagnostics.tangent(j, j) < diagonal_floor_) {
                    diagnostics.tangent(j, j) = diagonal_floor_;
                }
            }
        }

        diagnostics.tangent_regularized =
            (diagnostics.tangent - tangent_before_regularization).norm()
            > 1.0e-12;

        return diagnostics;
    }

    [[nodiscard]] Eigen::Matrix<double, 6, 6>
    compute_homogenized_tangent(double h_pert = 1.0e-6) const
    {
        return compute_tangent(h_pert).tangent;
    }

    [[nodiscard]] Eigen::Vector<double, 6> compute_homogenized_forces() const
    {
        return section_forces_from_reactions();
    }

    [[nodiscard]] SectionHomogenizedResponse
    section_response(double h_pert = 1.0e-6) const
    {
        SectionHomogenizedResponse response;
        response.operator_used = HomogenizationOperator::BoundaryReaction;
        response.regularization = regularization_policy_;
        response.diagonal_floor = diagonal_floor_;

        if (!model_ || !sub_ || !displacement_) {
            response.status = ResponseStatus::NotReady;
            return response;
        }

        response.forces = compute_homogenized_forces();
        const auto diagnostics = compute_tangent(h_pert);
        response.tangent = diagnostics.tangent;
        response.tangent_symmetry_error =
            diagnostics.symmetry_error_before_regularization;
        response.tangent_regularized = diagnostics.tangent_regularized;
        response.tangent_scheme =
            TangentLinearizationScheme::AdaptiveFiniteDifference;
        response.perturbation_sizes = diagnostics.perturbation_sizes;
        response.tangent_column_valid = diagnostics.column_valid;
        response.tangent_column_central = diagnostics.column_central;
        response.failed_perturbations = diagnostics.failed_perturbations;

        const bool all_columns_valid = std::all_of(
            diagnostics.column_valid.begin(),
            diagnostics.column_valid.end(),
            [](bool valid) { return valid; });
        const bool all_columns_central = std::all_of(
            diagnostics.column_central.begin(),
            diagnostics.column_central.end(),
            [](bool central) { return central; });

        response.forces_consistent_with_tangent =
            all_columns_valid && all_columns_central
            && !diagnostics.tangent_regularized;

        if (response.tangent.norm() == 0.0 && response.forces.norm() == 0.0) {
            response.status = ResponseStatus::SolveFailed;
        } else if (!all_columns_valid) {
            response.status = ResponseStatus::InvalidOperator;
        } else if (all_columns_central
                   && diagnostics.failed_perturbations == 0
                   && !diagnostics.tangent_regularized)
        {
            response.status = ResponseStatus::Ok;
        } else {
            response.status = ResponseStatus::Degraded;
        }

        return response;
    }
};

}  // namespace fall_n

#endif // FALL_N_SRC_RECONSTRUCTION_BOUNDARY_REACTION_HOMOGENIZER_HH
