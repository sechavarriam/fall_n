#ifndef FALL_N_SRC_RECONSTRUCTION_BOUNDARY_REACTION_HOMOGENIZER_HH
#define FALL_N_SRC_RECONSTRUCTION_BOUNDARY_REACTION_HOMOGENIZER_HH

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <optional>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <petsc.h>

#include "../analysis/PenaltyCoupling.hh"
#include "../analysis/MultiscaleTypes.hh"
#include "../numerics/SparseSchurComplement.hh"
#include "LocalBoundaryConditionApplicator.hh"
#include "PersistentLocalStateOps.hh"

namespace fall_n {

template <typename ModelT, typename SubModelT>
class BoundaryReactionHomogenizer {
public:
    using SparseMatrixT = Eigen::SparseMatrix<double>;

    struct TangentDiagnostics {
        Eigen::Matrix<double, 6, 6> tangent{
            Eigen::Matrix<double, 6, 6>::Zero()};
        std::array<double, 6> perturbation_sizes{};
        std::array<bool, 6> column_valid{};
        std::array<bool, 6> column_central{};
        double symmetry_error_before_regularization{0.0};
        double condensed_solve_residual{0.0};
        bool condensed_pattern_reused{false};
        std::size_t condensed_symbolic_factorizations{0};
        int failed_perturbations{0};
        bool tangent_regularized{false};
        TangentLinearizationScheme scheme{
            TangentLinearizationScheme::Unknown};
        TangentComputationMode requested_mode{
            TangentComputationMode::PreferLinearizedCondensation};
        CondensedTangentStatus condensed_status{
            CondensedTangentStatus::NotAttempted};
    };

private:
    struct CondensedAttemptResult {
        std::optional<TangentDiagnostics> diagnostics{};
        CondensedTangentStatus status{
            CondensedTangentStatus::NotAttempted};
        double solve_residual{0.0};
    };

    ModelT* model_{nullptr};
    SubModelT* sub_{nullptr};
    Vec displacement_{nullptr};
    Vec displacement_work_{nullptr};
    Vec imposed_work_{nullptr};
    SNES snes_{nullptr};
    RegularizationPolicyKind regularization_policy_{
        RegularizationPolicyKind::DiagonalFloor};
    double diagonal_floor_{1.0};
    TangentComputationMode tangent_mode_{
        TangentComputationMode::PreferLinearizedCondensation};
    const std::vector<PenaltyCouplingEntry>* penalty_couplings_{nullptr};
    double alpha_penalty_{0.0};
    LocalBoundaryConditionApplicator<ModelT, SubModelT> bc_applicator_{};
    PersistentLocalStateOps<ModelT> state_ops_{};
    condensation::SparseSchurComplementWorkspace<SparseMatrixT>*
        condensed_workspace_{nullptr};

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

    struct ConstraintPartition {
        std::vector<Eigen::Index> free_dofs{};
        std::vector<Eigen::Index> constrained_dofs{};
        std::vector<int> local_to_free{};
        std::vector<int> local_to_constrained{};
    };

    [[nodiscard]] static bool valid_local_range_(PetscInt offset,
                                                 PetscInt width,
                                                 Eigen::Index size)
    {
        return offset >= 0
            && width >= 0
            && static_cast<Eigen::Index>(offset + width) <= size;
    }

    [[nodiscard]] ConstraintPartition build_constraint_partition_() const
    {
        ConstraintPartition partition;
        if (!model_) {
            return partition;
        }

        PetscInt n_local = 0;
        VecGetLocalSize(model_->state_vector(), &n_local);
        if (n_local <= 0) {
            return partition;
        }

        partition.local_to_constrained.assign(
            static_cast<std::size_t>(n_local), -1);
        partition.local_to_free.assign(
            static_cast<std::size_t>(n_local), -1);

        for (const auto& node : sub_->domain.nodes()) {
            const auto dofs = node.dof_index();
            for (std::size_t local_dof = 0;
                 local_dof < dofs.size() && local_dof < node.num_dof();
                 ++local_dof)
            {
                const auto idx =
                    static_cast<Eigen::Index>(dofs[local_dof]);
                if (idx < 0 || idx >= static_cast<Eigen::Index>(n_local)) {
                    continue;
                }

                if (model_->is_dof_constrained(node.id(), local_dof)) {
                    partition.local_to_constrained[static_cast<std::size_t>(idx)] =
                        static_cast<int>(partition.constrained_dofs.size());
                    partition.constrained_dofs.push_back(idx);
                } else {
                    partition.local_to_free[static_cast<std::size_t>(idx)] =
                        static_cast<int>(partition.free_dofs.size());
                    partition.free_dofs.push_back(idx);
                }
            }
        }

        return partition;
    }

    [[nodiscard]] Eigen::MatrixXd
    build_max_face_transfer_(const ConstraintPartition& partition) const
    {
        Eigen::MatrixXd T = Eigen::MatrixXd::Zero(
            static_cast<Eigen::Index>(partition.constrained_dofs.size()), 6);

        if (!sub_) {
            return T;
        }

        const Eigen::Matrix3d rotation_T = sub_->kin_B.R.transpose();
        const Eigen::Vector3d& centroid = sub_->kin_B.centroid;

        for (const auto nid_raw : sub_->face_max_z_ids) {
            const auto nid = static_cast<std::size_t>(nid_raw);
            const auto& node = sub_->domain.node(nid);
            if (node.num_dof() == 0) {
                continue;
            }

            const Eigen::Vector3d pos{
                node.coord(0), node.coord(1), node.coord(2)};
            const Eigen::Vector3d offset_local =
                sub_->kin_B.R * (pos - centroid);
            const double y = offset_local[1];
            const double z = offset_local[2];

            const std::array<Eigen::Vector3d, 6> delta_local{{
                Eigen::Vector3d{1.0, 0.0, 0.0},
                Eigen::Vector3d{z,   0.0, 0.0},
                Eigen::Vector3d{-y,  0.0, 0.0},
                Eigen::Vector3d{0.0, 1.0, 0.0},
                Eigen::Vector3d{0.0, 0.0, 1.0},
                Eigen::Vector3d{0.0, -z,   y  },
            }};

            const auto dofs = node.dof_index();
            for (std::size_t local_dof = 0;
                 local_dof < dofs.size() && local_dof < 3;
                 ++local_dof)
            {
                if (!model_->is_dof_constrained(nid, local_dof)) {
                    continue;
                }

                const auto idx =
                    static_cast<Eigen::Index>(dofs[local_dof]);
                if (idx < 0 || idx >= static_cast<Eigen::Index>(
                        partition.local_to_constrained.size()))
                {
                    continue;
                }

                const int constrained_pos =
                    partition.local_to_constrained[static_cast<std::size_t>(idx)];
                if (constrained_pos < 0) {
                    continue;
                }

                for (int j = 0; j < 6; ++j) {
                    const Eigen::Vector3d delta_global =
                        rotation_T * delta_local[static_cast<std::size_t>(j)];
                    T(constrained_pos, j) =
                        delta_global[static_cast<Eigen::Index>(local_dof)];
                }
            }
        }

        return T;
    }

    static void append_triplet_(std::vector<Eigen::Triplet<double>>& triplets,
                                Eigen::Index row,
                                Eigen::Index col,
                                double value)
    {
        if (std::abs(value) > 0.0) {
            triplets.emplace_back(row, col, value);
        }
    }

    void append_penalty_coupling_triplets_(
        std::vector<Eigen::Triplet<double>>& triplets,
        Eigen::Index local_size) const
    {
        if (!model_ || !penalty_couplings_ || penalty_couplings_->empty()) {
            return;
        }

        PetscSection section;
        DMGetLocalSection(model_->get_plex(), &section);

        for (const auto& pc : *penalty_couplings_) {
            PetscInt r_dof = 0;
            PetscSectionGetDof(section, pc.rebar_sieve_pt, &r_dof);
            if (r_dof <= 0) {
                continue;
            }

            PetscInt r_off = 0;
            PetscSectionGetOffset(section, pc.rebar_sieve_pt, &r_off);
            if (!valid_local_range_(r_off, 3, local_size)) {
                continue;
            }

            for (int d = 0; d < 3; ++d) {
                append_triplet_(triplets, r_off + d, r_off + d, alpha_penalty_);
            }

            for (const auto& [sp, Ni] : pc.hex_weights) {
                PetscInt h_dof = 0;
                PetscSectionGetDof(section, sp, &h_dof);
                if (h_dof <= 0) {
                    continue;
                }

                PetscInt h_off = 0;
                PetscSectionGetOffset(section, sp, &h_off);
                if (!valid_local_range_(h_off, 3, local_size)) {
                    continue;
                }

                for (int d = 0; d < 3; ++d) {
                    const double v = -alpha_penalty_ * Ni;
                    append_triplet_(triplets, r_off + d, h_off + d, v);
                    append_triplet_(triplets, h_off + d, r_off + d, v);
                }
            }

            for (const auto& [si, Ni] : pc.hex_weights) {
                PetscInt gi_dof = 0;
                PetscSectionGetDof(section, si, &gi_dof);
                if (gi_dof <= 0) {
                    continue;
                }

                PetscInt gi_off = 0;
                PetscSectionGetOffset(section, si, &gi_off);
                if (!valid_local_range_(gi_off, 3, local_size)) {
                    continue;
                }

                for (const auto& [sj, Nj] : pc.hex_weights) {
                    PetscInt gj_dof = 0;
                    PetscSectionGetDof(section, sj, &gj_dof);
                    if (gj_dof <= 0) {
                        continue;
                    }

                    PetscInt gj_off = 0;
                    PetscSectionGetOffset(section, sj, &gj_off);
                    if (!valid_local_range_(gj_off, 3, local_size)) {
                        continue;
                    }

                    for (int d = 0; d < 3; ++d) {
                        append_triplet_(
                            triplets,
                            gi_off + d,
                            gj_off + d,
                            alpha_penalty_ * Ni * Nj);
                    }
                }
            }
        }
    }

    [[nodiscard]] std::optional<SparseMatrixT>
    assemble_sparse_local_tangent_() const
    {
        if (!model_) {
            return std::nullopt;
        }

        state_ops_.sync_state_vector();

        PetscInt n_local = 0;
        VecGetLocalSize(model_->state_vector(), &n_local);
        if (n_local <= 0) {
            return std::nullopt;
        }

        std::vector<Eigen::Triplet<double>> triplets;
        Vec u_local = model_->state_vector();

        for (auto& elem : model_->elements()) {
            auto u_e = elem.extract_element_dofs(u_local);
            auto K_e = elem.compute_tangent_stiffness_matrix(u_e);
            const auto& dofs = elem.get_dof_indices();

            if (u_e.size() == 0 || K_e.rows() == 0 || K_e.cols() == 0
                || dofs.empty()
                || K_e.rows() != static_cast<Eigen::Index>(dofs.size())
                || K_e.cols() != static_cast<Eigen::Index>(dofs.size()))
            {
                return std::nullopt;
            }

            for (Eigen::Index i = 0; i < K_e.rows(); ++i) {
                const auto row = static_cast<Eigen::Index>(
                    dofs[static_cast<std::size_t>(i)]);
                if (row < 0 || row >= static_cast<Eigen::Index>(n_local)) {
                    return std::nullopt;
                }
                for (Eigen::Index j = 0; j < K_e.cols(); ++j) {
                    const auto col = static_cast<Eigen::Index>(
                        dofs[static_cast<std::size_t>(j)]);
                    if (col < 0 || col >= static_cast<Eigen::Index>(n_local)) {
                        return std::nullopt;
                    }
                    append_triplet_(triplets, row, col, K_e(i, j));
                }
            }
        }

        append_penalty_coupling_triplets_(triplets, n_local);

        SparseMatrixT tangent(n_local, n_local);
        tangent.setFromTriplets(
            triplets.begin(), triplets.end(), std::plus<double>{});
        tangent.makeCompressed();
        return tangent;
    }

    void finalize_tangent_(TangentDiagnostics& diagnostics) const
    {
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
    }

    [[nodiscard]] static CondensedTangentStatus
    condensed_status_from_sparse_schur_(
        condensation::SparseSchurStatus status)
    {
        switch (status) {
            case condensation::SparseSchurStatus::Success:
                return CondensedTangentStatus::Success;
            case condensation::SparseSchurStatus::FactorizationFailed:
                return CondensedTangentStatus::FactorizationFailed;
            case condensation::SparseSchurStatus::SolveFailed:
                return CondensedTangentStatus::SolveFailed;
            case condensation::SparseSchurStatus::ResidualTooLarge:
                return CondensedTangentStatus::ResidualTooLarge;
            case condensation::SparseSchurStatus::InvalidArguments:
                return CondensedTangentStatus::AssemblyFailed;
        }
        return CondensedTangentStatus::AssemblyFailed;
    }

    [[nodiscard]] CondensedAttemptResult
    compute_linearized_condensed_tangent_() const
    {
        CondensedAttemptResult attempt;
        if (!model_ || !sub_ || !displacement_) {
            attempt.status = CondensedTangentStatus::MissingModel;
            return attempt;
        }

        attempt.status = CondensedTangentStatus::AssemblyFailed;

        auto tangent_full = assemble_sparse_local_tangent_();
        if (!tangent_full) {
            return attempt;
        }

        const auto partition = build_constraint_partition_();
        if (partition.constrained_dofs.empty()) {
            attempt.status = CondensedTangentStatus::NoConstrainedDofs;
            return attempt;
        }

        const Eigen::MatrixXd transfer = build_max_face_transfer_(partition);
        if (transfer.rows() == 0 || transfer.norm() == 0.0) {
            attempt.status = CondensedTangentStatus::ZeroTransfer;
            return attempt;
        }

        Eigen::MatrixXd k_cc_times_t = Eigen::MatrixXd::Zero(
            static_cast<Eigen::Index>(partition.constrained_dofs.size()), 6);
        Eigen::MatrixXd k_fc_times_t = Eigen::MatrixXd::Zero(
            static_cast<Eigen::Index>(partition.free_dofs.size()), 6);
        std::vector<Eigen::Triplet<double>> ff_triplets;
        std::vector<Eigen::Triplet<double>> cf_triplets;

        for (Eigen::Index outer = 0; outer < tangent_full->outerSize(); ++outer) {
            for (typename SparseMatrixT::InnerIterator it(*tangent_full, outer);
                 it; ++it)
            {
                const auto row = it.row();
                const auto col = it.col();
                const double value = it.value();

                const int row_free =
                    partition.local_to_free[static_cast<std::size_t>(row)];
                const int row_constrained =
                    partition.local_to_constrained[static_cast<std::size_t>(row)];
                const int col_free =
                    partition.local_to_free[static_cast<std::size_t>(col)];
                const int col_constrained =
                    partition.local_to_constrained[static_cast<std::size_t>(col)];

                if (row_free >= 0 && col_free >= 0) {
                    append_triplet_(ff_triplets, row_free, col_free, value);
                }

                if (col_constrained >= 0) {
                    if (row_free >= 0) {
                        k_fc_times_t.row(row_free).noalias() +=
                            value * transfer.row(col_constrained);
                    } else if (row_constrained >= 0) {
                        k_cc_times_t.row(row_constrained).noalias() +=
                            value * transfer.row(col_constrained);
                    }
                }

                if (row_constrained >= 0 && col_free >= 0) {
                    append_triplet_(cf_triplets, row_constrained, col_free, value);
                }
            }
        }

        Eigen::MatrixXd condensed_boundary = k_cc_times_t;
        if (!partition.free_dofs.empty()) {
            SparseMatrixT K_ff(
                static_cast<Eigen::Index>(partition.free_dofs.size()),
                static_cast<Eigen::Index>(partition.free_dofs.size()));
            K_ff.setFromTriplets(
                ff_triplets.begin(), ff_triplets.end(), std::plus<double>{});
            K_ff.makeCompressed();
            SparseMatrixT K_cf(
                static_cast<Eigen::Index>(partition.constrained_dofs.size()),
                static_cast<Eigen::Index>(partition.free_dofs.size()));
            K_cf.setFromTriplets(
                cf_triplets.begin(), cf_triplets.end(), std::plus<double>{});
            K_cf.makeCompressed();

            const auto condensed =
                condensation::apply_condensed_operator(
                    K_ff,
                    K_cf,
                    k_fc_times_t,
                    k_cc_times_t,
                    1.0e-8,
                    condensed_workspace_);
            attempt.solve_residual = condensed.solve_residual;
            if (condensed.status != condensation::SparseSchurStatus::Success) {
                attempt.status =
                    condensed_status_from_sparse_schur_(condensed.status);
                return attempt;
            }

            condensed_boundary = std::move(condensed.condensed_times_transfer);
            attempt.solve_residual = condensed.solve_residual;
            attempt.diagnostics.emplace();
            attempt.diagnostics->condensed_pattern_reused =
                condensed.pattern_reused;
            attempt.diagnostics->condensed_symbolic_factorizations =
                condensed.symbolic_factorizations;
        }

        TangentDiagnostics diagnostics;
        if (attempt.diagnostics) {
            diagnostics.condensed_pattern_reused =
                attempt.diagnostics->condensed_pattern_reused;
            diagnostics.condensed_symbolic_factorizations =
                attempt.diagnostics->condensed_symbolic_factorizations;
        } else if (condensed_workspace_) {
            diagnostics.condensed_symbolic_factorizations =
                condensed_workspace_->symbolic_factorizations();
        }
        diagnostics.scheme =
            TangentLinearizationScheme::LinearizedCondensation;
        diagnostics.requested_mode = tangent_mode_;
        diagnostics.condensed_status = CondensedTangentStatus::Success;
        diagnostics.condensed_solve_residual = attempt.solve_residual;
        diagnostics.tangent =
            transfer.transpose() * condensed_boundary;

        for (int j = 0; j < 6; ++j) {
            const bool column_nontrivial =
                transfer.col(j).norm() > 1.0e-14
                && diagnostics.tangent.col(j).allFinite();
            diagnostics.column_valid[static_cast<std::size_t>(j)] =
                column_nontrivial;
            diagnostics.column_central[static_cast<std::size_t>(j)] =
                column_nontrivial;
        }

        finalize_tangent_(diagnostics);
        attempt.status = CondensedTangentStatus::Success;
        attempt.diagnostics = diagnostics;
        return attempt;
    }

    [[nodiscard]] TangentDiagnostics
    compute_adaptive_fd_tangent_(double h_pert) const
    {
        TangentDiagnostics diagnostics;
        diagnostics.scheme =
            TangentLinearizationScheme::AdaptiveFiniteDifference;
        diagnostics.requested_mode = tangent_mode_;

        if (!model_ || !sub_ || !displacement_ || !displacement_work_
            || !imposed_work_ || !snes_)
        {
            diagnostics.condensed_status =
                CondensedTangentStatus::MissingModel;
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

        finalize_tangent_(diagnostics);
        return diagnostics;
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
        TangentComputationMode tangent_mode,
        const std::vector<PenaltyCouplingEntry>* penalty_couplings,
        double alpha_penalty,
        LocalBoundaryConditionApplicator<ModelT, SubModelT> bc_applicator,
        PersistentLocalStateOps<ModelT> state_ops,
        condensation::SparseSchurComplementWorkspace<SparseMatrixT>*
            condensed_workspace) noexcept
        : model_{model}
        , sub_{sub}
        , displacement_{displacement}
        , displacement_work_{displacement_work}
        , imposed_work_{imposed_work}
        , snes_{snes}
        , regularization_policy_{regularization_policy}
        , diagonal_floor_{diagonal_floor}
        , tangent_mode_{tangent_mode}
        , penalty_couplings_{penalty_couplings}
        , alpha_penalty_{alpha_penalty}
        , bc_applicator_{bc_applicator}
        , state_ops_{state_ops}
        , condensed_workspace_{condensed_workspace}
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
        if (tangent_mode_
            == TangentComputationMode::ForceAdaptiveFiniteDifference)
        {
            auto diagnostics = compute_adaptive_fd_tangent_(h_pert);
            diagnostics.condensed_status =
                CondensedTangentStatus::ForcedAdaptiveFiniteDifference;
            return diagnostics;
        }

        const auto condensed = compute_linearized_condensed_tangent_();
        if (condensed.diagnostics) {
            return *condensed.diagnostics;
        }

        auto diagnostics = compute_adaptive_fd_tangent_(h_pert);
        diagnostics.condensed_status = condensed.status;
        diagnostics.condensed_solve_residual = condensed.solve_residual;
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
        response.tangent_scheme = diagnostics.scheme;
        response.tangent_mode_requested = diagnostics.requested_mode;
        response.condensed_tangent_status = diagnostics.condensed_status;
        response.condensed_solve_residual =
            diagnostics.condensed_solve_residual;
        response.condensed_pattern_reused =
            diagnostics.condensed_pattern_reused;
        response.condensed_symbolic_factorizations =
            diagnostics.condensed_symbolic_factorizations;
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
            all_columns_valid
            && !diagnostics.tangent_regularized
            && (diagnostics.scheme
                    == TangentLinearizationScheme::LinearizedCondensation
                || all_columns_central);

        if (response.tangent.norm() == 0.0 && response.forces.norm() == 0.0) {
            response.status = ResponseStatus::SolveFailed;
        } else if (!all_columns_valid) {
            response.status = ResponseStatus::InvalidOperator;
        } else if (diagnostics.scheme
                       == TangentLinearizationScheme::LinearizedCondensation
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
