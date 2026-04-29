#ifndef FALL_N_SRC_ANALYSIS_PETSC_BORDERED_MIXED_CONTROL_NEWTON_HH
#define FALL_N_SRC_ANALYSIS_PETSC_BORDERED_MIXED_CONTROL_NEWTON_HH

// =============================================================================
//  PetscBorderedMixedControlNewton
// =============================================================================
//
//  PETSc/KSP backend for the second-generation mixed-control continuation seam.
//  The dense Eigen kernel in BorderedMixedControlNewton.hh fixes the algebraic
//  contract; this header executes the same bordered Newton correction with
//  PETSc Vec/Mat/KSP objects:
//
//      [ K   r_lambda ] [du]      -[R]
//      [ g^T c_lambda ] [dlambda] = -[c]
//
//  The caller owns the physical residual, tangent, load column and scalar
//  constraint.  This layer owns only:
//
//    - explicit assembly of the bordered linear system,
//    - KSP configuration and diagnostics,
//    - state updates and optional merit line-search,
//    - a narrow contract that can be reused by global XFEM, FE2 locals, or
//      future dynamic mixed-control steps.
//
//  The implementation intentionally assembles an explicit AIJ bordered matrix
//  instead of hiding the row/column in ad-hoc callbacks.  That is a little less
//  clever, but it is inspectable, debuggable, and maps directly to PETSc
//  preconditioning experiments (LU for small validation runs, field-split or
//  Schur complement for larger multiscale batches).
//
// =============================================================================

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <mpi.h>
#include <petscksp.h>

#include "BorderedMixedControlNewton.hh"
#include "../petsc/PetscRaii.hh"

namespace fall_n {

struct PetscBorderedMixedControlState {
    Vec unknowns{nullptr};
    double load_parameter{0.0};
};

struct PetscBorderedMixedControlEvaluation {
    petsc::OwnedVec residual{};
    petsc::OwnedMat tangent{};
    petsc::OwnedVec load_column{};
    petsc::OwnedVec constraint_gradient{};
    double constraint{0.0};
    double constraint_load_derivative{0.0};
};

enum class PetscBorderedLineSearchMeritKind {
    residual_and_constraint,
    residual_only
};

struct PetscBorderedMixedControlNewtonSettings {
    int max_iterations{30};
    double residual_tolerance{1.0e-10};
    double constraint_tolerance{1.0e-10};
    double correction_tolerance{1.0e-12};

    bool line_search_enabled{true};
    int max_line_search_cutbacks{10};
    double line_search_cutback_factor{0.5};
    double line_search_min_alpha{1.0e-4};
    int line_search_extra_trial_count{0};
    double merit_decrease{1.0e-4};
    PetscBorderedLineSearchMeritKind line_search_merit{
        PetscBorderedLineSearchMeritKind::residual_and_constraint};
    bool accept_best_line_search_trial{false};
    double best_line_search_trial_growth_tolerance{0.0};

    std::string ksp_type{KSPPREONLY};
    std::string pc_type{PCLU};
    double ksp_rtol{1.0e-12};
    double ksp_atol{1.0e-14};
    int ksp_max_iterations{1000};
    bool reuse_preconditioner{false};
};

struct PetscBorderedMixedControlNewtonResult {
    BorderedMixedControlNewtonStatus status{
        BorderedMixedControlNewtonStatus::invalid_evaluation};
    petsc::OwnedVec unknowns{};
    double load_parameter{0.0};
    int iterations{0};
    int last_ksp_iterations{0};
    int total_ksp_iterations{0};
    KSPConvergedReason last_ksp_reason{KSP_CONVERGED_ITERATING};
    double residual_norm{0.0};
    double constraint_abs{0.0};
    double correction_norm{0.0};
    int evaluator_calls{0};
    int augmented_system_allocations{0};
    std::vector<BorderedMixedControlNewtonIterationRecord> records{};

    [[nodiscard]] bool converged() const noexcept
    {
        return status == BorderedMixedControlNewtonStatus::converged;
    }
};

namespace detail {

[[nodiscard]] inline MPI_Comm petsc_object_comm(PetscObject obj)
{
    MPI_Comm comm = PETSC_COMM_SELF;
    FALL_N_PETSC_CHECK(PetscObjectGetComm(obj, &comm));
    return comm;
}

[[nodiscard]] inline PetscInt petsc_vec_size(Vec v)
{
    PetscInt n = 0;
    FALL_N_PETSC_CHECK(VecGetSize(v, &n));
    return n;
}

[[nodiscard]] inline double petsc_vec_norm(Vec v)
{
    PetscReal n = 0.0;
    FALL_N_PETSC_CHECK(VecNorm(v, NORM_2, &n));
    return static_cast<double>(n);
}

[[nodiscard]] inline bool petsc_bordered_eval_has_consistent_dimensions(
    const PetscBorderedMixedControlEvaluation& e,
    PetscInt n)
{
    if (n <= 0 || !e.residual || !e.tangent || !e.load_column ||
        !e.constraint_gradient || !std::isfinite(e.constraint) ||
        !std::isfinite(e.constraint_load_derivative))
    {
        return false;
    }

    PetscInt r_n = 0;
    PetscInt load_n = 0;
    PetscInt grad_n = 0;
    PetscInt k_m = 0;
    PetscInt k_n = 0;
    FALL_N_PETSC_CHECK(VecGetSize(e.residual.get(), &r_n));
    FALL_N_PETSC_CHECK(VecGetSize(e.load_column.get(), &load_n));
    FALL_N_PETSC_CHECK(VecGetSize(e.constraint_gradient.get(), &grad_n));
    FALL_N_PETSC_CHECK(MatGetSize(e.tangent.get(), &k_m, &k_n));
    return r_n == n && load_n == n && grad_n == n && k_m == n && k_n == n;
}

[[nodiscard]] inline double petsc_bordered_merit(
    const PetscBorderedMixedControlEvaluation& e,
    PetscBorderedLineSearchMeritKind kind)
{
    const double r = petsc_vec_norm(e.residual.get());
    const double residual_merit = r * r;
    if (kind == PetscBorderedLineSearchMeritKind::residual_only) {
        return residual_merit;
    }
    return residual_merit + e.constraint * e.constraint;
}

[[nodiscard]] inline petsc::OwnedVec duplicate_vec(Vec prototype)
{
    petsc::OwnedVec out;
    FALL_N_PETSC_CHECK(VecDuplicate(prototype, out.ptr()));
    return out;
}

[[nodiscard]] inline petsc::OwnedVec clone_vec(Vec source)
{
    auto out = duplicate_vec(source);
    FALL_N_PETSC_CHECK(VecCopy(source, out.get()));
    return out;
}

[[nodiscard]] inline petsc::OwnedVec create_augmented_vec(
    MPI_Comm comm,
    PetscInt n)
{
    petsc::OwnedVec out;
    FALL_N_PETSC_CHECK(VecCreate(comm, out.ptr()));
    FALL_N_PETSC_CHECK(VecSetSizes(out.get(), PETSC_DECIDE, n + 1));
    FALL_N_PETSC_CHECK(VecSetFromOptions(out.get()));
    FALL_N_PETSC_CHECK(VecSet(out.get(), 0.0));
    return out;
}

[[nodiscard]] inline petsc::OwnedMat create_augmented_mat(
    MPI_Comm comm,
    PetscInt n)
{
    petsc::OwnedMat out;
    FALL_N_PETSC_CHECK(MatCreate(comm, out.ptr()));
    FALL_N_PETSC_CHECK(
        MatSetSizes(out.get(), PETSC_DECIDE, PETSC_DECIDE, n + 1, n + 1));
    FALL_N_PETSC_CHECK(MatSetType(out.get(), MATAIJ));
    FALL_N_PETSC_CHECK(MatSetFromOptions(out.get()));
    FALL_N_PETSC_CHECK(MatSetUp(out.get()));
    FALL_N_PETSC_CHECK(
        MatSetOption(out.get(), MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
    FALL_N_PETSC_CHECK(
        MatSetOption(out.get(), MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE));
    return out;
}

inline void set_last_scalar_entry(Vec v, PetscInt n, PetscScalar value)
{
    MPI_Comm comm = petsc_object_comm(reinterpret_cast<PetscObject>(v));
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    if (rank == 0) {
        FALL_N_PETSC_CHECK(VecSetValue(v, n, value, INSERT_VALUES));
    }
}

inline void assemble_petsc_bordered_rhs(
    const PetscBorderedMixedControlEvaluation& e,
    Vec rhs)
{
    PetscInt r0 = 0;
    PetscInt r1 = 0;
    FALL_N_PETSC_CHECK(VecGetOwnershipRange(e.residual.get(), &r0, &r1));
    const PetscScalar* residual = nullptr;
    FALL_N_PETSC_CHECK(VecGetArrayRead(e.residual.get(), &residual));
    for (PetscInt r = r0; r < r1; ++r) {
        const PetscScalar value = -residual[r - r0];
        FALL_N_PETSC_CHECK(VecSetValue(rhs, r, value, INSERT_VALUES));
    }
    FALL_N_PETSC_CHECK(VecRestoreArrayRead(e.residual.get(), &residual));
    set_last_scalar_entry(rhs, petsc_vec_size(e.residual.get()), -e.constraint);
    FALL_N_PETSC_CHECK(VecAssemblyBegin(rhs));
    FALL_N_PETSC_CHECK(VecAssemblyEnd(rhs));
}

inline void assemble_petsc_bordered_matrix(
    const PetscBorderedMixedControlEvaluation& e,
    Mat augmented)
{
    const PetscInt n = petsc_vec_size(e.residual.get());

    PetscInt k0 = 0;
    PetscInt k1 = 0;
    FALL_N_PETSC_CHECK(MatGetOwnershipRange(e.tangent.get(), &k0, &k1));
    for (PetscInt r = k0; r < k1; ++r) {
        PetscInt ncols = 0;
        const PetscInt* cols = nullptr;
        const PetscScalar* values = nullptr;
        FALL_N_PETSC_CHECK(MatGetRow(
            e.tangent.get(), r, &ncols, &cols, &values));
        if (ncols > 0) {
            FALL_N_PETSC_CHECK(MatSetValues(
                augmented, 1, &r, ncols, cols, values, INSERT_VALUES));
        }
        FALL_N_PETSC_CHECK(MatRestoreRow(
            e.tangent.get(), r, &ncols, &cols, &values));
    }

    PetscInt v0 = 0;
    PetscInt v1 = 0;
    FALL_N_PETSC_CHECK(VecGetOwnershipRange(e.load_column.get(), &v0, &v1));
    const PetscScalar* load = nullptr;
    FALL_N_PETSC_CHECK(VecGetArrayRead(e.load_column.get(), &load));
    for (PetscInt r = v0; r < v1; ++r) {
        const PetscScalar value = load[r - v0];
        FALL_N_PETSC_CHECK(
            MatSetValue(augmented, r, n, value, INSERT_VALUES));
    }
    FALL_N_PETSC_CHECK(VecRestoreArrayRead(e.load_column.get(), &load));

    PetscInt g0 = 0;
    PetscInt g1 = 0;
    FALL_N_PETSC_CHECK(
        VecGetOwnershipRange(e.constraint_gradient.get(), &g0, &g1));
    const PetscScalar* gradient = nullptr;
    FALL_N_PETSC_CHECK(
        VecGetArrayRead(e.constraint_gradient.get(), &gradient));
    for (PetscInt c = g0; c < g1; ++c) {
        const PetscScalar value = gradient[c - g0];
        FALL_N_PETSC_CHECK(
            MatSetValue(augmented, n, c, value, INSERT_VALUES));
    }
    FALL_N_PETSC_CHECK(
        VecRestoreArrayRead(e.constraint_gradient.get(), &gradient));

    MPI_Comm comm = petsc_object_comm(reinterpret_cast<PetscObject>(augmented));
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    if (rank == 0) {
        FALL_N_PETSC_CHECK(MatSetValue(
            augmented,
            n,
            n,
            static_cast<PetscScalar>(e.constraint_load_derivative),
            INSERT_VALUES));
    }

    FALL_N_PETSC_CHECK(MatAssemblyBegin(augmented, MAT_FINAL_ASSEMBLY));
    FALL_N_PETSC_CHECK(MatAssemblyEnd(augmented, MAT_FINAL_ASSEMBLY));
}

inline void apply_augmented_step_to_unknowns(
    Vec unknowns,
    Vec augmented_step,
    double alpha)
{
    PetscInt u0 = 0;
    PetscInt u1 = 0;
    FALL_N_PETSC_CHECK(VecGetOwnershipRange(unknowns, &u0, &u1));
    if (u1 > u0) {
        std::vector<PetscInt> rows(static_cast<std::size_t>(u1 - u0));
        std::vector<PetscScalar> values(rows.size(), 0.0);
        for (PetscInt i = u0; i < u1; ++i) {
            rows[static_cast<std::size_t>(i - u0)] = i;
        }
        FALL_N_PETSC_CHECK(VecGetValues(
            augmented_step,
            static_cast<PetscInt>(rows.size()),
            rows.data(),
            values.data()));
        for (auto& value : values) {
            value *= static_cast<PetscScalar>(alpha);
        }
        FALL_N_PETSC_CHECK(VecSetValues(
            unknowns,
            static_cast<PetscInt>(rows.size()),
            rows.data(),
            values.data(),
            ADD_VALUES));
        FALL_N_PETSC_CHECK(VecAssemblyBegin(unknowns));
        FALL_N_PETSC_CHECK(VecAssemblyEnd(unknowns));
    }
}

[[nodiscard]] inline petsc::OwnedVec make_trial_unknowns(
    Vec current,
    Vec augmented_step,
    double alpha)
{
    auto trial = clone_vec(current);
    apply_augmented_step_to_unknowns(
        trial.get(),
        augmented_step,
        alpha);
    return trial;
}

inline void make_trial_unknowns_inplace(
    Vec current,
    Vec augmented_step,
    double alpha,
    Vec trial)
{
    FALL_N_PETSC_CHECK(VecCopy(current, trial));
    apply_augmented_step_to_unknowns(trial, augmented_step, alpha);
}

[[nodiscard]] inline double read_augmented_load_step(
    Vec augmented_step,
    PetscInt n)
{
    MPI_Comm comm =
        petsc_object_comm(reinterpret_cast<PetscObject>(augmented_step));
    PetscInt r0 = 0;
    PetscInt r1 = 0;
    FALL_N_PETSC_CHECK(VecGetOwnershipRange(augmented_step, &r0, &r1));
    double dlambda = 0.0;
    if (n >= r0 && n < r1) {
        PetscScalar value = 0.0;
        FALL_N_PETSC_CHECK(VecGetValues(augmented_step, 1, &n, &value));
        dlambda = static_cast<double>(value);
    }
    MPI_Allreduce(MPI_IN_PLACE, &dlambda, 1, MPI_DOUBLE, MPI_SUM, comm);
    return dlambda;
}

[[nodiscard]] inline double augmented_correction_norm(
    Vec augmented_step,
    PetscInt n)
{
    (void)n;
    PetscInt r0 = 0;
    PetscInt r1 = 0;
    FALL_N_PETSC_CHECK(VecGetOwnershipRange(augmented_step, &r0, &r1));
    const PetscScalar* values = nullptr;
    FALL_N_PETSC_CHECK(VecGetArrayRead(augmented_step, &values));
    double local = 0.0;
    for (PetscInt i = r0; i < r1; ++i) {
        const auto value = static_cast<double>(values[i - r0]);
        local += value * value;
    }
    FALL_N_PETSC_CHECK(VecRestoreArrayRead(augmented_step, &values));
    MPI_Comm comm =
        petsc_object_comm(reinterpret_cast<PetscObject>(augmented_step));
    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, comm);
    return std::sqrt(std::max(global, 0.0));
}

inline void configure_bordered_ksp(
    KSP ksp,
    Mat augmented,
    const PetscBorderedMixedControlNewtonSettings& settings)
{
    FALL_N_PETSC_CHECK(KSPSetOperators(ksp, augmented, augmented));
    FALL_N_PETSC_CHECK(KSPSetType(ksp, settings.ksp_type.c_str()));
    FALL_N_PETSC_CHECK(KSPSetTolerances(
        ksp,
        settings.ksp_rtol,
        settings.ksp_atol,
        PETSC_DEFAULT,
        settings.ksp_max_iterations));
    PC pc = nullptr;
    FALL_N_PETSC_CHECK(KSPGetPC(ksp, &pc));
    FALL_N_PETSC_CHECK(PCSetType(pc, settings.pc_type.c_str()));
    if (settings.reuse_preconditioner) {
        FALL_N_PETSC_CHECK(KSPSetReusePreconditioner(ksp, PETSC_TRUE));
    }
    FALL_N_PETSC_CHECK(KSPSetFromOptions(ksp));
}

} // namespace detail

template <typename EvaluatorT>
[[nodiscard]] PetscBorderedMixedControlNewtonResult
solve_petsc_bordered_mixed_control_newton(
    PetscBorderedMixedControlState initial_state,
    EvaluatorT&& evaluator,
    PetscBorderedMixedControlNewtonSettings settings = {})
{
    PetscBorderedMixedControlNewtonResult result{};
    if (!initial_state.unknowns) {
        return result;
    }

    settings.max_iterations = std::max(settings.max_iterations, 1);
    settings.max_line_search_cutbacks =
        std::max(settings.max_line_search_cutbacks, 0);
    settings.line_search_cutback_factor =
        std::clamp(settings.line_search_cutback_factor, 0.05, 0.95);
    settings.line_search_min_alpha =
        std::clamp(settings.line_search_min_alpha, 1.0e-12, 1.0);
    settings.line_search_extra_trial_count =
        std::max(settings.line_search_extra_trial_count, 0);
    settings.merit_decrease = std::clamp(settings.merit_decrease, 0.0, 0.5);
    settings.best_line_search_trial_growth_tolerance = std::max(
        0.0,
        settings.best_line_search_trial_growth_tolerance);
    settings.ksp_max_iterations = std::max(settings.ksp_max_iterations, 1);

    result.unknowns = detail::clone_vec(initial_state.unknowns);
    result.load_parameter = initial_state.load_parameter;
    const PetscInt n = detail::petsc_vec_size(result.unknowns.get());
    if (n <= 0) {
        result.status = BorderedMixedControlNewtonStatus::invalid_evaluation;
        return result;
    }
    MPI_Comm solve_comm = detail::petsc_object_comm(
        reinterpret_cast<PetscObject>(result.unknowns.get()));
    petsc::OwnedKSP ksp;
    FALL_N_PETSC_CHECK(KSPCreate(solve_comm, ksp.ptr()));
    auto augmented = detail::create_augmented_mat(solve_comm, n);
    auto rhs = detail::create_augmented_vec(solve_comm, n);
    auto step = detail::create_augmented_vec(solve_comm, n);
    auto trial_unknowns = detail::duplicate_vec(result.unknowns.get());
    result.augmented_system_allocations = 4;

    for (int iter = 0; iter < settings.max_iterations; ++iter) {
        auto eval = evaluator(PetscBorderedMixedControlState{
            .unknowns = result.unknowns.get(),
            .load_parameter = result.load_parameter});
        ++result.evaluator_calls;
        if (!detail::petsc_bordered_eval_has_consistent_dimensions(eval, n)) {
            result.status =
                BorderedMixedControlNewtonStatus::invalid_evaluation;
            return result;
        }

        const double residual_norm =
            detail::petsc_vec_norm(eval.residual.get());
        const double constraint_abs = std::abs(eval.constraint);
        if (residual_norm <= settings.residual_tolerance &&
            constraint_abs <= settings.constraint_tolerance)
        {
            result.status = BorderedMixedControlNewtonStatus::converged;
            result.iterations = iter;
            result.residual_norm = residual_norm;
            result.constraint_abs = constraint_abs;
            result.correction_norm = 0.0;
            return result;
        }

        FALL_N_PETSC_CHECK(MatZeroEntries(augmented.get()));
        FALL_N_PETSC_CHECK(VecSet(rhs.get(), 0.0));
        FALL_N_PETSC_CHECK(VecSet(step.get(), 0.0));
        detail::assemble_petsc_bordered_matrix(eval, augmented.get());
        detail::assemble_petsc_bordered_rhs(eval, rhs.get());

        detail::configure_bordered_ksp(ksp.get(), augmented.get(), settings);
        FALL_N_PETSC_CHECK(KSPSolve(ksp.get(), rhs.get(), step.get()));
        KSPConvergedReason ksp_reason{KSP_CONVERGED_ITERATING};
        PetscInt ksp_iterations = 0;
        FALL_N_PETSC_CHECK(KSPGetConvergedReason(ksp.get(), &ksp_reason));
        FALL_N_PETSC_CHECK(KSPGetIterationNumber(ksp.get(), &ksp_iterations));
        result.last_ksp_reason = ksp_reason;
        result.last_ksp_iterations = static_cast<int>(ksp_iterations);
        result.total_ksp_iterations += static_cast<int>(ksp_iterations);
        if (ksp_reason < 0) {
            result.status =
                BorderedMixedControlNewtonStatus::singular_augmented_system;
            result.iterations = iter;
            result.residual_norm = residual_norm;
            result.constraint_abs = constraint_abs;
            return result;
        }

        const double dlambda =
            detail::read_augmented_load_step(step.get(), n);
        const double correction_norm =
            detail::augmented_correction_norm(step.get(), n);

        double alpha = 1.0;
        if (settings.line_search_enabled) {
            const double old_merit = detail::petsc_bordered_merit(
                eval,
                settings.line_search_merit);
            bool accepted = false;
            bool best_valid = false;
            double best_alpha = alpha;
            double best_merit = std::numeric_limits<double>::infinity();
            const auto try_alpha = [&](double trial_alpha) {
                if (!std::isfinite(trial_alpha) ||
                    trial_alpha < settings.line_search_min_alpha ||
                    trial_alpha > 1.0)
                {
                    return false;
                }
                detail::make_trial_unknowns_inplace(
                    result.unknowns.get(),
                    step.get(),
                    trial_alpha,
                    trial_unknowns.get());
                auto trial_eval = evaluator(PetscBorderedMixedControlState{
                    .unknowns = trial_unknowns.get(),
                    .load_parameter =
                        result.load_parameter + trial_alpha * dlambda});
                ++result.evaluator_calls;
                if (!detail::petsc_bordered_eval_has_consistent_dimensions(
                        trial_eval, n))
                {
                    return false;
                }

                const double trial_merit = detail::petsc_bordered_merit(
                    trial_eval,
                    settings.line_search_merit);
                if (std::isfinite(trial_merit) &&
                    (!best_valid || trial_merit < best_merit))
                {
                    best_valid = true;
                    best_alpha = trial_alpha;
                    best_merit = trial_merit;
                }
                const double target =
                    (1.0 - settings.merit_decrease * trial_alpha) *
                    old_merit;
                if (trial_merit <= target || trial_merit <= old_merit) {
                    alpha = trial_alpha;
                    return true;
                }
                return false;
            };
            for (int cutback = 0;
                 cutback <= settings.max_line_search_cutbacks;
                 ++cutback)
            {
                if (try_alpha(alpha)) {
                    accepted = true;
                    break;
                }
                alpha *= settings.line_search_cutback_factor;
                if (alpha < settings.line_search_min_alpha) {
                    break;
                }
            }
            if (!accepted && settings.line_search_extra_trial_count > 0) {
                const int samples = settings.line_search_extra_trial_count;
                for (int i = samples; i >= 1; --i) {
                    const double trial_alpha =
                        static_cast<double>(i) /
                        static_cast<double>(samples + 1);
                    if (try_alpha(trial_alpha)) {
                        accepted = true;
                        break;
                    }
                }
            }
            if (!accepted && settings.accept_best_line_search_trial &&
                best_valid)
            {
                const double allowed =
                    (1.0 + settings.best_line_search_trial_growth_tolerance) *
                    old_merit;
                if (best_merit <= allowed) {
                    alpha = best_alpha;
                    accepted = true;
                }
            }
            if (!accepted) {
                result.status =
                    BorderedMixedControlNewtonStatus::line_search_failed;
                result.iterations = iter;
                result.residual_norm = residual_norm;
                result.constraint_abs = constraint_abs;
                result.correction_norm = correction_norm;
                return result;
            }
        }

        detail::apply_augmented_step_to_unknowns(
            result.unknowns.get(),
            step.get(),
            alpha);
        result.load_parameter += alpha * dlambda;
        result.records.push_back(
            BorderedMixedControlNewtonIterationRecord{
                .iteration = iter + 1,
                .residual_norm = residual_norm,
                .constraint_abs = constraint_abs,
                .correction_norm = correction_norm,
                .load_correction_abs = std::abs(dlambda),
                .line_search_alpha = alpha});
        result.iterations = iter + 1;
        result.residual_norm = residual_norm;
        result.constraint_abs = constraint_abs;
        result.correction_norm = correction_norm;

        if (correction_norm <= settings.correction_tolerance &&
            residual_norm <= 10.0 * settings.residual_tolerance &&
            constraint_abs <= 10.0 * settings.constraint_tolerance)
        {
            result.status = BorderedMixedControlNewtonStatus::converged;
            return result;
        }
    }

    auto eval = evaluator(PetscBorderedMixedControlState{
        .unknowns = result.unknowns.get(),
        .load_parameter = result.load_parameter});
    ++result.evaluator_calls;
    if (detail::petsc_bordered_eval_has_consistent_dimensions(eval, n)) {
        result.residual_norm = detail::petsc_vec_norm(eval.residual.get());
        result.constraint_abs = std::abs(eval.constraint);
        if (result.residual_norm <= settings.residual_tolerance &&
            result.constraint_abs <= settings.constraint_tolerance)
        {
            result.status = BorderedMixedControlNewtonStatus::converged;
            return result;
        }
    }
    result.status = BorderedMixedControlNewtonStatus::max_iterations;
    return result;
}

} // namespace fall_n

#endif // FALL_N_SRC_ANALYSIS_PETSC_BORDERED_MIXED_CONTROL_NEWTON_HH
