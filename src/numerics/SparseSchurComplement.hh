#ifndef FALL_N_SPARSE_SCHUR_COMPLEMENT_HH
#define FALL_N_SPARSE_SCHUR_COMPLEMENT_HH

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>

namespace condensation {

enum class SparseSchurStatus {
    Success,
    InvalidArguments,
    FactorizationFailed,
    SolveFailed,
    ResidualTooLarge
};

template <typename SparseMatrixT = Eigen::SparseMatrix<double>>
class SparseSchurComplementWorkspace {
public:
    using StorageIndex = typename SparseMatrixT::StorageIndex;
    using SolverT = Eigen::SparseLU<SparseMatrixT>;

    void reset()
    {
        pattern_ready_ = false;
        pattern_reused_last_call_ = false;
        symbolic_factorizations_ = 0;
        rows_ = -1;
        cols_ = -1;
        outer_index_cache_.clear();
        inner_index_cache_.clear();
    }

    [[nodiscard]] std::size_t symbolic_factorizations() const noexcept
    {
        return symbolic_factorizations_;
    }

    [[nodiscard]] bool pattern_reused_last_call() const noexcept
    {
        return pattern_reused_last_call_;
    }

    [[nodiscard]] bool prepare_pattern(const SparseMatrixT& matrix)
    {
        const bool reuse = pattern_matches_(matrix);
        if (!reuse) {
            solver_.analyzePattern(matrix);
            rows_ = matrix.rows();
            cols_ = matrix.cols();
            outer_index_cache_.assign(
                matrix.outerIndexPtr(),
                matrix.outerIndexPtr() + matrix.outerSize() + 1);
            inner_index_cache_.assign(
                matrix.innerIndexPtr(),
                matrix.innerIndexPtr() + matrix.nonZeros());
            pattern_ready_ = true;
            ++symbolic_factorizations_;
        }

        pattern_reused_last_call_ = reuse;
        return reuse;
    }

    SolverT& solver() noexcept { return solver_; }

private:
    [[nodiscard]] bool pattern_matches_(const SparseMatrixT& matrix) const
    {
        if (!pattern_ready_
            || rows_ != matrix.rows()
            || cols_ != matrix.cols()
            || outer_index_cache_.size()
                   != static_cast<std::size_t>(matrix.outerSize() + 1)
            || inner_index_cache_.size()
                   != static_cast<std::size_t>(matrix.nonZeros()))
        {
            return false;
        }

        for (Eigen::Index i = 0; i < matrix.outerSize() + 1; ++i) {
            if (outer_index_cache_[static_cast<std::size_t>(i)]
                != matrix.outerIndexPtr()[i])
            {
                return false;
            }
        }

        for (Eigen::Index i = 0; i < matrix.nonZeros(); ++i) {
            if (inner_index_cache_[static_cast<std::size_t>(i)]
                != matrix.innerIndexPtr()[i])
            {
                return false;
            }
        }

        return true;
    }

    SolverT solver_{};
    bool pattern_ready_{false};
    bool pattern_reused_last_call_{false};
    std::size_t symbolic_factorizations_{0};
    Eigen::Index rows_{-1};
    Eigen::Index cols_{-1};
    std::vector<StorageIndex> outer_index_cache_{};
    std::vector<StorageIndex> inner_index_cache_{};
};

template <typename SparseMatrixT = Eigen::SparseMatrix<double>>
struct SparseSchurResult {
    SparseSchurStatus status{SparseSchurStatus::InvalidArguments};
    Eigen::MatrixXd condensed_times_transfer{};
    double solve_residual{0.0};
    bool pattern_reused{false};
    std::size_t symbolic_factorizations{0};
};

template <typename SparseMatrixT = Eigen::SparseMatrix<double>>
inline auto apply_condensed_operator(
    const SparseMatrixT& K_ff_input,
    const SparseMatrixT& K_cf_input,
    const Eigen::Ref<const Eigen::MatrixXd>& K_fc_times_transfer,
    const Eigen::Ref<const Eigen::MatrixXd>& K_cc_times_transfer,
    double residual_tolerance = 1.0e-8,
    SparseSchurComplementWorkspace<SparseMatrixT>* workspace = nullptr)
    -> SparseSchurResult<SparseMatrixT>
{
    SparseSchurResult<SparseMatrixT> result;

    if (K_ff_input.rows() != K_ff_input.cols()
        || K_cf_input.cols() != K_ff_input.rows()
        || K_fc_times_transfer.rows() != K_ff_input.rows()
        || K_cf_input.rows() != K_cc_times_transfer.rows()
        || K_fc_times_transfer.cols() != K_cc_times_transfer.cols())
    {
        return result;
    }

    result.condensed_times_transfer = K_cc_times_transfer;
    if (K_ff_input.rows() == 0) {
        result.status = SparseSchurStatus::Success;
        return result;
    }

    const SparseMatrixT* K_ff_ptr = &K_ff_input;
    const SparseMatrixT* K_cf_ptr = &K_cf_input;
    SparseMatrixT K_ff_storage;
    SparseMatrixT K_cf_storage;

    if (!K_ff_input.isCompressed()) {
        K_ff_storage = K_ff_input;
        K_ff_storage.makeCompressed();
        K_ff_ptr = &K_ff_storage;
    }
    if (!K_cf_input.isCompressed()) {
        K_cf_storage = K_cf_input;
        K_cf_storage.makeCompressed();
        K_cf_ptr = &K_cf_storage;
    }

    SparseSchurComplementWorkspace<SparseMatrixT> local_workspace;
    auto& active_workspace = workspace ? *workspace : local_workspace;

    result.pattern_reused = active_workspace.prepare_pattern(*K_ff_ptr);
    result.symbolic_factorizations =
        active_workspace.symbolic_factorizations();

    active_workspace.solver().factorize(*K_ff_ptr);
    if (active_workspace.solver().info() != Eigen::Success) {
        result.status = SparseSchurStatus::FactorizationFailed;
        return result;
    }

    const Eigen::MatrixXd free_response =
        active_workspace.solver().solve(K_fc_times_transfer);
    if (active_workspace.solver().info() != Eigen::Success
        || !free_response.allFinite())
    {
        result.status = SparseSchurStatus::SolveFailed;
        return result;
    }

    result.solve_residual =
        ((*K_ff_ptr) * free_response - K_fc_times_transfer).norm()
        / std::max(1.0, K_fc_times_transfer.norm());
    if (!std::isfinite(result.solve_residual)
        || result.solve_residual > residual_tolerance)
    {
        result.status = SparseSchurStatus::ResidualTooLarge;
        return result;
    }

    result.condensed_times_transfer.noalias() -= (*K_cf_ptr) * free_response;
    if (!result.condensed_times_transfer.allFinite()) {
        result.status = SparseSchurStatus::SolveFailed;
        return result;
    }

    result.status = SparseSchurStatus::Success;
    return result;
}

} // namespace condensation

#endif // FALL_N_SPARSE_SCHUR_COMPLEMENT_HH
