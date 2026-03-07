#ifndef FALL_N_ASSEMBLY_POLICY_HH
#define FALL_N_ASSEMBLY_POLICY_HH

// ============================================================================
//  Element Assembly Policies
// ============================================================================
//
//  Two strategies for how an element's full stiffness / load system
//  (which may include internal DOFs) enters the global system:
//
//  ┌─────────────────┬──────────────────────────────────────────────┐
//  │ DirectAssembly   │ Assembles the full K (ext+int DOFs).        │
//  │   (Option B)     │ Internal DOFs are visible globally           │
//  │                  │ (Hu-Washizu, multi-field formulations).     │
//  ├─────────────────┼──────────────────────────────────────────────┤
//  │ CondensedAssembly│ Condenses internal DOFs via Schur complement.│
//  │   (Option A)     │ Only reduced K̂ (ext DOFs) is assembled.     │
//  │                  │ Internal DOFs recovered after global solve. │
//  └─────────────────┴──────────────────────────────────────────────┘
//
//  Both policies satisfy the AssemblyPolicyLike concept,
//  allowing the element to be parametrized on the policy.
//
//  Usage inside an element:
//
//    auto [K_asm, f_asm] = assembly_policy_.prepare(K_full, f_full, n_ext);
//    // ──  assembly_policy_.dof_count()  gives the number of DOFs to assemble
//    // ──  assembly_policy_.recover_internal(u_ext) after global solve
//
// ============================================================================

#include <cassert>
#include <concepts>
#include <cstddef>
#include <optional>
#include <utility>

#include <Eigen/Dense>

#include "../../numerics/StaticCondensation.hh"

namespace assembly {

// ── Prepared system: what gets assembled into the global matrix ──────────────

struct PreparedSystem {
    Eigen::MatrixXd K;   // stiffness to assemble
    Eigen::VectorXd f;   // load vector to assemble
};

// ── Concept ─────────────────────────────────────────────────────────────────

template <typename P>
concept AssemblyPolicyLike = requires(
    P&                      p,
    const Eigen::MatrixXd&  K_full,
    const Eigen::VectorXd&  f_full,
    std::size_t             n_ext,
    const Eigen::VectorXd&  u_ext)
{
    // Compile-time flag: does this policy expose internal DOFs globally?
    { P::exposes_internal_dofs } -> std::convertible_to<bool>;

    // Prepare the (K, f) for assembly — may condense or pass through.
    { p.prepare(K_full, f_full, n_ext) } -> std::same_as<PreparedSystem>;

    // After the global solve, recover internal DOFs (no-op for DirectAssembly).
    { p.recover_internal(u_ext) } -> std::same_as<Eigen::VectorXd>;
};

// ============================================================================
//  DirectAssembly — Option B (Hu-Washizu / multi-field)
// ============================================================================
//
//  The full element system (external + internal DOFs) is assembled into
//  the global stiffness matrix.  PetscSection assigns DOFs to both
//  vertices (nodes) and cells (elements).
//
//  prepare()          → returns K_full unchanged (no condensation).
//  recover_internal() → no-op (internal DOFs are solved globally).

struct DirectAssembly {
    static constexpr bool exposes_internal_dofs = true;

    auto prepare(const Eigen::MatrixXd& K_full,
                 const Eigen::VectorXd& f_full,
                 [[maybe_unused]] std::size_t n_ext) -> PreparedSystem
    {
        return PreparedSystem{K_full, f_full};
    }

    auto recover_internal([[maybe_unused]] const Eigen::VectorXd& u_ext)
        -> Eigen::VectorXd
    {
        return {};   // nothing to recover — all DOFs are in the global system
    }
};

// ============================================================================
//  CondensedAssembly — Option A (static condensation)
// ============================================================================
//
//  Condenses internal DOFs out via the Schur complement (see
//  StaticCondensation.hh).  Only the reduced K̂ (n_ext × n_ext) is
//  assembled into the global matrix.  After the global solve delivers
//  u_ext, recover_internal() back-substitutes to obtain u_int.
//
//  The cache (CondensedSystem) is stored between prepare() and
//  recover_internal() — it is per-element, per-iteration data.

struct CondensedAssembly {
    static constexpr bool exposes_internal_dofs = false;

    auto prepare(const Eigen::MatrixXd& K_full,
                 const Eigen::VectorXd& f_full,
                 std::size_t            n_ext) -> PreparedSystem
    {
        cache_ = condensation::condense(K_full, f_full, n_ext);
        return PreparedSystem{cache_->K_hat, cache_->f_hat};   // copies
    }

    auto recover_internal(const Eigen::VectorXd& u_ext) -> Eigen::VectorXd
    {
        assert(cache_.has_value() &&
               "CondensedAssembly::recover_internal called before prepare()");
        auto u_int = condensation::recover_internal(*cache_, u_ext);
        cache_.reset();   // consumed — force re-preparation next iteration
        return u_int;
    }

    [[nodiscard]] bool has_cache() const noexcept { return cache_.has_value(); }

private:
    std::optional<condensation::CondensedSystem> cache_;
};

// ── Static assertions ───────────────────────────────────────────────────────

static_assert(AssemblyPolicyLike<DirectAssembly>,
              "DirectAssembly must satisfy AssemblyPolicyLike.");

static_assert(AssemblyPolicyLike<CondensedAssembly>,
              "CondensedAssembly must satisfy AssemblyPolicyLike.");

} // namespace assembly

#endif // FALL_N_ASSEMBLY_POLICY_HH
