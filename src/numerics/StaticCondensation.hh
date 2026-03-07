#ifndef FALL_N_STATIC_CONDENSATION_HH
#define FALL_N_STATIC_CONDENSATION_HH

// ============================================================================
//  Static Condensation  –  Pure Linear-Algebra Module
// ============================================================================
//
//  Given a partitioned system
//
//     ┌          ┐ ┌    ┐   ┌    ┐
//     │ K_ee K_ei│ │ u_e│   │ f_e│
//     │ K_ie K_ii│ │ u_i│ = │ f_i│
//     └          ┘ └    ┘   └    ┘
//
//  where subscript 'e' denotes external (exposed) DOFs and 'i' denotes
//  internal (condensed-out) DOFs, the condensation eliminates u_i to
//  obtain a reduced system
//
//      K̂ · u_e = f̂
//
//  with
//      K̂ = K_ee – K_ei · K_ii⁻¹ · K_ie          (Schur complement)
//      f̂ = f_e – K_ei · K_ii⁻¹ · f_i
//
//  After the global solve yields u_e, internal DOFs are recovered via
//
//      u_i = K_ii⁻¹ · (f_i – K_ie · u_e)
//
//  This module is intentionally dependency-free except for Eigen.
//  It does NOT know about elements, PETSc, or any other FEM object.
// ============================================================================

#include <cstddef>
#include <stdexcept>

#include <Eigen/Dense>

namespace condensation {

// ── Result type returned by condense() ──────────────────────────────────────

struct CondensedSystem {
    Eigen::MatrixXd K_hat;      // K̂  (n_ext × n_ext)
    Eigen::VectorXd f_hat;      // f̂  (n_ext)

    // Cached intermediates required for recovery
    Eigen::MatrixXd K_ii_inv;   // K_ii⁻¹  (n_int × n_int)
    Eigen::MatrixXd K_ie;       // K_ie     (n_int × n_ext)
    Eigen::VectorXd f_i;        // f_i      (n_int)
};

// ── condense() ──────────────────────────────────────────────────────────────
//
//  K_full : (n × n)  full element stiffness
//  f_full : (n)      full element load vector
//  n_ext  : number of external DOFs (first n_ext rows/cols)
//
//  Convention: the full system is ordered as
//      [ external DOFs  |  internal DOFs ]
//  i.e. external indices come FIRST (rows/cols 0 … n_ext-1).
//
//  Returns a CondensedSystem with the Schur complement and cached data
//  needed by recover_internal().
//
//  Throws std::invalid_argument if dimensions are inconsistent,
//  or if K_ii is singular (determined by a threshold on the determinant
//  of the LDLT decomposition, which is more robust than a raw inverse).

inline auto condense(const Eigen::MatrixXd& K_full,
                     const Eigen::VectorXd& f_full,
                     std::size_t            n_ext) -> CondensedSystem
{
    const auto n = static_cast<std::size_t>(K_full.rows());

    if (static_cast<std::size_t>(K_full.cols()) != n)
        throw std::invalid_argument("condense: K_full must be square.");

    if (static_cast<std::size_t>(f_full.size()) != n)
        throw std::invalid_argument("condense: f_full size must match K_full dimension.");

    if (n_ext == 0 || n_ext >= n)
        throw std::invalid_argument("condense: n_ext must satisfy 0 < n_ext < n.");

    const auto n_int = n - n_ext;

    const auto ne = static_cast<Eigen::Index>(n_ext);
    const auto ni = static_cast<Eigen::Index>(n_int);

    // ── Extract sub-blocks ──────────────────────────────────────────────
    Eigen::MatrixXd K_ee = K_full.topLeftCorner    (ne, ne);
    Eigen::MatrixXd K_ei = K_full.topRightCorner   (ne, ni);
    Eigen::MatrixXd K_ie = K_full.bottomLeftCorner (ni, ne);
    Eigen::MatrixXd K_ii = K_full.bottomRightCorner(ni, ni);

    Eigen::VectorXd f_e = f_full.head(ne);
    Eigen::VectorXd f_i = f_full.tail(ni);

    // ── Invert K_ii via robust LDLT decomposition ───────────────────────
    Eigen::LDLT<Eigen::MatrixXd> ldlt(K_ii);

    if (ldlt.info() != Eigen::Success || !ldlt.isPositive()
        || ldlt.vectorD().minCoeff() <= 0.0)
        throw std::runtime_error(
            "condense: K_ii is singular or not positive-definite; "
            "cannot condense.");

    // Compute K_ii⁻¹ explicitly — the dimension is typically small
    // (number of internal element DOFs), so the explicit inverse is fine.
    Eigen::MatrixXd K_ii_inv = ldlt.solve(
        Eigen::MatrixXd::Identity(ni, ni));

    // ── Schur complement ────────────────────────────────────────────────
    // K̂ = K_ee – K_ei · K_ii⁻¹ · K_ie
    Eigen::MatrixXd K_hat = K_ee - K_ei * K_ii_inv * K_ie;

    // f̂ = f_e – K_ei · K_ii⁻¹ · f_i
    Eigen::VectorXd f_hat = f_e - K_ei * K_ii_inv * f_i;

    return CondensedSystem{
        .K_hat    = std::move(K_hat),
        .f_hat    = std::move(f_hat),
        .K_ii_inv = std::move(K_ii_inv),
        .K_ie     = std::move(K_ie),
        .f_i      = std::move(f_i),
    };
}

// ── recover_internal() ──────────────────────────────────────────────────────
//
//  Given a solved external displacement vector u_ext and the cached
//  CondensedSystem, recover the internal DOFs:
//
//      u_i = K_ii⁻¹ · (f_i – K_ie · u_e)

inline auto recover_internal(const CondensedSystem& cs,
                             const Eigen::VectorXd& u_ext) -> Eigen::VectorXd
{
    if (u_ext.size() != cs.K_hat.rows())
        throw std::invalid_argument(
            "recover_internal: u_ext size must match K_hat dimension.");

    return cs.K_ii_inv * (cs.f_i - cs.K_ie * u_ext);
}

// ── condense (stiffness-only overload, no load vector) ──────────────────────
//
//  For eigenvalue problems or cases where only the stiffness Schur
//  complement is needed.

struct CondensedStiffness {
    Eigen::MatrixXd K_hat;
    Eigen::MatrixXd K_ii_inv;
    Eigen::MatrixXd K_ie;
};

inline auto condense(const Eigen::MatrixXd& K_full,
                     std::size_t            n_ext) -> CondensedStiffness
{
    const auto n = static_cast<std::size_t>(K_full.rows());

    if (static_cast<std::size_t>(K_full.cols()) != n)
        throw std::invalid_argument("condense: K_full must be square.");

    if (n_ext == 0 || n_ext >= n)
        throw std::invalid_argument("condense: n_ext must satisfy 0 < n_ext < n.");

    const auto n_int = n - n_ext;

    const auto ne = static_cast<Eigen::Index>(n_ext);
    const auto ni = static_cast<Eigen::Index>(n_int);

    Eigen::MatrixXd K_ee = K_full.topLeftCorner    (ne, ne);
    Eigen::MatrixXd K_ei = K_full.topRightCorner   (ne, ni);
    Eigen::MatrixXd K_ie = K_full.bottomLeftCorner (ni, ne);
    Eigen::MatrixXd K_ii = K_full.bottomRightCorner(ni, ni);

    Eigen::LDLT<Eigen::MatrixXd> ldlt(K_ii);

    if (ldlt.info() != Eigen::Success || !ldlt.isPositive()
        || ldlt.vectorD().minCoeff() <= 0.0)
        throw std::runtime_error(
            "condense: K_ii is singular or not positive-definite; "
            "cannot condense.");

    Eigen::MatrixXd K_ii_inv = ldlt.solve(
        Eigen::MatrixXd::Identity(ni, ni));

    Eigen::MatrixXd K_hat = K_ee - K_ei * K_ii_inv * K_ie;

    return CondensedStiffness{
        .K_hat    = std::move(K_hat),
        .K_ii_inv = std::move(K_ii_inv),
        .K_ie     = std::move(K_ie),
    };
}

} // namespace condensation

#endif // FALL_N_STATIC_CONDENSATION_HH
