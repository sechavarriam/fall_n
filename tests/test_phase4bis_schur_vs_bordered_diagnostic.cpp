// Plan v2 §Fase 4-bis — Schur condensation vs bordered mixed-control
// diagnostic comparator (analytical elastic stub).
//
// For an elastic local sub-model with stiffness K partitioned into
// boundary (B) and interior (I) DOFs:
//
//     [ K_BB  K_BI ] [ u_B ]   [ f_B ]
//     [ K_IB  K_II ] [ u_I ] = [ 0   ]
//
// the homogenised tangent obtained by Schur condensation is
//
//     D_hom_schur = K_BB - K_BI * K_II^-1 * K_IB
//
// The bordered mixed-control mechanism imposes u_B as control and
// solves for the resulting f_B; for an elastic problem (linear, no
// active set) the resulting tangent should agree with `D_hom_schur` to
// numerical precision.
//
// This test wires both routes against a synthetic 6-boundary / 12-
// interior elastic system and asserts:
//   - both methods produce well-formed UpscalingResult
//   - frobenius_residual < 0.005  (tighter than the 0.03 guarded-smoke gate)
//   - the per-mode peak relative gap is below 1e-9
//
// The synthetic K is a positive-definite shifted mass matrix; this
// keeps the test deterministic and dependency-free.

#include <cassert>
#include <cmath>
#include <cstdio>

#include <Eigen/Dense>

#include "src/analysis/MultiscaleTypes.hh"

using fall_n::CondensedTangentStatus;
using fall_n::ResponseStatus;
using fall_n::TangentLinearizationScheme;
using fall_n::UpscalingResult;

namespace {

Eigen::MatrixXd make_spd(int n, double diag_shift, unsigned seed) {
    std::srand(seed);
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(n, n);
    Eigen::MatrixXd K = A.transpose() * A;
    K += diag_shift * Eigen::MatrixXd::Identity(n, n);
    return K;
}

UpscalingResult schur_condense(const Eigen::MatrixXd& K_BB,
                               const Eigen::MatrixXd& K_BI,
                               const Eigen::MatrixXd& K_IB,
                               const Eigen::MatrixXd& K_II) {
    UpscalingResult r{};
    Eigen::LLT<Eigen::MatrixXd> chol(K_II);
    if (chol.info() != Eigen::Success) {
        r.status = ResponseStatus::SolveFailed;
        r.condensed_status = CondensedTangentStatus::FactorizationFailed;
        return r;
    }
    r.D_hom = K_BB - K_BI * chol.solve(K_IB);
    r.f_hom = Eigen::VectorXd::Zero(K_BB.rows());
    r.eps_ref = Eigen::VectorXd::Zero(K_BB.rows());
    r.converged = true;
    r.snes_iters = 1;
    r.frobenius_residual = 0.0;
    r.status = ResponseStatus::Ok;
    r.tangent_scheme = TangentLinearizationScheme::LinearizedCondensation;
    r.condensed_status = CondensedTangentStatus::Success;
    return r;
}

UpscalingResult bordered_solve(const Eigen::MatrixXd& K_BB,
                               const Eigen::MatrixXd& K_BI,
                               const Eigen::MatrixXd& K_IB,
                               const Eigen::MatrixXd& K_II) {
    // For an elastic problem the bordered mixed-control tangent is
    // recovered column-by-column from probe displacements u_B = e_k.
    UpscalingResult r{};
    const Eigen::Index nB = K_BB.rows();
    Eigen::MatrixXd D(nB, nB);
    Eigen::LLT<Eigen::MatrixXd> chol(K_II);
    if (chol.info() != Eigen::Success) {
        r.status = ResponseStatus::SolveFailed;
        return r;
    }
    for (Eigen::Index k = 0; k < nB; ++k) {
        Eigen::VectorXd uB = Eigen::VectorXd::Zero(nB);
        uB(k) = 1.0;
        Eigen::VectorXd uI = -chol.solve(K_IB * uB);
        Eigen::VectorXd fB = K_BB * uB + K_BI * uI;
        D.col(k) = fB;
    }
    r.D_hom = D;
    r.f_hom = Eigen::VectorXd::Zero(nB);
    r.eps_ref = Eigen::VectorXd::Zero(nB);
    r.converged = true;
    r.snes_iters = static_cast<std::size_t>(nB);
    r.frobenius_residual = 0.0;
    r.status = ResponseStatus::Ok;
    r.tangent_scheme = TangentLinearizationScheme::LinearizedCondensation;
    r.condensed_status = CondensedTangentStatus::Success;
    return r;
}

}  // namespace

int main() {
    constexpr int nB = 6;
    constexpr int nI = 12;
    const Eigen::MatrixXd K = make_spd(nB + nI, 1.0e-1, 42u);

    const Eigen::MatrixXd K_BB = K.topLeftCorner(nB, nB);
    const Eigen::MatrixXd K_BI = K.topRightCorner(nB, nI);
    const Eigen::MatrixXd K_IB = K.bottomLeftCorner(nI, nB);
    const Eigen::MatrixXd K_II = K.bottomRightCorner(nI, nI);

    UpscalingResult r_schur    = schur_condense(K_BB, K_BI, K_IB, K_II);
    UpscalingResult r_bordered = bordered_solve(K_BB, K_BI, K_IB, K_II);

    assert(r_schur.is_well_formed());
    assert(r_bordered.is_well_formed());
    assert(r_schur.passes_guarded_smoke_gate(0.005, 1));
    assert(r_bordered.passes_guarded_smoke_gate(0.005, nB));

    const double frob_gap =
        (r_bordered.D_hom - r_schur.D_hom).norm() / r_schur.D_hom.norm();
    const double max_abs_gap =
        (r_bordered.D_hom - r_schur.D_hom).cwiseAbs().maxCoeff();

    std::printf("[schur_vs_bordered_diagnostic] frob_gap=%.3e max_abs_gap=%.3e\n",
                frob_gap, max_abs_gap);
    assert(frob_gap < 1.0e-9);
    assert(max_abs_gap < 1.0e-9);

    // Symmetry check (elastic problem ⇒ D_hom symmetric).
    const double sym_gap =
        (r_schur.D_hom - r_schur.D_hom.transpose()).norm() /
        r_schur.D_hom.norm();
    assert(sym_gap < 1.0e-9);

    std::printf("[schur_vs_bordered_diagnostic] ALL PASS\n");
    return 0;
}
