// Plan v2 §Fase 4C verification — UpscalingResult primitive smoke test.

#include <cassert>
#include <print>

#include "src/analysis/MultiscaleTypes.hh"

int main() {
    using fall_n::UpscalingResult;
    using fall_n::ResponseStatus;
    using fall_n::TangentLinearizationScheme;
    using fall_n::CondensedTangentStatus;

    // Default-constructed: not well-formed, not gated.
    UpscalingResult empty{};
    assert(!empty.is_well_formed());
    assert(!empty.passes_guarded_smoke_gate());

    // Well-formed elastic stub: 6x6 (axial+bending+shear+torsion package).
    UpscalingResult ok{};
    ok.eps_ref = Eigen::VectorXd::Zero(6);
    ok.f_hom   = Eigen::VectorXd::Zero(6);
    ok.D_hom   = Eigen::MatrixXd::Identity(6, 6);
    ok.converged = true;
    ok.snes_iters = 3;
    ok.frobenius_residual = 1.0e-6;
    ok.status = ResponseStatus::Ok;
    ok.tangent_scheme = TangentLinearizationScheme::LinearizedCondensation;
    ok.condensed_status = CondensedTangentStatus::Success;

    assert(ok.is_well_formed());
    assert(ok.passes_guarded_smoke_gate());           // default gate
    assert(ok.passes_guarded_smoke_gate(0.03, 6));     // explicit gate
    assert(!ok.passes_guarded_smoke_gate(0.03, 2));    // tighter SNES budget

    // Non-converged → fails gate even with low residual.
    UpscalingResult diverged = ok;
    diverged.converged = false;
    assert(!diverged.passes_guarded_smoke_gate());

    // SolveFailed status fails gate.
    UpscalingResult failed = ok;
    failed.status = ResponseStatus::SolveFailed;
    assert(!failed.passes_guarded_smoke_gate());

    // Frobenius residual above threshold fails gate.
    UpscalingResult drifty = ok;
    drifty.frobenius_residual = 0.05;
    assert(!drifty.passes_guarded_smoke_gate(0.03, 6));

    // Mismatched dimensions are rejected by is_well_formed.
    UpscalingResult mismatch = ok;
    mismatch.D_hom = Eigen::MatrixXd::Identity(5, 6);
    assert(!mismatch.is_well_formed());
    assert(!mismatch.passes_guarded_smoke_gate());

    std::println("[multiscale_types] UpscalingResult ALL PASS");
    return 0;
}
