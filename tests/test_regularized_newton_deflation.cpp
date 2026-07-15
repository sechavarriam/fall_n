// =============================================================================
//  test_regularized_newton_deflation.cpp   (LIBS CORE)
//
//  Branch selection by deflation in RegularizedNewtonContinuation. A toy 1-DOF
//  backend R(u) = (u-1)(u-2)(u-4) has stable roots at u=1 and u=4 (R'>0) and an
//  unstable one at u=2. Plain continuation from u0=3.8 lands on root 4; with root
//  4 registered as spurious, deflation must escape it and settle on another root.
//
//  Also checks that the default (NoDeflation) path is unchanged and that
//  capture_state()/restore_state() round-trips the solution.
// =============================================================================

#include "src/analysis/RegularizedNewtonContinuation.hh"
#include "src/petsc/PetscRaii.hh"

#include <petsc.h>

#include <cmath>
#include <cstdio>

using fall_n::LevenbergMarquardt;
using fall_n::NoDeflation;
using fall_n::RegularizedNewtonConfig;
using fall_n::RegularizedNewtonContinuation;
using fall_n::ShermanMorrisonDeflation;

namespace {

int g_failed = 0;
void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? " ok " : "FAIL", what);
    if (!ok) ++g_failed;
}

double get1(Vec v) {
    const PetscScalar* a = nullptr;
    VecGetArrayRead(v, &a);
    const double x = PetscRealPart(a[0]);
    VecRestoreArrayRead(v, &a);
    return x;
}
void set1(Vec v, double val) {
    VecSetValue(v, 0, val, INSERT_VALUES);
    VecAssemblyBegin(v);
    VecAssemblyEnd(v);
}
double cubic(double u) { return (u - 1.0) * (u - 2.0) * (u - 4.0); }

// Toy 1-DOF continuation backend: a fixed cubic residual (control is inert, so
// advance_to() is a root find). Models ContinuationBackend.
struct CubicBackend {
    [[nodiscard]] petsc::OwnedVec create_vector() const {
        Vec v = nullptr;
        VecCreateSeq(PETSC_COMM_SELF, 1, &v);
        VecSet(v, 0.0);
        return petsc::OwnedVec(v);
    }
    [[nodiscard]] petsc::OwnedMat create_matrix() const {
        Mat k = nullptr;
        MatCreateSeqDense(PETSC_COMM_SELF, 1, 1, nullptr, &k);
        MatZeroEntries(k);
        return petsc::OwnedMat(k);
    }
    void apply_control(double) const {}
    void residual(Vec u, Vec r) const { set1(r, cubic(get1(u))); }
    void tangent(Vec u, Mat k) const {
        const double uu = get1(u);
        const double d = 3.0 * uu * uu - 14.0 * uu + 14.0;   // R'(u)
        MatSetValue(k, 0, 0, d, INSERT_VALUES);
        MatAssemblyBegin(k, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(k, MAT_FINAL_ASSEMBLY);
    }
    void accept(Vec, double) const {}
};

// small helper: set the solver's current iterate to a scalar value
template <class Solver>
void seed(Solver& s, double value) {
    Vec v = nullptr;
    VecCreateSeq(PETSC_COMM_SELF, 1, &v);
    set1(v, value);
    s.set_solution(v);
    VecDestroy(&v);
}

}  // namespace

int main(int argc, char** argv) {
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    RegularizedNewtonConfig cfg;
    cfg.max_newton = 200;
    cfg.tol_abs = 1.0e-9;

    // --- A. Baseline: no deflation, u0 = 3.8 -> nearest stable root (u = 4) --
    {
        RegularizedNewtonContinuation<CubicBackend> solver(CubicBackend{}, cfg);
        seed(solver, 3.8);
        const auto r = solver.advance_to(0.0);
        const double u = get1(solver.solution());
        std::printf("A  no-deflation  u=%.6f  R=%.3e  conv=%d\n", u, cubic(u), r.converged);
        check(std::abs(cubic(u)) < 1.0e-6, "A: converges to a root of the cubic");
        check(std::abs(u - 4.0) < 1.0e-3, "A: from 3.8 lands on the high root u=4");
    }

    // --- B. Deflation: register root 4, must escape to another root ----------
    {
        RegularizedNewtonContinuation<CubicBackend, LevenbergMarquardt,
                                      ShermanMorrisonDeflation>
            solver(CubicBackend{}, cfg, LevenbergMarquardt{},
                   ShermanMorrisonDeflation{/*power=*/2.0, /*shift=*/1.0,
                                            /*tau_max=*/50.0});
        seed(solver, 3.8);
        Vec root4 = nullptr;
        VecCreateSeq(PETSC_COMM_SELF, 1, &root4);
        set1(root4, 4.0);
        solver.register_spurious_root(root4);
        VecDestroy(&root4);
        check(solver.spurious_root_count() == 1, "B: one spurious root registered");

        const auto r = solver.advance_to(0.0);
        const double u = get1(solver.solution());
        std::printf("B  deflation     u=%.6f  R=%.3e  conv=%d\n", u, cubic(u), r.converged);
        check(std::abs(cubic(u)) < 1.0e-6, "B: still converges to a root");
        check(std::abs(u - 4.0) > 0.5, "B: deflation escapes the registered root u=4");
    }

    // --- C. capture_state / restore_state round-trips the solution -----------
    {
        RegularizedNewtonContinuation<CubicBackend> solver(CubicBackend{}, cfg);
        seed(solver, 3.8);
        solver.advance_to(0.0);
        const double before = get1(solver.solution());
        const auto snapshot = solver.capture_state();

        seed(solver, 0.5);          // perturb away
        solver.advance_to(0.0);
        solver.restore_state(snapshot);
        const double after = get1(solver.solution());
        std::printf("C  restore       before=%.6f  after=%.6f\n", before, after);
        check(std::abs(after - before) < 1.0e-12, "C: restore_state restores the solution");
    }

    std::printf("%s (%d failure%s)\n", g_failed ? "TEST FAILED" : "TEST PASSED",
                g_failed, g_failed == 1 ? "" : "s");

    PetscFinalize();
    return g_failed;
}
