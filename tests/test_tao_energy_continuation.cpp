// =============================================================================
//  test_tao_energy_continuation.cpp   (LIBS CORE)
//
//  TaoEnergyContinuation minimises the incremental potential instead of hunting a
//  residual zero.  Two checks:
//    A. Quadratic SPD  Pi = 1/2 uT A u - bT u  ->  TAO lands on A^-1 b.
//    B. Double well    Pi = (u^2-1)^2          ->  from u0=0.1 TAO descends to a
//       MINIMISER (|u|=1) and never stops at the maximum u=0, even though u=0 is
//       a residual zero (R = 4u(u^2-1) = 0 there).  This is the whole point of an
//       energy-driven step over a residual-driven one at a softening reversal.
// =============================================================================

#include "src/analysis/TaoEnergyContinuation.hh"
#include "src/petsc/PetscRaii.hh"

#include <petsc.h>

#include <cmath>
#include <cstdio>

using fall_n::TaoEnergyContinuation;

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

// Pi(u) = 1/2 (2 u0^2 + 3 u1^2) - (2 u0 + 3 u1);  A = diag(2,3), b = (2,3).
struct QuadraticBackend {
    static constexpr PetscInt N = 2;
    [[nodiscard]] petsc::OwnedVec create_vector() const {
        Vec v = nullptr;
        VecCreateSeq(PETSC_COMM_SELF, N, &v);
        VecSet(v, 0.0);
        return petsc::OwnedVec(v);
    }
    [[nodiscard]] petsc::OwnedMat create_matrix() const {
        Mat k = nullptr;
        MatCreateSeqDense(PETSC_COMM_SELF, N, N, nullptr, &k);
        MatZeroEntries(k);
        return petsc::OwnedMat(k);
    }
    void apply_control(double) const {}
    void residual(Vec u, Vec r) const {
        const PetscScalar* a = nullptr;
        VecGetArrayRead(u, &a);
        const double u0 = PetscRealPart(a[0]), u1 = PetscRealPart(a[1]);
        VecRestoreArrayRead(u, &a);
        VecSetValue(r, 0, 2.0 * u0 - 2.0, INSERT_VALUES);
        VecSetValue(r, 1, 3.0 * u1 - 3.0, INSERT_VALUES);
        VecAssemblyBegin(r);
        VecAssemblyEnd(r);
    }
    void tangent(Vec, Mat k) const {
        MatSetValue(k, 0, 0, 2.0, INSERT_VALUES);
        MatSetValue(k, 1, 1, 3.0, INSERT_VALUES);
        MatSetValue(k, 0, 1, 0.0, INSERT_VALUES);
        MatSetValue(k, 1, 0, 0.0, INSERT_VALUES);
        MatAssemblyBegin(k, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(k, MAT_FINAL_ASSEMBLY);
    }
    void accept(Vec, double) const {}
    [[nodiscard]] double incremental_energy(Vec u, Vec uref) const {
        return pi(u) - pi(uref);
    }
    static double pi(Vec u) {
        const PetscScalar* a = nullptr;
        VecGetArrayRead(u, &a);
        const double u0 = PetscRealPart(a[0]), u1 = PetscRealPart(a[1]);
        VecRestoreArrayRead(u, &a);
        return 0.5 * (2.0 * u0 * u0 + 3.0 * u1 * u1) - (2.0 * u0 + 3.0 * u1);
    }
};

// Pi(u) = (u^2 - 1)^2 ;  R = 4u(u^2-1) ;  K = 12u^2 - 4.
struct DoubleWellBackend {
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
    void residual(Vec u, Vec r) const {
        const double x = get1(u);
        set1(r, 4.0 * x * (x * x - 1.0));
    }
    void tangent(Vec u, Mat k) const {
        const double x = get1(u);
        MatSetValue(k, 0, 0, 12.0 * x * x - 4.0, INSERT_VALUES);
        MatAssemblyBegin(k, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(k, MAT_FINAL_ASSEMBLY);
    }
    void accept(Vec, double) const {}
    [[nodiscard]] double incremental_energy(Vec u, Vec uref) const {
        const double a = get1(u), b = get1(uref);
        return (a * a - 1.0) * (a * a - 1.0) - (b * b - 1.0) * (b * b - 1.0);
    }
};

}  // namespace

int main(int argc, char** argv) {
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    // --- A. Quadratic SPD -> A^-1 b = (1,1) ---------------------------------
    {
        TaoEnergyContinuation<QuadraticBackend> solver(QuadraticBackend{});
        const auto r = solver.advance_to(0.0);
        const PetscScalar* a = nullptr;
        VecGetArrayRead(solver.solution(), &a);
        const double u0 = PetscRealPart(a[0]), u1 = PetscRealPart(a[1]);
        VecRestoreArrayRead(solver.solution(), &a);
        std::printf("A  quadratic   u=(%.6f, %.6f)  |R|=%.3e  conv=%d\n",
                    u0, u1, r.residual, r.converged);
        check(std::abs(u0 - 1.0) < 1.0e-5 && std::abs(u1 - 1.0) < 1.0e-5,
              "A: TAO minimises to A^-1 b = (1,1)");
        check(r.converged, "A: converged on the residual floor");
    }

    // --- B. Double well from 0.1 -> minimiser, never the maximum ------------
    {
        TaoEnergyContinuation<DoubleWellBackend> solver(DoubleWellBackend{});
        Vec s = nullptr;
        VecCreateSeq(PETSC_COMM_SELF, 1, &s);
        set1(s, 0.1);
        solver.set_initial_guess(s);
        VecDestroy(&s);

        const auto r = solver.advance_to(0.0);
        const double u = get1(solver.solution());
        std::printf("B  double-well u=%.6f  |R|=%.3e  conv=%d\n",
                    u, r.residual, r.converged);
        check(std::abs(std::abs(u) - 1.0) < 1.0e-4,
              "B: TAO descends to a well minimiser |u|=1");
        check(std::abs(u) > 0.5,
              "B: TAO does NOT stall at the maximum u=0 (a residual zero)");
    }

    std::printf("%s (%d failure%s)\n", g_failed ? "TEST FAILED" : "TEST PASSED",
                g_failed, g_failed == 1 ? "" : "s");

    PetscFinalize();
    return g_failed;
}
