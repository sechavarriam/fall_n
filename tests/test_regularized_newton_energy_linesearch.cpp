// =============================================================================
//  test_regularized_newton_energy_linesearch.cpp   (LIBS CORE)
//
//  Energy line search in RegularizedNewtonContinuation. On the double well
//  Pi(u) = (u^2-1)^2 the residual R = grad Pi = 4u(u^2-1) has a root at u=0
//  that is a MAXIMUM of Pi: plain LM from u0=0.15 walks straight into it
//  (the residual decreases monotonically toward the saddle). With the energy
//  line search on, every step that raises Pi is backtracked and finally
//  rejected, so the solver refuses the maximum.
//
//  Also checks the healthy paths: descent inside a good basin still converges,
//  the cubic benchmark of the deflation test is unaffected, and with the gate
//  OFF an energy-capable backend reproduces the plain backend bit-exactly.
// =============================================================================

#include "src/analysis/RegularizedNewtonContinuation.hh"
#include "src/petsc/PetscRaii.hh"

#include <petsc.h>

#include <cmath>
#include <cstdio>

using fall_n::RegularizedNewtonConfig;
using fall_n::RegularizedNewtonContinuation;

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

double well_pi(double u) {
    const double q = u * u - 1.0;
    return q * q;
}
double cubic(double u) { return (u - 1.0) * (u - 2.0) * (u - 4.0); }
double cubic_pi(double u) {
    return u * u * u * u / 4.0 - 7.0 * u * u * u / 3.0 + 7.0 * u * u - 8.0 * u;
}

// Toy 1-DOF backends (control inert -> advance_to() is a root find / descent).
struct SeqOneDofBackend {
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
    void accept(Vec, double) const {}
};

// Pi = (u^2-1)^2, R = 4u(u^2-1), K = 12u^2-4. The root u=0 is a maximum of Pi.
struct DoubleWellEnergyBackend : SeqOneDofBackend {
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
    [[nodiscard]] double incremental_energy(Vec u, Vec uref) const {
        return well_pi(get1(u)) - well_pi(get1(uref));
    }
};

// The deflation-test cubic, without and with analytic energy.
struct CubicBackend : SeqOneDofBackend {
    void residual(Vec u, Vec r) const { set1(r, cubic(get1(u))); }
    void tangent(Vec u, Mat k) const {
        const double uu = get1(u);
        MatSetValue(k, 0, 0, 3.0 * uu * uu - 14.0 * uu + 14.0, INSERT_VALUES);
        MatAssemblyBegin(k, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(k, MAT_FINAL_ASSEMBLY);
    }
};
struct CubicEnergyBackend : CubicBackend {
    [[nodiscard]] double incremental_energy(Vec u, Vec uref) const {
        return cubic_pi(get1(u)) - cubic_pi(get1(uref));
    }
};

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

    RegularizedNewtonConfig cfg_ls = cfg;
    cfg_ls.energy_linesearch = true;

    // --- A. Pathology: plain LM from u0=0.15 lands on the MAXIMUM u=0 --------
    {
        RegularizedNewtonContinuation<DoubleWellEnergyBackend> solver(
            DoubleWellEnergyBackend{}, cfg);
        seed(solver, 0.15);
        const auto r = solver.advance_to(0.0);
        const double u = get1(solver.solution());
        std::printf("A  ls-off        u=%.6f  Pi=%.6f  conv=%d\n", u,
                    well_pi(u), r.converged);
        check(std::abs(u) < 1.0e-3, "A: plain LM stalls at the maximum u=0");
        check(r.converged, "A: ...and reports it as a converged root");
    }

    // --- B. Line search refuses the maximum ----------------------------------
    {
        RegularizedNewtonContinuation<DoubleWellEnergyBackend> solver(
            DoubleWellEnergyBackend{}, cfg_ls);
        seed(solver, 0.15);
        const auto r = solver.advance_to(0.0);
        const double u = get1(solver.solution());
        std::printf("B  ls-on         u=%.6f  Pi=%.6f  conv=%d\n", u,
                    well_pi(u), r.converged);
        check(std::abs(u) > 0.05, "B: energy line search refuses the maximum");
        check(well_pi(u) <= well_pi(0.15) + 1.0e-12,
              "B: the energy never increased");
    }

    // --- B2. Inside a good basin the descent is unimpeded ---------------------
    {
        RegularizedNewtonContinuation<DoubleWellEnergyBackend> solver(
            DoubleWellEnergyBackend{}, cfg_ls);
        seed(solver, 0.8);
        const auto r = solver.advance_to(0.0);
        const double u = get1(solver.solution());
        std::printf("B2 ls-on basin   u=%.6f  Pi=%.6f  conv=%d\n", u,
                    well_pi(u), r.converged);
        check(r.converged && std::abs(u - 1.0) < 1.0e-3,
              "B2: from u0=0.8 converges to the minimum u=1");
    }

    // --- C. Healthy cubic benchmark unaffected by the line search ------------
    {
        RegularizedNewtonContinuation<CubicEnergyBackend> solver(
            CubicEnergyBackend{}, cfg_ls);
        seed(solver, 3.8);
        const auto r = solver.advance_to(0.0);
        const double u = get1(solver.solution());
        std::printf("C  ls-on cubic   u=%.6f  R=%.3e  conv=%d\n", u, cubic(u),
                    r.converged);
        check(r.converged && std::abs(u - 4.0) < 1.0e-3,
              "C: from 3.8 still lands on the (energy-downhill) root u=4");
    }

    // --- D. Gate OFF: the energy-capable backend is bit-identical ------------
    {
        RegularizedNewtonContinuation<CubicBackend> plain(CubicBackend{}, cfg);
        RegularizedNewtonContinuation<CubicEnergyBackend> ecap(
            CubicEnergyBackend{}, cfg);
        seed(plain, 3.8);
        seed(ecap, 3.8);
        plain.advance_to(0.0);
        ecap.advance_to(0.0);
        const double u1 = get1(plain.solution());
        const double u2 = get1(ecap.solution());
        std::printf("D  off-identity  plain=%.17g  energy=%.17g\n", u1, u2);
        check(u1 == u2,
              "D: with the gate off the energy backend is bit-identical");
    }

    std::printf("%s (%d failure%s)\n", g_failed ? "TEST FAILED" : "TEST PASSED",
                g_failed, g_failed == 1 ? "" : "s");

    PetscFinalize();
    return g_failed;
}
