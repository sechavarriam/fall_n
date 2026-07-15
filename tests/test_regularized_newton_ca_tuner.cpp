// =============================================================================
//  test_regularized_newton_ca_tuner.cpp   (LIBS CORE)
//
//  Phase P3 in isolation: the Cultural Algorithm (Phase M) tunes the
//  Levenberg-Marquardt continuation config (Phase P1) on a problem where the
//  DEFAULT config stalls.
//
//  The 1-DOF cubic R(u)=(u-1)(u-2)(u-4) from u0=3.5 stalls under the default LM:
//  the Newton step overshoots root 4 to ~4.5, where the residual is larger, and
//  the monotone-residual acceptance rejects it. More regularisation (a larger
//  mu_max_frac) shrinks the step so it lands below 4.5 and the iteration accepts
//  and converges. The CA searches the config box and must FIND such a config,
//  demonstrating the meta-tuner end to end (genome = solver config, objective =
//  a replay of one continuation step, fitness = -|R|).
// =============================================================================

#include "src/analysis/RegularizedNewtonContinuation.hh"
#include "src/algorithms/optimization/BoundedSearchSpace.hh"
#include "src/algorithms/cultural/CulturalAlgorithm.hh"
#include "src/algorithms/cultural/KnowledgeSources.hh"
#include "src/petsc/PetscRaii.hh"

#include <petsc.h>

#include <cmath>
#include <cstdio>
#include <span>
#include <vector>

using fall_n::LevenbergMarquardt;
using fall_n::RegularizedNewtonConfig;
using fall_n::RegularizedNewtonContinuation;
using fall_n::algorithms::BoundedSearchSpace;
using fall_n::algorithms::cultural::CulturalAlgorithm;
using fall_n::algorithms::cultural::CulturalConfig;
using fall_n::algorithms::cultural::NormativeKnowledge;
using fall_n::algorithms::cultural::SituationalKnowledge;

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

struct CubicBackend {
    [[nodiscard]] petsc::OwnedVec create_vector() const {
        Vec v = nullptr; VecCreateSeq(PETSC_COMM_SELF, 1, &v); VecSet(v, 0.0);
        return petsc::OwnedVec(v);
    }
    [[nodiscard]] petsc::OwnedMat create_matrix() const {
        Mat k = nullptr; MatCreateSeqDense(PETSC_COMM_SELF, 1, 1, nullptr, &k);
        MatZeroEntries(k); return petsc::OwnedMat(k);
    }
    void apply_control(double) const {}
    void residual(Vec u, Vec r) const { set1(r, cubic(get1(u))); }
    void tangent(Vec u, Mat k) const {
        const double uu = get1(u);
        MatSetValue(k, 0, 0, 3.0 * uu * uu - 14.0 * uu + 14.0, INSERT_VALUES);
        MatAssemblyBegin(k, MAT_FINAL_ASSEMBLY); MatAssemblyEnd(k, MAT_FINAL_ASSEMBLY);
    }
    void accept(Vec, double) const {}
};

// Run one continuation step from u0=3.5 with the given (cfg, lm) and return |R|.
double residual_from_config(const RegularizedNewtonConfig& cfg,
                            const LevenbergMarquardt& lm) {
    RegularizedNewtonContinuation<CubicBackend> solver(CubicBackend{}, cfg, lm);
    Vec s = nullptr; VecCreateSeq(PETSC_COMM_SELF, 1, &s); set1(s, 3.5);
    solver.set_solution(s); VecDestroy(&s);
    solver.advance_to(0.0);
    return std::abs(cubic(get1(solver.solution())));
}

// Genome -> (config, LM schedule).
//   g[0] log10(mu0) in [-4,0]   g[1] grow in [1.5,10]   g[2] drop in [0.05,0.9]
//   g[3] mu_max_frac in [0.02,2] (the decisive knob)    g[4] stag_max in [4,40]
void decode(std::span<const double> g, RegularizedNewtonConfig& cfg,
            LevenbergMarquardt& lm) {
    cfg = RegularizedNewtonConfig{};
    cfg.max_newton  = 200;
    cfg.mu_max_frac = g[3];
    cfg.stag_max    = static_cast<int>(std::lround(g[4]));
    lm = LevenbergMarquardt{};
    lm.mu0  = std::pow(10.0, g[0]);
    lm.grow = g[1];
    lm.drop = g[2];
}

double objective(std::span<const double> g) {
    RegularizedNewtonConfig cfg; LevenbergMarquardt lm;
    decode(g, cfg, lm);
    return -residual_from_config(cfg, lm);   // maximise => minimise |R|
}

}  // namespace

int main(int argc, char** argv) {
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    // --- the default config stalls from u0 = 3.5 ----------------------------
    const double r_default =
        residual_from_config(RegularizedNewtonConfig{}, LevenbergMarquardt{});
    std::printf("default config: |R|=%.4g  (stall => large)\n", r_default);
    check(r_default > 1.0e-2, "default LM config stalls from u0=3.5");

    // --- the Cultural Algorithm tunes a config that converges ---------------
    BoundedSearchSpace space(
        std::vector<double>{-4.0, 1.5, 0.05, 0.02,  4.0},
        std::vector<double>{ 0.0, 10.0, 0.90, 2.00, 40.0});
    CulturalConfig cfg;
    cfg.population_size = 16;
    cfg.max_generations = 30;
    cfg.seed = 20260715;
    cfg.target_fitness = -1.0e-6;   // stop early once a converging config is found

    CulturalAlgorithm<BoundedSearchSpace, NormativeKnowledge, SituationalKnowledge>
        ca(space, cfg);
    const auto result = ca.maximize(objective);

    RegularizedNewtonConfig best_cfg; LevenbergMarquardt best_lm;
    decode(std::span<const double>(result.best.genome), best_cfg, best_lm);
    std::printf("CA-tuned config: mu0=%.3g grow=%.2f drop=%.2f mu_max_frac=%.3f "
                "stag=%d  best|R|=%.4g  gens=%zu\n",
                best_lm.mu0, best_lm.grow, best_lm.drop, best_cfg.mu_max_frac,
                best_cfg.stag_max, -result.best.fitness, result.generations);

    check(-result.best.fitness < 1.0e-4,
          "CA finds a config that converges from u0=3.5");
    check(best_cfg.mu_max_frac > RegularizedNewtonConfig{}.mu_max_frac,
          "the tuned config uses more regularisation than the default");

    // --- determinism --------------------------------------------------------
    CulturalAlgorithm<BoundedSearchSpace, NormativeKnowledge, SituationalKnowledge>
        ca2(space, cfg);
    const auto result2 = ca2.maximize(objective);
    check(result2.best.fitness == result.best.fitness,
          "CA-tuner is deterministic under a fixed seed");

    std::printf("%s (%d failure%s)\n", g_failed ? "TEST FAILED" : "TEST PASSED",
                g_failed, g_failed == 1 ? "" : "s");
    PetscFinalize();
    return g_failed;
}
