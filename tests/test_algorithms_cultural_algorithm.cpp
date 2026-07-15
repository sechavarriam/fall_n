// =============================================================================
//  test_algorithms_cultural_algorithm.cpp
//
//  Std-only (LIBS NONE) unit test for the Cultural Algorithm module. Verifies
//  concept conformance, convergence on standard benchmarks, box respect and
//  bit-for-bit determinism under a fixed seed.
// =============================================================================

#include "src/algorithms/optimization/BoundedSearchSpace.hh"
#include "src/algorithms/cultural/CulturalAlgorithm.hh"
#include "src/algorithms/cultural/KnowledgeSources.hh"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <random>
#include <span>
#include <vector>

using fall_n::algorithms::BoundedSearchSpace;
using fall_n::algorithms::cultural::CulturalAlgorithm;
using fall_n::algorithms::cultural::CulturalConfig;
using fall_n::algorithms::cultural::CulturalResult;
using fall_n::algorithms::cultural::NormativeKnowledge;
using fall_n::algorithms::cultural::SituationalKnowledge;

// --- concept conformance (checked with concrete types) -----------------------
namespace {
using fall_n::algorithms::cultural::KnowledgeSource;
static_assert(KnowledgeSource<NormativeKnowledge, BoundedSearchSpace, std::mt19937_64>,
              "NormativeKnowledge must model KnowledgeSource");
static_assert(KnowledgeSource<SituationalKnowledge, BoundedSearchSpace, std::mt19937_64>,
              "SituationalKnowledge must model KnowledgeSource");

int g_failed = 0;
void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? " ok " : "FAIL", what);
    if (!ok) ++g_failed;
}

BoundedSearchSpace box(std::size_t dim, double lo, double hi) {
    return BoundedSearchSpace(std::vector<double>(dim, lo),
                              std::vector<double>(dim, hi));
}

using CA = CulturalAlgorithm<BoundedSearchSpace, NormativeKnowledge, SituationalKnowledge>;

// Objectives: maximised, so each returns the negated cost.
double sphere(std::span<const double> x) {
    double s = 0.0;
    for (double v : x) s += v * v;
    return -s;
}
double rosenbrock2(std::span<const double> x) {
    const double a = 1.0 - x[0];
    const double b = x[1] - x[0] * x[0];
    return -(a * a + 100.0 * b * b);
}
double rastrigin(std::span<const double> x) {
    constexpr double kPi = 3.14159265358979323846;
    constexpr double A = 10.0;
    double s = A * static_cast<double>(x.size());
    for (double v : x) s += v * v - A * std::cos(2.0 * kPi * v);
    return -s;
}

bool within(const std::vector<double>& g, double lo, double hi) {
    for (double v : g) if (v < lo || v > hi) return false;
    return true;
}
}  // namespace

int main() {
    // --- Sphere 5D: should drive the sum of squares close to zero -----------
    {
        CulturalConfig cfg;
        cfg.population_size = 40;
        cfg.max_generations = 300;
        cfg.seed = 12345;
        CA ca(box(5, -5.0, 5.0), cfg);
        const CulturalResult r = ca.maximize(sphere);
        std::printf("sphere5D     best=%.6g  gens=%zu\n", r.best.fitness, r.generations);
        check(r.best.fitness > -1e-2, "sphere5D converges near optimum");
        check(within(r.best.genome, -5.0, 5.0), "sphere5D best inside box");
        check(!r.best_fitness_history.empty(), "history recorded");
        // elitism => monotonically non-decreasing best history
        bool monotone = true;
        for (std::size_t i = 1; i < r.best_fitness_history.size(); ++i)
            monotone = monotone &&
                       (r.best_fitness_history[i] >= r.best_fitness_history[i - 1]);
        check(monotone, "best-fitness history non-decreasing (elitism)");
    }

    // --- Rosenbrock 2D: narrow curved valley --------------------------------
    {
        CulturalConfig cfg;
        cfg.population_size = 40;
        cfg.max_generations = 500;
        cfg.seed = 777;
        CA ca(box(2, -5.0, 5.0), cfg);
        const CulturalResult r = ca.maximize(rosenbrock2);
        std::printf("rosenbrock2D best=%.6g\n", r.best.fitness);
        check(r.best.fitness > -0.1, "rosenbrock2D descends the valley");
        check(within(r.best.genome, -5.0, 5.0), "rosenbrock2D best inside box");
    }

    // --- Rastrigin 5D: highly multimodal ------------------------------------
    {
        CulturalConfig cfg;
        cfg.population_size = 60;
        cfg.max_generations = 500;
        cfg.seed = 2024;
        CA ca(box(5, -5.12, 5.12), cfg);
        const CulturalResult r = ca.maximize(rastrigin);
        std::printf("rastrigin5D  best=%.6g\n", r.best.fitness);
        // random start averages ~ -50; a working CA gets well below that.
        check(r.best.fitness > -8.0, "rastrigin5D improves markedly over random");
        check(within(r.best.genome, -5.12, 5.12), "rastrigin5D best inside box");
    }

    // --- Determinism: identical results for identical seed ------------------
    {
        CulturalConfig cfg;
        cfg.population_size = 30;
        cfg.max_generations = 120;
        cfg.seed = 999;
        auto run = [&] { return CA(box(4, -3.0, 3.0), cfg).maximize(sphere); };
        const CulturalResult a = run();
        const CulturalResult b = run();
        bool same = a.best.genome.size() == b.best.genome.size() &&
                    a.best.fitness == b.best.fitness;
        for (std::size_t i = 0; same && i < a.best.genome.size(); ++i)
            same = same && (a.best.genome[i] == b.best.genome[i]);
        check(same, "bit-identical result across runs with the same seed");
    }

    std::printf("%s (%d failure%s)\n", g_failed ? "TEST FAILED" : "TEST PASSED",
                g_failed, g_failed == 1 ? "" : "s");
    return g_failed;
}
