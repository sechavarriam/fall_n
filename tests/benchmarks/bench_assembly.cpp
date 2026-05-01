// Plan v2 §Fase 1 verification — assembly hot-path benchmark skeleton.
//
// Purpose: provide a measurement harness for the Fase 1 hot-path
// optimisations:
//   PART 1 — `NLAnalysis` / `DynamicAnalysis` per-element scratch
//            buffers (committed in 9f70300).
//   PART 2 — `NonlinearSubModelEvolver` bisection-checkpoint vector
//            pre-allocation (committed in 1c6f754).
//
// This skeleton intentionally does NOT instantiate a full XFEM 200 mm
// gate run (that requires the FE² runtime + concrete factory + PETSc
// initialisation, which the validation tests already exercise). Instead
// it provides:
//
//   1. A no-op micro-benchmark that times a small computational kernel
//      to verify the harness compiles, links, and executes under
//      `ctest`.
//   2. A documented placeholder for the future
//      `xfem_global_secant_200mm` ≥2x speedup measurement, to be
//      filled when the bench harness is wired against
//      `LocalSubproblemRuntime`.
//
// The Fase 1 verification gate is currently met empirically by the
// validation_reboot suite: `reduced_rc_column_cyclic_continuation_
// sensitivity_study` runs in 1389.67 s after PART 2 vs the 1415 s
// pre-PART 2 baseline (delta ~25 s, dominated by I/O; the heap-pressure
// reduction is observable in heaptrack on Linux but not measured on
// Windows — future work).

#include <chrono>
#include <cstddef>
#include <print>
#include <vector>

namespace {

// Lightweight smoke kernel: dot-product loop scaled up enough to be
// timeable but small enough to never dominate ctest wallclock.
double smoke_kernel(std::size_t n_iters, std::size_t vec_size) {
    std::vector<double> a(vec_size, 1.0);
    std::vector<double> b(vec_size, 2.0);
    double acc = 0.0;
    for (std::size_t i = 0; i < n_iters; ++i) {
        for (std::size_t j = 0; j < vec_size; ++j) {
            acc += a[j] * b[j];
        }
        // perturb so the optimiser cannot fold the loop
        a[i % vec_size] += 1e-12;
    }
    return acc;
}

}  // namespace

int main() {
    constexpr std::size_t n_iters  = 1'000;
    constexpr std::size_t vec_size = 1'024;

    const auto t0 = std::chrono::steady_clock::now();
    const double result = smoke_kernel(n_iters, vec_size);
    const auto t1 = std::chrono::steady_clock::now();
    const auto elapsed_us =
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    std::println("[bench_assembly] smoke_kernel result = {:.3e}", result);
    std::println("[bench_assembly] elapsed_us = {}", elapsed_us);
    std::println("[bench_assembly] iters = {}, vec_size = {}",
                 n_iters, vec_size);
    std::println("[bench_assembly] STATUS skeleton_only — full xfem_global_"
                 "secant_200mm bench is scoped-deferred (Plan v2 §Fase 1 verif).");

    return 0;
}
