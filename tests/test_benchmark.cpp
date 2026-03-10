// =============================================================================
//  test_benchmark.cpp — Phase 5: Performance & scalability infrastructure
// =============================================================================
//
//  Validates the timing and solver configuration utilities introduced
//  in Phase 5.
//
//  Tests:
//
//    ── Timing infrastructure ──────────────────────────────────────────
//    1. ScopedTimer accumulates wall-clock time correctly
//    2. StopWatch manual start/stop + reset
//    3. AnalysisTimer phase tracking + print_summary
//    4. ElementTimer statistics (total, avg, min, max, stddev)
//
//    ── PETSc log stage helpers ────────────────────────────────────────
//    5. perf::register_analysis_stages + ScopedStage push/pop
//
//    ── Matrix diagnostics (SolverConfig) ─────────────────────────────
//    6. compute_bandwidth on a small tridiagonal matrix
//    7. print_matrix_info diagnostic output
//    8. apply_rcm reduces bandwidth of a banded matrix
//
//    ── Solver presets ────────────────────────────────────────────────
//    9. Solver preset functions execute without error
//
//    ── Analysis timer integration ─────────────────────────────────────
//   10. LinearAnalysis timer records setup/assembly/solve/commit phases
//   11. NonlinearAnalysis timer records setup/solve/commit phases
//   12. DynamicAnalysis timer records setup/solve/post phases
//
//  Mesh: single hex8 (unit cube) built programmatically (8 nodes).
//
//  Requires PETSc runtime (PetscInitialize / PetscFinalize).
//
// =============================================================================

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <vector>
#include <array>
#include <thread>
#include <chrono>

#include <petsc.h>

// ── Project headers ───────────────────────────────────────────────────────────

#include "header_files.hh"
#include "src/utils/SolverConfig.hh"

// ── Constants ─────────────────────────────────────────────────────────────────

static constexpr std::size_t DIM  = 3;

static constexpr double E_modulus  = 1000.0;
static constexpr double nu_poisson = 0.0;
static constexpr double rho_density = 1.0;

static int passed = 0;
static int failed = 0;

// ── Helpers ───────────────────────────────────────────────────────────────────

static void check(bool cond, const char* msg) {
    if (cond) {
        std::cout << "  [PASS] " << msg << "\n";
        ++passed;
    } else {
        std::cout << "  [FAIL] " << msg << "\n";
        ++failed;
    }
}

static void create_unit_cube(Domain<DIM>& D) {
    D.preallocate_node_capacity(8);
    D.add_node(0, 0.0, 0.0, 0.0);
    D.add_node(1, 1.0, 0.0, 0.0);
    D.add_node(2, 0.0, 1.0, 0.0);
    D.add_node(3, 1.0, 1.0, 0.0);
    D.add_node(4, 0.0, 0.0, 1.0);
    D.add_node(5, 1.0, 0.0, 1.0);
    D.add_node(6, 0.0, 1.0, 1.0);
    D.add_node(7, 1.0, 1.0, 1.0);

    std::array<PetscInt, 8> conn{0, 1, 2, 3, 4, 5, 6, 7};
    D.make_element<LagrangeElement<3, 2, 2, 2>>(
        GaussLegendreCellIntegrator<2, 2, 2>{}, 0, conn.data());
    D.assemble_sieve();
}


// =============================================================================
//  Test 1: ScopedTimer accumulates wall-clock time
// =============================================================================

static void test_1_scoped_timer() {
    std::cout << "\n── Test 1: ScopedTimer ──\n";

    TimingRecord rec;

    {
        ScopedTimer t(rec);
        // Burn ~10 ms
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    check(rec.call_count == 1, "ScopedTimer: call_count == 1");
    check(rec.elapsed_s > 0.005, "ScopedTimer: elapsed > 5 ms");
    check(rec.elapsed_s < 1.0,   "ScopedTimer: elapsed < 1 s (sanity)");

    // Accumulate a second measurement
    {
        ScopedTimer t(rec);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    check(rec.call_count == 2, "ScopedTimer: accumulated call_count == 2");
    check(rec.elapsed_s > 0.010, "ScopedTimer: accumulated elapsed > 10 ms");
    check(rec.average_s() > 0.005, "ScopedTimer: average > 5 ms");
}


// =============================================================================
//  Test 2: StopWatch manual start/stop
// =============================================================================

static void test_2_stopwatch() {
    std::cout << "\n── Test 2: StopWatch ──\n";

    StopWatch sw;
    check(!sw.running(), "StopWatch: initially not running");
    check(sw.calls() == 0, "StopWatch: initial calls == 0");

    sw.start();
    check(sw.running(), "StopWatch: running after start()");
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    double dt = sw.stop();

    check(!sw.running(), "StopWatch: not running after stop()");
    check(dt > 0.005, "StopWatch: stop() returns positive time");
    check(sw.calls() == 1, "StopWatch: calls == 1 after one measurement");
    check(sw.elapsed() > 0.005, "StopWatch: elapsed > 5 ms");

    // Reset
    sw.reset();
    check(sw.elapsed() == 0.0, "StopWatch: elapsed == 0 after reset");
    check(sw.calls() == 0, "StopWatch: calls == 0 after reset");
}


// =============================================================================
//  Test 3: AnalysisTimer phase tracking
// =============================================================================

static void test_3_analysis_timer() {
    std::cout << "\n── Test 3: AnalysisTimer ──\n";

    AnalysisTimer timer;

    timer.start("setup");
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    timer.stop("setup");

    timer.start("assembly");
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    timer.stop("assembly");

    timer.start("solve");
    std::this_thread::sleep_for(std::chrono::milliseconds(15));
    timer.stop("solve");

    check(timer.elapsed("setup") > 0.001,    "AnalysisTimer: setup > 1 ms");
    check(timer.elapsed("assembly") > 0.005,  "AnalysisTimer: assembly > 5 ms");
    check(timer.elapsed("solve") > 0.010,     "AnalysisTimer: solve > 10 ms");
    check(timer.calls("setup") == 1,          "AnalysisTimer: setup calls == 1");
    check(timer.calls("assembly") == 1,       "AnalysisTimer: assembly calls == 1");
    check(timer.calls("solve") == 1,          "AnalysisTimer: solve calls == 1");
    check(timer.elapsed("nonexistent") == 0.0, "AnalysisTimer: unknown phase == 0");

    double total = timer.total_elapsed();
    check(total > 0.020, "AnalysisTimer: total > 20 ms");

    // Print for visual inspection
    timer.print_summary();

    // Reset
    timer.reset();
    check(timer.elapsed("setup") == 0.0, "AnalysisTimer: 0 after reset");
}


// =============================================================================
//  Test 4: ElementTimer statistics
// =============================================================================

static void test_4_element_timer() {
    std::cout << "\n── Test 4: ElementTimer ──\n";

    constexpr std::size_t N = 10;
    ElementTimer et(N);

    check(et.num_elements() == N, "ElementTimer: num_elements == 10");

    for (std::size_t e = 0; e < N; ++e) {
        et.start(e);
        // Variable workload — element 9 is a bit slower
        volatile double sum = 0.0;
        std::size_t iters = 100'000 * (e + 1);
        for (std::size_t i = 0; i < iters; ++i) sum += 0.001;
        (void)sum;
        et.stop(e);
    }

    check(et.total() > 0.0,     "ElementTimer: total > 0");
    check(et.average() > 0.0,   "ElementTimer: average > 0");
    check(et.min_time() >= 0.0, "ElementTimer: min >= 0");
    check(et.max_time() >= et.min_time(), "ElementTimer: max >= min");
    check(et.slowest_element() < N, "ElementTimer: slowest index valid");
    check(et.element_calls(0) == 1, "ElementTimer: call count == 1");

    // Print for visual inspection
    et.print_statistics();

    // Reset
    et.reset();
    check(et.total() == 0.0, "ElementTimer: 0 after reset");
}


// =============================================================================
//  Test 5: PETSc log stage helpers
// =============================================================================

static void test_5_petsc_log_stages() {
    std::cout << "\n── Test 5: PETSc log stages ──\n";

    const auto& stages = perf::register_analysis_stages();

    // Re-calling should return the same handles (idempotent via static flag).
    const auto& stages2 = perf::register_analysis_stages();
    check(stages.setup == stages2.setup,       "PetscLogStage: idempotent setup");
    check(stages.assembly == stages2.assembly, "PetscLogStage: idempotent assembly");
    check(stages.solve == stages2.solve,       "PetscLogStage: idempotent solve");
    check(stages.post == stages2.post,         "PetscLogStage: idempotent post");

    // ScopedStage push/pop should work without crash
    {
        perf::ScopedStage s(stages.assembly);
        // Some trivial PETSc operation inside the stage
        Vec v;
        VecCreateSeq(PETSC_COMM_SELF, 10, &v);
        VecSet(v, 1.0);
        VecDestroy(&v);
    }
    check(true, "ScopedStage: push/pop without crash");
}


// =============================================================================
//  Test 6: compute_bandwidth on a small matrix
// =============================================================================

static void test_6_compute_bandwidth() {
    std::cout << "\n── Test 6: compute_bandwidth ──\n";

    // Create a 10×10 tridiagonal matrix via PETSc
    Mat A;
    MatCreateSeqAIJ(PETSC_COMM_SELF, 10, 10, 3, nullptr, &A);

    for (PetscInt i = 0; i < 10; ++i) {
        PetscScalar v = 2.0;
        MatSetValue(A, i, i, v, INSERT_VALUES);
        if (i > 0)  MatSetValue(A, i, i - 1, -1.0, INSERT_VALUES);
        if (i < 9)  MatSetValue(A, i, i + 1, -1.0, INSERT_VALUES);
    }
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

    PetscInt bw = solver_config::compute_bandwidth(A);
    check(bw == 1, "compute_bandwidth: tridiag bandwidth == 1");

    // Create a new matrix with corner entries to increase bandwidth
    Mat B;
    MatCreateSeqAIJ(PETSC_COMM_SELF, 10, 10, 4, nullptr, &B);
    for (PetscInt i = 0; i < 10; ++i) {
        MatSetValue(B, i, i, 2.0, INSERT_VALUES);
        if (i > 0)  MatSetValue(B, i, i - 1, -1.0, INSERT_VALUES);
        if (i < 9)  MatSetValue(B, i, i + 1, -1.0, INSERT_VALUES);
    }
    MatSetValue(B, 0, 9, 0.5, INSERT_VALUES);
    MatSetValue(B, 9, 0, 0.5, INSERT_VALUES);
    MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY);

    bw = solver_config::compute_bandwidth(B);
    check(bw == 9, "compute_bandwidth: with corner entry, bandwidth == 9");

    MatDestroy(&B);

    MatDestroy(&A);
}


// =============================================================================
//  Test 7: print_matrix_info diagnostic output
// =============================================================================

static void test_7_matrix_info() {
    std::cout << "\n── Test 7: print_matrix_info ──\n";

    Mat A;
    MatCreateSeqAIJ(PETSC_COMM_SELF, 5, 5, 3, nullptr, &A);
    for (PetscInt i = 0; i < 5; ++i) {
        MatSetValue(A, i, i, 2.0, INSERT_VALUES);
        if (i > 0) MatSetValue(A, i, i - 1, -1.0, INSERT_VALUES);
        if (i < 4) MatSetValue(A, i, i + 1, -1.0, INSERT_VALUES);
    }
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

    // Should print without crashing
    solver_config::print_matrix_info(A, "test_tridiag");
    check(true, "print_matrix_info: completes without crash");

    MatDestroy(&A);
}


// =============================================================================
//  Test 8: apply_rcm bandwidth reduction
// =============================================================================

static void test_8_apply_rcm() {
    std::cout << "\n── Test 8: apply_rcm ──\n";

    // Create a 20×20 banded matrix with some off-diagonal entries
    // that a permutation can help
    const PetscInt n = 20;
    Mat A;
    MatCreateSeqAIJ(PETSC_COMM_SELF, n, n, 5, nullptr, &A);

    // Build a mesh-connectivity-like sparsity:
    // Chain connectivity: 0-1-2-...-19 but with numbering shuffled
    // Apply a scrambled numbering to create artificial bandwidth
    std::vector<PetscInt> perm = {0, 10, 5, 15, 2, 12, 7, 17,
                                   1, 11, 6, 16, 3, 13, 8, 18,
                                   4, 14, 9, 19};

    for (PetscInt i = 0; i < n - 1; ++i) {
        PetscInt r = perm[static_cast<std::size_t>(i)];
        PetscInt c = perm[static_cast<std::size_t>(i + 1)];
        MatSetValue(A, r, c,  1.0, INSERT_VALUES);
        MatSetValue(A, c, r,  1.0, INSERT_VALUES);
        MatSetValue(A, r, r, 10.0, ADD_VALUES);
        MatSetValue(A, c, c, 10.0, ADD_VALUES);
    }

    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

    PetscInt bw_before = solver_config::compute_bandwidth(A);

    double ratio = solver_config::apply_rcm(&A);

    PetscInt bw_after = solver_config::compute_bandwidth(A);

    check(bw_after <= bw_before, "apply_rcm: bandwidth did not increase");
    check(ratio >= 1.0,          "apply_rcm: reduction ratio >= 1.0");

    std::cout << "    Before: " << bw_before
              << "  After: " << bw_after
              << "  Ratio: " << ratio << "\n";

    MatDestroy(&A);
}


// =============================================================================
//  Test 9: Solver presets execute without error
// =============================================================================

static void test_9_solver_presets() {
    std::cout << "\n── Test 9: Solver presets ──\n";

    // These just set PETSc options; they should not crash
    solver_config::direct_lu();
    check(true, "direct_lu(): no crash");

    solver_config::iterative_cg_icc(2);
    check(true, "iterative_cg_icc(2): no crash");

    solver_config::iterative_gmres_ilu(1);
    check(true, "iterative_gmres_ilu(1): no crash");

    solver_config::amg_cg();
    check(true, "amg_cg(): no crash");

    solver_config::enable_rcm();
    check(true, "enable_rcm(): no crash");

    solver_config::enable_nd();
    check(true, "enable_nd(): no crash");

    // Reset to direct LU for subsequent tests
    solver_config::direct_lu();
}


// =============================================================================
//  Test 10: LinearAnalysis timer integration
// =============================================================================

static void test_10_linear_timer() {
    std::cout << "\n── Test 10: LinearAnalysis timer ──\n";

    // Reset options for this test
    PetscOptionsSetValue(nullptr, "-ksp_type", "preonly");
    PetscOptionsSetValue(nullptr, "-pc_type",  "lu");

    Domain<DIM> D;
    create_unit_cube(D);

    ContinuumIsotropicElasticMaterial mat_inst{E_modulus, nu_poisson};
    Material<ThreeDimensionalMaterial> mat{mat_inst, ElasticUpdate{}};

    Model<ThreeDimensionalMaterial> M{D, mat};

    // Fix x=0 face
    M.fix_x(0.0);
    M.setup();

    // Apply a traction
    D.create_boundary_from_plane("LoadX1", 0, 1.0);
    M.apply_surface_traction("LoadX1", 10.0, 0.0, 0.0);

    LinearAnalysis<ThreeDimensionalMaterial> lin{&M};
    lin.solve();

    const auto& timer = lin.timer();

    check(timer.elapsed("setup") > 0.0,    "LinearAnalysis timer: setup > 0");
    check(timer.elapsed("assembly") > 0.0,  "LinearAnalysis timer: assembly > 0");
    check(timer.elapsed("solve") > 0.0,     "LinearAnalysis timer: solve > 0");
    check(timer.elapsed("commit") > 0.0,    "LinearAnalysis timer: commit > 0");
    check(timer.calls("setup") == 1,        "LinearAnalysis timer: setup calls == 1");
    check(timer.calls("solve") == 1,        "LinearAnalysis timer: solve calls == 1");

    std::cout << "    Linear timing breakdown:\n";
    std::cout << "      setup:    " << timer.elapsed("setup")    << " s\n";
    std::cout << "      assembly: " << timer.elapsed("assembly") << " s\n";
    std::cout << "      solve:    " << timer.elapsed("solve")    << " s\n";
    std::cout << "      commit:   " << timer.elapsed("commit")   << " s\n";
}


// =============================================================================
//  Test 11: NonlinearAnalysis timer integration
// =============================================================================

static void test_11_nonlinear_timer() {
    std::cout << "\n── Test 11: NonlinearAnalysis timer ──\n";

    PetscOptionsSetValue(nullptr, "-ksp_type", "preonly");
    PetscOptionsSetValue(nullptr, "-pc_type",  "lu");
    PetscOptionsSetValue(nullptr, "-snes_rtol", "1e-8");
    PetscOptionsSetValue(nullptr, "-snes_atol", "1e-10");
    PetscOptionsSetValue(nullptr, "-snes_max_it", "50");
    PetscOptionsSetValue(nullptr, "-snes_linesearch_type", "basic");

    Domain<DIM> D;
    create_unit_cube(D);

    ContinuumIsotropicElasticMaterial mat_inst{E_modulus, nu_poisson};
    Material<ThreeDimensionalMaterial> mat{mat_inst, ElasticUpdate{}};

    Model<ThreeDimensionalMaterial> M{D, mat};
    M.fix_x(0.0);
    M.setup();

    D.create_boundary_from_plane("LoadX1", 0, 1.0);
    M.apply_surface_traction("LoadX1", 10.0, 0.0, 0.0);

    NonlinearAnalysis<ThreeDimensionalMaterial> nl{&M};
    bool ok = nl.solve();

    check(ok, "NonlinearAnalysis: solve converged");

    const auto& timer = nl.timer();

    check(timer.elapsed("setup") > 0.0,  "NL timer: setup > 0");
    check(timer.elapsed("solve") > 0.0,  "NL timer: solve > 0");
    check(timer.elapsed("commit") > 0.0, "NL timer: commit > 0");
    check(timer.calls("setup") == 1,     "NL timer: setup calls == 1");
    check(timer.calls("solve") == 1,     "NL timer: solve calls == 1");

    std::cout << "    NL timing breakdown:\n";
    std::cout << "      setup:  " << timer.elapsed("setup")  << " s\n";
    std::cout << "      solve:  " << timer.elapsed("solve")  << " s\n";
    std::cout << "      commit: " << timer.elapsed("commit") << " s\n";
}


// =============================================================================
//  Test 12: DynamicAnalysis timer integration
// =============================================================================

static void test_12_dynamic_timer() {
    std::cout << "\n── Test 12: DynamicAnalysis timer ──\n";

    Domain<DIM> D;
    create_unit_cube(D);

    ContinuumIsotropicElasticMaterial mat_inst{E_modulus, nu_poisson};
    Material<ThreeDimensionalMaterial> mat{mat_inst, ElasticUpdate{}};

    Model<ThreeDimensionalMaterial> M{D, mat};
    M.fix_x(0.0);
    M.setup();
    M.set_density(rho_density);

    DynamicAnalysis<ThreeDimensionalMaterial> dyn(&M);

    // Small initial displacement to trigger dynamics
    BoundaryConditionSet<3> bcs;
    for (std::size_t id : {1ul, 3ul, 5ul, 7ul}) {
        bcs.add_initial_condition({id, {0.001, 0.0, 0.0}, {0.0, 0.0, 0.0}});
    }
    dyn.set_initial_conditions(bcs);

    // Very short simulation (a few steps)
    double omega_est = (M_PI / 2.0) * std::sqrt(E_modulus / rho_density);
    double T_est = 2.0 * M_PI / omega_est;
    double dt = T_est / 20.0;
    double t_final = 0.25 * T_est;  // Just a few steps

    bool ok = dyn.solve(t_final, dt);
    check(ok, "DynamicAnalysis: solve completed");

    const auto& timer = dyn.timer();

    check(timer.elapsed("setup") > 0.0, "Dynamic timer: setup > 0");
    check(timer.elapsed("solve") > 0.0, "Dynamic timer: solve > 0");
    check(timer.elapsed("post") > 0.0,  "Dynamic timer: post > 0");
    check(timer.calls("setup") == 1,    "Dynamic timer: setup calls == 1");
    check(timer.calls("solve") == 1,    "Dynamic timer: solve calls == 1");

    std::cout << "    Dynamic timing breakdown:\n";
    std::cout << "      setup: " << timer.elapsed("setup") << " s\n";
    std::cout << "      solve: " << timer.elapsed("solve") << " s\n";
    std::cout << "      post:  " << timer.elapsed("post")  << " s\n";
}


// =============================================================================
//  Main
// =============================================================================

int main(int argc, char* argv[]) {
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    // Default solver options
    PetscOptionsSetValue(nullptr, "-ksp_type", "preonly");
    PetscOptionsSetValue(nullptr, "-pc_type",  "lu");

    std::cout << "\n"
        << "  ╔══════════════════════════════════════════════════════════╗\n"
        << "  ║   Phase 5: Performance & Scalability Infrastructure    ║\n"
        << "  ╚══════════════════════════════════════════════════════════╝\n";

    test_1_scoped_timer();
    test_2_stopwatch();
    test_3_analysis_timer();
    test_4_element_timer();
    test_5_petsc_log_stages();
    test_6_compute_bandwidth();
    test_7_matrix_info();
    test_8_apply_rcm();
    test_9_solver_presets();
    test_10_linear_timer();
    test_11_nonlinear_timer();
    test_12_dynamic_timer();

    std::cout << "\n"
        << "  ══════════════════════════════════════════════════════════\n"
        << "  Results: " << passed << " passed, " << failed << " failed "
        << "out of " << (passed + failed) << " total\n"
        << "  ══════════════════════════════════════════════════════════\n\n";

    PetscFinalize();
    return (failed > 0) ? 1 : 0;
}
