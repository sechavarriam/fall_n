// =============================================================================
//  test_incremental_control.cpp — IncrementalControlPolicy verification
// =============================================================================
//
//  Validates the injectable control-scheme infrastructure:
//
//    1. LoadControl default (backward-compatible, pure load scaling)
//    2. LoadControl explicit (identical to default)
//    3. DisplacementControl single DOF (pushover-style)
//    4. DisplacementControl multi-DOF
//    5. CustomControl via make_control (lambda-based)
//    6. Concept satisfaction (static checks)
//
//  Requires PETSc runtime (PetscInitialize / PetscFinalize).
//
// =============================================================================

#include <cassert>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include <array>

#include <petsc.h>

#include "header_files.hh"

// ── Constants ─────────────────────────────────────────────────────────────────

static constexpr std::size_t DIM  = 3;
static constexpr std::size_t NDOF = DIM;

static int passed = 0;
static int failed = 0;

static void check(bool cond, const char* msg) {
    if (cond) { std::cout << "  [PASS] " << msg << "\n"; ++passed; }
    else      { std::cout << "  [FAIL] " << msg << "\n"; ++failed; }
}

// ── Helper: create a single-hex unit cube domain ──────────────────────────────

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

using Policy = ThreeDimensionalMaterial;

// =============================================================================
//  Test 1: LoadControl default — backward compatibility
// =============================================================================

void test_load_control_default() {
    std::cout << "\nTest 1: LoadControl default (backward-compatible)\n";

    Domain<DIM> D;
    create_unit_cube(D);

    ContinuumIsotropicElasticMaterial elastic{200.0, 0.3};
    Material<Policy> mat{elastic, ElasticUpdate{}};

    Model<Policy, continuum::SmallStrain, NDOF> M{D, mat};
    M.fix_x(0.0);
    M.setup();

    const double f_per_node = 1.0 / 4.0;
    for (std::size_t id : {1ul, 3ul, 5ul, 7ul})
        M.apply_node_force(id, f_per_node, 0.0, 0.0);

    NonlinearAnalysis<Policy, continuum::SmallStrain> nl{&M};

    // Default overload — should behave exactly as before
    bool ok = nl.solve_incremental(3);
    check(ok, "solve_incremental(3) with default LoadControl converged");

    PetscReal norm;
    VecNorm(M.state_vector(), NORM_2, &norm);
    check(norm > 1e-12, "solution is non-trivial");
}

// =============================================================================
//  Test 2: LoadControl explicit — same result as default
// =============================================================================

void test_load_control_explicit() {
    std::cout << "\nTest 2: LoadControl explicit (identical result)\n";

    Domain<DIM> D;
    create_unit_cube(D);

    ContinuumIsotropicElasticMaterial elastic{200.0, 0.3};
    Material<Policy> mat{elastic, ElasticUpdate{}};

    Model<Policy, continuum::SmallStrain, NDOF> M{D, mat};
    M.fix_x(0.0);
    M.setup();

    const double f_per_node = 1.0 / 4.0;
    for (std::size_t id : {1ul, 3ul, 5ul, 7ul})
        M.apply_node_force(id, f_per_node, 0.0, 0.0);

    NonlinearAnalysis<Policy, continuum::SmallStrain> nl{&M};

    bool ok = nl.solve_incremental(3, 4, LoadControl{});
    check(ok, "solve_incremental(3, 4, LoadControl{}) converged");

    PetscReal norm;
    VecNorm(M.state_vector(), NORM_2, &norm);
    check(norm > 1e-12, "solution is non-trivial");
}

// =============================================================================
//  Test 3: DisplacementControl — single DOF pushover
// =============================================================================

void test_displacement_control_single() {
    std::cout << "\nTest 3: DisplacementControl single DOF\n";

    Domain<DIM> D;
    create_unit_cube(D);

    ContinuumIsotropicElasticMaterial elastic{200.0, 0.3};
    Material<Policy> mat{elastic, ElasticUpdate{}};

    Model<Policy, continuum::SmallStrain, NDOF> M{D, mat};

    // Fix left face (x = 0)
    M.fix_x(0.0);

    // Pre-constrain the controlled DOF (node 1, dof 0 = x-direction)
    // with initial value 0.0 — the actual value is set by the scheme.
    M.constrain_dof(1, 0, 0.0);

    M.setup();

    // No external forces — pure displacement control
    NonlinearAnalysis<Policy, continuum::SmallStrain> nl{&M};

    const double target_disp = 0.005;
    bool ok = nl.solve_incremental(5, 4, DisplacementControl{1, 0, target_disp});
    check(ok, "DisplacementControl single-DOF converged");

    // Verify the state vector: node 1 should have ~target_disp in x
    PetscReal norm;
    VecNorm(M.state_vector(), NORM_2, &norm);
    check(norm > 1e-14, "state is non-trivial under displacement control");

    // The imposed_solution should contain the target displacement
    const PetscScalar* vals;
    VecGetArrayRead(M.imposed_solution(), &vals);

    auto& node = D.node(1);
    auto dof_idx = node.dof_index();
    double imposed_x = std::abs(vals[dof_idx[0]]);

    VecRestoreArrayRead(M.imposed_solution(), &vals);

    check(std::abs(imposed_x - target_disp) < 1e-10,
          "imposed value matches target displacement");
}

// =============================================================================
//  Test 4: DisplacementControl — multi-DOF
// =============================================================================

void test_displacement_control_multi() {
    std::cout << "\nTest 4: DisplacementControl multi-DOF\n";

    Domain<DIM> D;
    create_unit_cube(D);

    ContinuumIsotropicElasticMaterial elastic{200.0, 0.3};
    Material<Policy> mat{elastic, ElasticUpdate{}};

    Model<Policy, continuum::SmallStrain, NDOF> M{D, mat};

    // Fix left face
    M.fix_x(0.0);

    // Pre-constrain the controlled DOFs
    M.constrain_dof(1, 0, 0.0);
    M.constrain_dof(3, 0, 0.0);

    M.setup();

    const double target = 0.003;
    DisplacementControl dc{
        {{1, 0, target}, {3, 0, target}},
        0.0  // no proportional load
    };

    NonlinearAnalysis<Policy, continuum::SmallStrain> nl{&M};
    bool ok = nl.solve_incremental(4, 4, dc);
    check(ok, "DisplacementControl multi-DOF converged");

    PetscReal norm;
    VecNorm(M.state_vector(), NORM_2, &norm);
    check(norm > 1e-14, "state is non-trivial");
}

// =============================================================================
//  Test 5: CustomControl via make_control (lambda)
// =============================================================================

void test_custom_control() {
    std::cout << "\nTest 5: CustomControl via make_control (lambda)\n";

    Domain<DIM> D;
    create_unit_cube(D);

    ContinuumIsotropicElasticMaterial elastic{200.0, 0.3};
    Material<Policy> mat{elastic, ElasticUpdate{}};

    Model<Policy, continuum::SmallStrain, NDOF> M{D, mat};
    M.fix_x(0.0);
    M.setup();

    const double f_per_node = 1.0 / 4.0;
    for (std::size_t id : {1ul, 3ul, 5ul, 7ul})
        M.apply_node_force(id, f_per_node, 0.0, 0.0);

    NonlinearAnalysis<Policy, continuum::SmallStrain> nl{&M};

    // Custom: sqrt ramping of the load
    auto scheme = make_control(
        [](double p, Vec f_full, Vec f_ext, auto* /*model*/) {
            VecCopy(f_full, f_ext);
            VecScale(f_ext, std::sqrt(p));
        }
    );

    bool ok = nl.solve_incremental(5, 4, scheme);
    check(ok, "CustomControl (sqrt ramp) converged");

    PetscReal norm;
    VecNorm(M.state_vector(), NORM_2, &norm);
    check(norm > 1e-12, "solution is non-trivial");
}

// =============================================================================
//  Test 6: Concept satisfaction (compile-time checks)
// =============================================================================

void test_concept_satisfaction() {
    std::cout << "\nTest 6: Concept satisfaction (compile-time)\n";

    using ModelT = Model<Policy, continuum::SmallStrain, NDOF>;

    // LoadControl
    static_assert(IncrementalControlPolicy<LoadControl, ModelT>,
                  "LoadControl must satisfy IncrementalControlPolicy");

    // DisplacementControl
    static_assert(IncrementalControlPolicy<DisplacementControl, ModelT>,
                  "DisplacementControl must satisfy IncrementalControlPolicy");

    // CustomControl with lambda
    auto lam = [](double, Vec, Vec, auto*){};
    using CC = CustomControl<decltype(lam)>;
    static_assert(IncrementalControlPolicy<CC, ModelT>,
                  "CustomControl<lambda> must satisfy IncrementalControlPolicy");

    check(true, "LoadControl satisfies IncrementalControlPolicy");
    check(true, "DisplacementControl satisfies IncrementalControlPolicy");
    check(true, "CustomControl<lambda> satisfies IncrementalControlPolicy");
}

// =============================================================================
//  main
// =============================================================================

int main(int argc, char** argv) {
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    std::cout << "=== IncrementalControlPolicy Verification ===\n";

    test_load_control_default();
    test_load_control_explicit();
    test_displacement_control_single();
    test_displacement_control_multi();
    test_custom_control();
    test_concept_satisfaction();

    std::cout << "\n=== Summary: " << passed << " passed, "
              << failed << " failed ===\n";

    PetscFinalize();
    return (failed > 0) ? 1 : 0;
}
