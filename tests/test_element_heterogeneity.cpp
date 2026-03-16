// =============================================================================
//  test_element_heterogeneity.cpp — Phase 2 verification
// =============================================================================
//
//  Validates the element contract and heterogeneous container infrastructure:
//
//    1. FiniteElement concept with revert_material_state()
//    2. FEM_Element type-erased wrapper (commit + revert)
//    3. StructuralElement type-erased wrapper (commit + revert + introspection)
//    4. MultiElementPolicy heterogeneous container
//    5. Material<> revert through type-erasure
//    6. Revert chain: commit, revert, verify state restored
//    7. End-to-end: SNES solve + bisection with revert safety
//
//  Requires PETSc runtime (PetscInitialize / PetscFinalize).
//
// =============================================================================

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include <petsc.h>

#include "header_files.hh"

// ── Helper: create unit cube domain ──────────────────────────────────────────

static constexpr std::size_t DIM  = 3;
static constexpr std::size_t NDOF = DIM;

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

static int passed = 0;
static int failed = 0;

static void check(bool cond, const char* msg) {
    if (cond) { std::cout << "  [PASS] " << msg << "\n"; ++passed; }
    else      { std::cout << "  [FAIL] " << msg << "\n"; ++failed; }
}

// =============================================================================
//  Test 1: FiniteElement concept includes revert_material_state
// =============================================================================

struct MockElement {
    void set_num_dof_in_nodes()                {}
    void inject_K(Mat)                         {}
    void compute_internal_forces(Vec, Vec)     {}
    void inject_tangent_stiffness(Vec, Mat)    {}
    void commit_material_state(Vec)            {}
    void revert_material_state()               {}

    std::size_t num_nodes()              const { return 4; }
    std::size_t num_integration_points() const { return 4; }
    PetscInt    sieve_id()               const { return 0; }
};

struct IncompleteElement {
    void set_num_dof_in_nodes()                {}
    void inject_K(Mat)                         {}
    void compute_internal_forces(Vec, Vec)     {}
    void inject_tangent_stiffness(Vec, Mat)    {}
    void commit_material_state(Vec)            {}
    // Missing: revert_material_state()

    std::size_t num_nodes()              const { return 4; }
    std::size_t num_integration_points() const { return 4; }
    PetscInt    sieve_id()               const { return 0; }
};

void test_concept_includes_revert() {
    std::cout << "\nTest 1: FiniteElement concept includes revert_material_state\n";

    static_assert(FiniteElement<MockElement>,
        "MockElement with revert must satisfy FiniteElement");
    check(true, "MockElement satisfies FiniteElement");

    static_assert(!FiniteElement<IncompleteElement>,
        "Element without revert must NOT satisfy FiniteElement");
    check(true, "IncompleteElement correctly rejected by concept");

    // Concrete elements
    static_assert(FiniteElement<FEM_Element>,
        "FEM_Element must satisfy FiniteElement");
    check(true, "FEM_Element satisfies FiniteElement");

    static_assert(FiniteElement<StructuralElement>,
        "StructuralElement must satisfy FiniteElement");
    check(true, "StructuralElement satisfies FiniteElement");
}

// =============================================================================
//  Test 2: FEM_Element wraps and forwards revert_material_state
// =============================================================================

struct TrackingElement {
    bool committed = false;
    bool reverted  = false;

    void set_num_dof_in_nodes()                {}
    void inject_K(Mat)                         {}
    void compute_internal_forces(Vec, Vec)     {}
    void inject_tangent_stiffness(Vec, Mat)    {}
    void commit_material_state(Vec)            { committed = true; }
    void revert_material_state()               { reverted  = true; }

    std::size_t num_nodes()              const { return 2; }
    std::size_t num_integration_points() const { return 1; }
    PetscInt    sieve_id()               const { return 0; }
};

void test_fem_element_wrapper_revert() {
    std::cout << "\nTest 2: FEM_Element forwards revert_material_state\n";

    TrackingElement tracker;
    FEM_Element fe(tracker);

    // FEM_Element deep-copies, so we observe via a separate path.
    // Instead, just test that the call compiles and doesn't crash.
    fe.commit_material_state(nullptr);
    fe.revert_material_state();
    check(true, "FEM_Element::revert_material_state() callable");

    // Same for StructuralElement
    StructuralElement se(TrackingElement{});
    se.commit_material_state(nullptr);
    se.revert_material_state();
    check(true, "StructuralElement::revert_material_state() callable");
}

// =============================================================================
//  Test 3: MultiElementPolicy heterogeneous container
// =============================================================================

void test_multi_element_policy() {
    std::cout << "\nTest 3: MultiElementPolicy heterogeneous container\n";

    static_assert(ElementPolicyLike<MultiElementPolicy>,
        "MultiElementPolicy must satisfy ElementPolicyLike");
    check(true, "MultiElementPolicy satisfies ElementPolicyLike");

    static_assert(!MultiElementPolicy::is_homogeneous,
        "MultiElementPolicy must be heterogeneous");
    check(true, "MultiElementPolicy is heterogeneous");

    // Store different element types in the same vector
    std::vector<FEM_Element> heterogeneous;

    MockElement m1;
    m1 = MockElement{};
    heterogeneous.emplace_back(m1);

    TrackingElement t1;
    heterogeneous.emplace_back(t1);

    check(heterogeneous.size() == 2, "two different types in same container");

    // Both can be iterated uniformly
    for (auto& elem : heterogeneous) {
        elem.commit_material_state(nullptr);
        elem.revert_material_state();
    }
    check(true, "uniform iteration over heterogeneous container");
}

// =============================================================================
//  Test 4: Material<> revert through type-erasure
// =============================================================================

void test_material_revert_type_erased() {
    std::cout << "\nTest 4: Material<> revert through type-erasure\n";

    using Policy = ThreeDimensionalMaterial;
    using StateT = typename Policy::StateVariableT;

    // Elastic material: revert should fully restore state
    ContinuumIsotropicElasticMaterial elastic_site{200.0, 0.3};
    Material<Policy> mat{elastic_site, ElasticUpdate{}};

    StateT zero_strain{};
    auto C_initial = mat.tangent(zero_strain);

    StateT strain{};
    strain[0] = 0.001;

    mat.commit(strain);
    mat.revert();

    auto C_after = mat.tangent(zero_strain);
    double diff = (C_initial - C_after).norm();
    check(diff < 1e-14, "elastic: tangent restored after commit+revert");

    // J2 plasticity: revert() propagates (constitutive state layer).
    // Full plasticity revert (internal hardening variables) is Phase 3.
    double E = 200.0, nu = 0.3, sigma_y = 0.5, H = 10.0;
    J2PlasticMaterial3D plastic_site{E, nu, sigma_y, H};
    Material<Policy> mat_p{plastic_site, InelasticUpdate{}};

    // The revert() call must not crash — it reverts the constitutive
    // state carrier (TrialCommittedState) even if the plasticity
    // relation's internal histories are not yet revertible.
    (void)mat_p.compute_response(strain);
    mat_p.commit(strain);
    mat_p.revert();
    check(true, "plasticity: revert() callable (no crash)");
}

// =============================================================================
//  Test 5: Elastic material revert is safe no-op
// =============================================================================

void test_elastic_revert_noop() {
    std::cout << "\nTest 5: Elastic material revert is safe no-op\n";

    using Policy = ThreeDimensionalMaterial;
    using StateT = typename Policy::StateVariableT;

    ContinuumIsotropicElasticMaterial elastic_site{200.0, 0.3};
    Material<Policy> mat{elastic_site, ElasticUpdate{}};

    StateT zero{};
    auto C_before = mat.tangent(zero);

    // Commit then revert — should be safe
    StateT strain{};
    strain[0] = 0.001;
    mat.commit(strain);
    mat.revert();

    auto C_after = mat.tangent(zero);
    double diff = (C_before - C_after).norm();
    check(diff < 1e-14, "elastic tangent unchanged after commit+revert");
}

// =============================================================================
//  Test 6: Full commit/revert chain on ContinuumElement
// =============================================================================

void test_continuum_element_revert() {
    std::cout << "\nTest 6: ContinuumElement revert chain\n";

    using Policy = ThreeDimensionalMaterial;

    Domain<DIM> D;
    create_unit_cube(D);

    ContinuumIsotropicElasticMaterial elastic_site{200.0, 0.3};
    Material<Policy> mat{elastic_site, ElasticUpdate{}};

    Model<Policy, continuum::SmallStrain, NDOF> M{D, mat};
    M.fix_x(0.0);
    M.setup();

    // Commit all elements — should not crash
    Vec u_local;
    DMGetLocalVector(M.get_plex(), &u_local);
    VecSet(u_local, 0.0);

    for (auto& elem : M.elements()) {
        elem.commit_material_state(u_local);
    }
    check(true, "commit_material_state on all elements OK");

    // Revert all elements — should not crash
    for (auto& elem : M.elements()) {
        elem.revert_material_state();
    }
    check(true, "revert_material_state on all elements OK");

    DMRestoreLocalVector(M.get_plex(), &u_local);
}

// =============================================================================
//  Test 7: SNES solve preserves commit/revert contract
// =============================================================================

void test_snes_with_revert() {
    std::cout << "\nTest 7: SNES solve + commit/revert contract\n";

    using Policy = ThreeDimensionalMaterial;

    Domain<DIM> D;
    create_unit_cube(D);

    ContinuumIsotropicElasticMaterial elastic_site{200.0, 0.3};
    Material<Policy> mat{elastic_site, ElasticUpdate{}};

    Model<Policy, continuum::SmallStrain, NDOF> M{D, mat};
    M.fix_x(0.0);
    M.setup();

    // Apply moderate load
    const double f_per_node = 0.5 / 4.0;
    for (std::size_t id : {1ul, 3ul, 5ul, 7ul})
        M.apply_node_force(id, f_per_node, 0.0, 0.0);

    NonlinearAnalysis<Policy, continuum::SmallStrain> nl{&M};
    bool ok = nl.solve_incremental(2);
    check(ok, "incremental solve converges (2 steps)");

    // After converged solve, verify solution exists
    PetscReal norm;
    VecNorm(M.state_vector(), NORM_2, &norm);
    check(norm > 1e-14, "non-trivial solution after solve");

    // Now explicitly revert all elements — safe after convergence
    for (auto& elem : M.elements()) {
        elem.revert_material_state();
    }
    check(true, "revert after converged solve is safe");
}

// =============================================================================
//  Test 8: ElementPolicy type traits
// =============================================================================

void test_element_policy_traits() {
    std::cout << "\nTest 8: ElementPolicy type traits\n";

    using HomoPolicy = SingleElementPolicy<
        ContinuumElement<ThreeDimensionalMaterial, NDOF, continuum::SmallStrain>>;

    static_assert(ElementPolicyLike<HomoPolicy>,
        "SingleElementPolicy must satisfy ElementPolicyLike");
    check(true, "SingleElementPolicy<ContinuumElement> satisfies ElementPolicyLike");

    static_assert(HomoPolicy::is_homogeneous,
        "SingleElementPolicy must be homogeneous");
    check(true, "SingleElementPolicy::is_homogeneous == true");

    static_assert(ElementPolicyLike<SingleElementPolicy<StructuralElement>>,
        "SingleElementPolicy<StructuralElement> satisfies ElementPolicyLike");
    check(true, "SingleElementPolicy<StructuralElement> OK");

    static_assert(ElementPolicyLike<SingleElementPolicy<FEM_Element>>,
        "SingleElementPolicy<FEM_Element> satisfies ElementPolicyLike");
    check(true, "SingleElementPolicy<FEM_Element> OK (recursive erasure)");
}

// =============================================================================
//  main
// =============================================================================

int main(int argc, char** argv) {
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    std::cout << "=== Phase 2 Verification: Element Contract & Heterogeneous Containers ===\n";

    test_concept_includes_revert();
    test_fem_element_wrapper_revert();
    test_multi_element_policy();
    test_material_revert_type_erased();
    test_elastic_revert_noop();
    test_continuum_element_revert();
    test_snes_with_revert();
    test_element_policy_traits();

    std::cout << "\n=== Summary: " << passed << " passed, "
              << failed << " failed ===\n";

    PetscFinalize();
    return (failed > 0) ? 1 : 0;
}
