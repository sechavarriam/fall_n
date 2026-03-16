// =============================================================================
//  test_finite_element_concept.cpp
// =============================================================================
//
//  Tests for:
//    1. FiniteElement concept — compile-time checks on mocks + ContinuumElement
//    2. FEM_Element  — type-erased wrapper (construction, forwarding, semantics)
//    3. ElementPolicy — SingleElementPolicy / MultiElementPolicy
//
// =============================================================================

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <petsc.h>

#include "src/elements/FiniteElementConcept.hh"
#include "src/elements/FEM_Element.hh"
#include "src/elements/ElementPolicy.hh"
#include "src/elements/ContinuumElement.hh"
#include "src/elements/StructuralElement.hh"
#include "src/materials/MaterialPolicy.hh"

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Mock elements (minimal FiniteElement implementations for testing)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

struct MockContinuum {
    int nodes_  = 4;
    int gps_    = 4;
    PetscInt id_ = 10;

    void set_num_dof_in_nodes()                {}
    void inject_K(Mat)                         {}
    void compute_internal_forces(Vec, Vec)     {}
    void inject_tangent_stiffness(Vec, Mat)    {}
    void commit_material_state(Vec)            {}
    void revert_material_state()               {}

    std::size_t num_nodes()              const { return static_cast<std::size_t>(nodes_); }
    std::size_t num_integration_points() const { return static_cast<std::size_t>(gps_); }
    PetscInt    sieve_id()               const { return id_; }
};

struct MockBeam {
    void set_num_dof_in_nodes()                {}
    void inject_K(Mat)                         {}
    void compute_internal_forces(Vec, Vec)     {}
    void inject_tangent_stiffness(Vec, Mat)    {}
    void commit_material_state(Vec)            {}
    void revert_material_state()               {}

    std::size_t num_nodes()              const { return 2; }
    std::size_t num_integration_points() const { return 3; }
    PetscInt    sieve_id()               const { return 99; }
};

// Negative test: a type that does NOT satisfy FiniteElement
struct NotAnElement {
    int x = 0;
};


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Compile-time checks (static_assert)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// 1. Mocks satisfy FiniteElement
static_assert( FiniteElement<MockContinuum>);
static_assert( FiniteElement<MockBeam>);
static_assert(!FiniteElement<NotAnElement>);
static_assert(!FiniteElement<int>);

// 2. FEM_Element (type-erased wrapper) satisfies FiniteElement
//    (also asserted inside FEM_Element.hh itself)
static_assert( FiniteElement<FEM_Element>);

// 3. ContinuumElement satisfies FiniteElement for all solid policies
static_assert( FiniteElement<ContinuumElement<ThreeDimensionalMaterial, 3>>);
static_assert( FiniteElement<ContinuumElement<PlaneMaterial, 2>>);

// 4. Element policies satisfy ElementPolicyLike
static_assert( ElementPolicyLike<SingleElementPolicy<MockContinuum>>);
static_assert( ElementPolicyLike<SingleElementPolicy<MockBeam>>);
static_assert( ElementPolicyLike<MultiElementPolicy>);
static_assert( ElementPolicyLike<SingleElementPolicy<StructuralElement>>);

// 5. is_homogeneous flag
static_assert( SingleElementPolicy<MockContinuum>::is_homogeneous);
static_assert(!MultiElementPolicy::is_homogeneous);
static_assert( SingleElementPolicy<StructuralElement>::is_homogeneous);

// 6. StructuralElement satisfies FiniteElement
static_assert( FiniteElement<StructuralElement>);


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Test utilities
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

static int g_pass = 0, g_fail = 0;

static void report(const char* name, bool ok) {
    if (ok) { std::printf("  PASS  %s\n", name); ++g_pass; }
    else    { std::printf("  FAIL  %s\n", name); ++g_fail; }
}

#define TEST(name) void name(); \
    struct name##_reg { name##_reg() { name(); } } name##_inst; \
    void name()


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Runtime tests
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// -- FEM_Element wrapper: construction + topology queries --

TEST(test_fem_element_wrap_and_query) {
    FEM_Element fe(MockContinuum{.nodes_ = 8, .gps_ = 27, .id_ = 42});

    bool ok = (fe.num_nodes()              == 8)
           && (fe.num_integration_points() == 27)
           && (fe.sieve_id()               == 42);

    report(__func__, ok);
}

// -- FEM_Element wrapper: assembly methods are callable --

TEST(test_fem_element_assembly_forwarding) {
    FEM_Element fe(MockContinuum{});

    // These should not throw/crash (mock ignores PETSc args)
    fe.set_num_dof_in_nodes();
    fe.inject_K(nullptr);
    fe.compute_internal_forces(nullptr, nullptr);
    fe.inject_tangent_stiffness(nullptr, nullptr);
    fe.commit_material_state(nullptr);

    report(__func__, true);  // reaching here without crash = pass
}

// -- FEM_Element: deep copy preserves values --

TEST(test_fem_element_copy_semantics) {
    FEM_Element original(MockContinuum{.nodes_ = 5, .gps_ = 10, .id_ = 55});
    FEM_Element copied(original);

    bool ok = (copied.num_nodes()              == 5)
           && (copied.num_integration_points() == 10)
           && (copied.sieve_id()               == 55);

    // Verify original unaffected
    ok = ok && (original.num_nodes() == 5);

    report(__func__, ok);
}

// -- FEM_Element: copy-assignment --

TEST(test_fem_element_copy_assignment) {
    FEM_Element a(MockContinuum{.nodes_ = 3, .gps_ = 1, .id_ = 33});
    FEM_Element b(MockBeam{});                     // different type inside

    b = a;  // copy-assign

    bool ok = (b.num_nodes() == 3)
           && (b.sieve_id()  == 33);

    report(__func__, ok);
}

// -- FEM_Element: move semantics --

TEST(test_fem_element_move_semantics) {
    FEM_Element original(MockContinuum{.nodes_ = 7, .gps_ = 14, .id_ = 77});
    FEM_Element moved(std::move(original));

    bool ok = (moved.num_nodes() == 7)
           && (moved.sieve_id()  == 77);

    report(__func__, ok);
}

// -- FEM_Element: move-assignment --

TEST(test_fem_element_move_assignment) {
    FEM_Element a(MockContinuum{.nodes_ = 9, .gps_ = 3, .id_ = 91});
    FEM_Element b(MockBeam{});

    b = std::move(a);

    bool ok = (b.num_nodes() == 9)
           && (b.sieve_id()  == 91);

    report(__func__, ok);
}

// -- FEM_Element passed to FEM_Element uses move, not double-wrap --

TEST(test_fem_element_no_double_wrap) {
    FEM_Element inner(MockContinuum{.nodes_ = 6, .gps_ = 12, .id_ = 66});
    FEM_Element outer(std::move(inner));  // move constructor, not wrapping

    bool ok = (outer.num_nodes() == 6)
           && (outer.sieve_id()  == 66);

    report(__func__, ok);
}

// -- MultiElementPolicy: heterogeneous container --

TEST(test_multi_element_heterogeneous) {
    MultiElementPolicy::container_type elements;

    elements.emplace_back(MockContinuum{.nodes_ = 4, .gps_ = 4, .id_ = 1});
    elements.emplace_back(MockBeam{});          // 2 nodes, 3 gps, id=99
    elements.emplace_back(MockContinuum{.nodes_ = 8, .gps_ = 8, .id_ = 2});

    bool ok = (elements.size() == 3)
           && (elements[0].num_nodes() == 4)
           && (elements[0].sieve_id()  == 1)
           && (elements[1].num_nodes() == 2)
           && (elements[1].sieve_id()  == 99)
           && (elements[2].num_nodes() == 8)
           && (elements[2].sieve_id()  == 2);

    // Assembly loop — identical to what Model/NLAnalysis would do
    for (auto& e : elements) {
        e.set_num_dof_in_nodes();
        e.inject_K(nullptr);
        e.compute_internal_forces(nullptr, nullptr);
        e.inject_tangent_stiffness(nullptr, nullptr);
        e.commit_material_state(nullptr);
    }

    report(__func__, ok);
}

// -- SingleElementPolicy: homogeneous container, zero overhead --

TEST(test_single_element_policy) {
    SingleElementPolicy<MockContinuum>::container_type elements;

    elements.emplace_back(MockContinuum{.nodes_ = 4, .gps_ = 4, .id_ = 10});
    elements.emplace_back(MockContinuum{.nodes_ = 4, .gps_ = 4, .id_ = 11});
    elements.emplace_back(MockContinuum{.nodes_ = 4, .gps_ = 4, .id_ = 12});

    bool ok = (elements.size() == 3)
           && (elements[0].sieve_id() == 10)
           && (elements[2].sieve_id() == 12);

    // Same assembly loop — works identically
    for (auto& e : elements) {
        e.inject_K(nullptr);
        e.compute_internal_forces(nullptr, nullptr);
    }

    report(__func__, ok);
}

// -- Generic function constrained by FiniteElement works with any type --

template <FiniteElement E>
bool generic_assembly_check(E& element) {
    element.set_num_dof_in_nodes();
    element.inject_K(nullptr);
    return element.num_nodes() > 0;
}

TEST(test_generic_function_with_concept) {
    MockContinuum mc{.nodes_ = 4, .gps_ = 4, .id_ = 1};
    MockBeam      mb;
    FEM_Element   fe(MockContinuum{.nodes_ = 8, .gps_ = 8, .id_ = 2});

    bool ok = generic_assembly_check(mc)
           && generic_assembly_check(mb)
           && generic_assembly_check(fe);

    report(__func__, ok);
}

// -- ElementPolicyLike concept constrains template parameters --

template <ElementPolicyLike Policy>
bool policy_container_check() {
    typename Policy::container_type c;
    return c.empty() && (Policy::is_homogeneous || !Policy::is_homogeneous); // trivially true
}

TEST(test_element_policy_concept_constraint) {
    bool ok = policy_container_check<SingleElementPolicy<MockContinuum>>()
           && policy_container_check<MultiElementPolicy>()
           && policy_container_check<SingleElementPolicy<StructuralElement>>();

    report(__func__, ok);
}

// -- StructuralElement: type-erased wrapper for structural elements --

TEST(test_structural_element_wraps_mock) {
    StructuralElement se(MockBeam{});

    bool ok = (se.num_nodes()              == 2)
           && (se.num_integration_points() == 3)
           && (se.sieve_id()               == 99);

    // Assembly loop works through type erasure
    se.set_num_dof_in_nodes();
    se.inject_K(nullptr);
    se.compute_internal_forces(nullptr, nullptr);
    se.inject_tangent_stiffness(nullptr, nullptr);
    se.commit_material_state(nullptr);

    report(__func__, ok);
}

TEST(test_structural_element_copy_semantics) {
    StructuralElement se1(MockBeam{});
    StructuralElement se2 = se1;  // copy

    bool ok = (se2.num_nodes()  == 2)
           && (se2.sieve_id()   == 99);

    // Move
    StructuralElement se3 = std::move(se1);
    ok = ok && (se3.sieve_id() == 99);

    report(__func__, ok);
}

TEST(test_structural_element_policy_container) {
    SingleElementPolicy<StructuralElement>::container_type elements;

    elements.emplace_back(MockBeam{});
    elements.emplace_back(MockContinuum{.nodes_ = 8, .gps_ = 8, .id_ = 55});

    bool ok = (elements.size() == 2)
           && (elements[0].num_nodes() == 2)
           && (elements[1].num_nodes() == 8)
           && (elements[1].sieve_id()  == 55);

    for (auto& e : elements) {
        e.inject_K(nullptr);
        e.compute_internal_forces(nullptr, nullptr);
    }

    report(__func__, ok);
}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  main
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

int main() {
    std::puts("=== FiniteElement Concept + FEM_Element + ElementPolicy Tests ===");

    // Tests already ran via static registration above.

    std::printf("\n=== %d PASSED, %d FAILED ===\n", g_pass, g_fail);
    return g_fail > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
