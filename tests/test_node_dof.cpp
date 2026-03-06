// =============================================================================
//  Tests for DoF Storage Policies and Refactored Node<Dim, Storage>
// =============================================================================

#include <cassert>
#include <cstddef>
#include <cstring>
#include <numeric>
#include <vector>

#include <petscsys.h>

#include "src/model/DoFStorage.hh"
#include "src/elements/Node.hh"

namespace {

// ─── InlineDoFs tests ────────────────────────────────────────────────────────

void test_inline_default() {
    dof::InlineDoFs<6> s;
    assert(s.size() == 0);
    assert(s.empty());
    assert(s.capacity() == 6);
}

void test_inline_resize_and_access() {
    dof::InlineDoFs<4> s;
    s.resize(3);
    assert(s.size() == 3);
    assert(!s.empty());

    // Initially zero
    for (std::size_t i = 0; i < 3; ++i) assert(s[i] == 0);

    s[0] = 10; s[1] = 20; s[2] = 30;
    assert(s.data()[0] == 10);
    assert(s.data()[1] == 20);
    assert(s.data()[2] == 30);
}

void test_inline_assign() {
    dof::InlineDoFs<6> s;
    dof::index_t arr[] = {5, 6, 7};
    s.assign(arr, arr + 3);
    assert(s.size() == 3);
    assert(s[0] == 5 && s[1] == 6 && s[2] == 7);
}

void test_inline_iterate() {
    dof::InlineDoFs<4> s;
    s.resize(3);
    s[0] = 1; s[1] = 2; s[2] = 3;

    dof::index_t sum = 0;
    for (auto v : s) sum += v;
    assert(sum == 6);
}

void test_inline_span() {
    dof::InlineDoFs<6> s;
    s.resize(2);
    s[0] = 42; s[1] = 43;
    auto sp = s.as_span();
    assert(sp.size() == 2);
    assert(sp[0] == 42 && sp[1] == 43);
}

// ─── DynamicDoFStorage tests ─────────────────────────────────────────────────

void test_dynamic_default() {
    dof::DynamicDoFStorage s;
    assert(s.size() == 0);
    assert(s.empty());
}

void test_dynamic_resize_and_access() {
    dof::DynamicDoFStorage s;
    s.resize(5);
    assert(s.size() == 5);
    for (std::size_t i = 0; i < 5; ++i) s[i] = static_cast<dof::index_t>(i * 10);
    assert(s[3] == 30);
}

void test_dynamic_assign() {
    dof::DynamicDoFStorage s;
    dof::index_t arr[] = {100, 200, 300, 400};
    s.assign(arr, arr + 4);
    assert(s.size() == 4);
    assert(s[2] == 300);
}

void test_dynamic_unlimited_capacity() {
    dof::DynamicDoFStorage s;
    s.resize(1000);
    assert(s.size() == 1000);
    s[999] = 42;
    assert(s[999] == 42);
}

// ─── SmallDoFs SBO tests ────────────────────────────────────────────────────

void test_sbo_default() {
    dof::SmallDoFs<4> s;
    assert(s.size() == 0);
    assert(s.empty());
    assert(s.capacity() == 4);
}

void test_sbo_inline_path() {
    dof::SmallDoFs<4> s;
    s.resize(3);  // within inline capacity
    assert(s.size() == 3);
    s[0] = 10; s[1] = 20; s[2] = 30;
    assert(s.data()[1] == 20);
}

void test_sbo_heap_fallback() {
    dof::SmallDoFs<2> s;
    s.resize(10);  // exceeds inline capacity → heap
    assert(s.size() == 10);
    assert(s.capacity() >= 10);
    for (std::size_t i = 0; i < 10; ++i) s[i] = static_cast<dof::index_t>(i);
    assert(s[9] == 9);
}

void test_sbo_copy() {
    dof::SmallDoFs<4> a;
    a.resize(3);
    a[0] = 1; a[1] = 2; a[2] = 3;

    dof::SmallDoFs<4> b = a;  // copy ctor
    assert(b.size() == 3);
    assert(b[0] == 1 && b[1] == 2 && b[2] == 3);

    // Mutating copy doesn't affect original
    b[0] = 99;
    assert(a[0] == 1);
}

void test_sbo_copy_heap() {
    dof::SmallDoFs<2> a;
    a.resize(5);  // heap path
    for (std::size_t i = 0; i < 5; ++i) a[i] = static_cast<dof::index_t>(i * 10);

    dof::SmallDoFs<2> b = a;
    assert(b.size() == 5);
    assert(b[3] == 30);

    b[3] = 999;
    assert(a[3] == 30);  // deep copy: original unchanged
}

void test_sbo_move() {
    dof::SmallDoFs<4> a;
    a.resize(3);
    a[0] = 7; a[1] = 8; a[2] = 9;

    dof::SmallDoFs<4> b = std::move(a);
    assert(b.size() == 3);
    assert(b[0] == 7 && b[2] == 9);
    assert(a.size() == 0);  // moved-from is empty // NOLINT
}

void test_sbo_move_heap() {
    dof::SmallDoFs<2> a;
    a.resize(8);
    for (std::size_t i = 0; i < 8; ++i) a[i] = static_cast<dof::index_t>(i);

    dof::SmallDoFs<2> b = std::move(a);
    assert(b.size() == 8);
    assert(b[7] == 7);
    assert(a.size() == 0);  // NOLINT
}

void test_sbo_assign_operator() {
    dof::SmallDoFs<4> a, b;
    a.resize(2); a[0] = 5; a[1] = 6;
    b.resize(3); b[0] = 10; b[1] = 20; b[2] = 30;

    b = a;
    assert(b.size() == 2);
    assert(b[0] == 5 && b[1] == 6);
}

void test_sbo_iterate() {
    dof::SmallDoFs<6> s;
    s.resize(4);
    for (std::size_t i = 0; i < 4; ++i) s[i] = static_cast<dof::index_t>(i + 1);

    dof::index_t sum = 0;
    for (auto v : s) sum += v;
    assert(sum == 10);  // 1+2+3+4
}

// ─── Concept static checks ──────────────────────────────────────────────────

void test_concept_satisfaction() {
    static_assert(dof::DoFStorageLike<dof::InlineDoFs<3>>);
    static_assert(dof::DoFStorageLike<dof::InlineDoFs<6>>);
    static_assert(dof::DoFStorageLike<dof::DynamicDoFStorage>);
    static_assert(dof::DoFStorageLike<dof::SmallDoFs<2>>);
    static_assert(dof::DoFStorageLike<dof::SmallDoFs<6>>);
    static_assert(dof::DoFStorageLike<dof::DefaultDoFStorage>);
}

// ─── Node<Dim, Storage> tests ────────────────────────────────────────────────

void test_node_construction() {
    Node<3> n{0, 1.0, 2.0, 3.0};
    assert(n.id() == 0);
    assert(n.coord(0) == 1.0);
    assert(n.coord(1) == 2.0);
    assert(n.coord(2) == 3.0);
    assert(n.num_dof() == 0);
    assert(n.dim == 3);
}

void test_node_set_num_dof() {
    Node<3> n{5, 0.0, 0.0, 0.0};
    n.set_num_dof(3);
    assert(n.num_dof() == 3);
    // Indices should be zero-initialized
    for (std::size_t i = 0; i < 3; ++i) assert(n.dof_index()[i] == 0);
}

void test_node_set_dof_index_single() {
    Node<2> n{0, 0.0, 0.0};
    n.set_num_dof(2);
    n.set_dof_index(0, 42);
    n.set_dof_index(1, 43);
    assert(n.dof_index()[0] == 42);
    assert(n.dof_index()[1] == 43);
}

void test_node_set_dof_index_range() {
    Node<3> n{0, 0.0, 0.0, 0.0};
    // Simulating Model::set_dof_index flow:
    //   1. set_num_dof from element setup
    //   2. set_dof_index from PetscSection
    n.set_num_dof(3);
    auto idxs = std::ranges::iota_view{PetscInt(10), PetscInt(13)};
    n.set_dof_index(idxs);
    assert(n.num_dof() == 3);
    assert(n.dof_index()[0] == 10);
    assert(n.dof_index()[1] == 11);
    assert(n.dof_index()[2] == 12);
}

void test_node_fix_dof_idempotent() {
    Node<2> n{0, 0.0, 0.0};
    n.set_num_dof(2);
    n.set_dof_index(0, 0);  // edge case: index 0
    n.set_dof_index(1, 5);

    n.fix_dof(0);
    assert(n.is_dof_fixed(0));
    assert(n.dof_index()[0] < 0);

    // Idempotent: fixing again doesn't change value
    auto val_after_first_fix = n.dof_index()[0];
    n.fix_dof(0);
    assert(n.dof_index()[0] == val_after_first_fix);

    // Release
    n.release_dof(0);
    assert(!n.is_dof_fixed(0));
    assert(n.dof_index()[0] == 0);

    // Fix index 5
    n.fix_dof(1);
    assert(n.dof_index()[1] == -(5 + 1));  // -(v+1) encoding
    n.release_dof(1);
    assert(n.dof_index()[1] == 5);
}

void test_node_sieve_id() {
    Node<3> n{0, 0.0, 0.0, 0.0};
    assert(!n.sieve_id.has_value());

    n.set_sieve_id(42);
    assert(n.sieve_id.has_value());
    assert(n.sieve_id.value() == 42);
}

void test_node_set_dof_interface_noop() {
    Node<3> n{0, 0.0, 0.0, 0.0};
    n.set_dof_interface();  // should be a no-op, no crash
    assert(n.num_dof() == 0);
}

void test_node_with_inline_storage() {
    Node<3, dof::InlineDoFs<3>> n{0, 1.0, 2.0, 3.0};
    n.set_num_dof(3);
    n.set_dof_index(0, 100);
    n.set_dof_index(1, 101);
    n.set_dof_index(2, 102);
    assert(n.dof_index()[2] == 102);
}

void test_node_with_dynamic_storage() {
    Node<3, dof::DynamicDoFStorage> n{0, 0.0, 0.0, 0.0};
    n.set_num_dof(100);  // unusual but valid
    assert(n.num_dof() == 100);
    n.set_dof_index(99, 999);
    assert(n.dof_index()[99] == 999);
}

void test_node_concept() {
    static_assert(NodeT<Node<3>>);
    static_assert(NodeT<Node<2>>);
    static_assert(NodeT<Node<1>>);
    static_assert(NodeT<Node<3, dof::InlineDoFs<6>>>);
    static_assert(NodeT<Node<3, dof::DynamicDoFStorage>>);
    static_assert(NodeT<Node<3, dof::SmallDoFs<3>>>);

    static_assert(geometry::PointT<Node<3>>);
}

void test_node_dof_span_contiguous() {
    Node<3> n{0, 0.0, 0.0, 0.0};
    n.set_num_dof(3);
    auto idxs = std::ranges::iota_view{PetscInt(0), PetscInt(3)};
    n.set_dof_index(idxs);

    auto sp = n.dof_index();
    // PETSc requires contiguous pointer:
    assert(sp.data() != nullptr);
    assert(sp.size() == 3);
    assert(sp[0] == 0 && sp[1] == 1 && sp[2] == 2);
}

void test_node_copy_value_semantics() {
    Node<3> a{0, 1.0, 2.0, 3.0};
    a.set_num_dof(3);
    a.set_dof_index(0, 10);
    a.set_dof_index(1, 11);
    a.set_dof_index(2, 12);

    Node<3> b = a;  // copy
    assert(b.dof_index()[0] == 10);

    // Mutating copy doesn't affect original (NO shared_ptr aliasing anymore!)
    b.set_dof_index(0, 999);
    assert(a.dof_index()[0] == 10);  // unchanged
    assert(b.dof_index()[0] == 999);
}

// ─── Test runner ─────────────────────────────────────────────────────────────

void print_pass(const char* name) {
    // Using printf to avoid <iostream> cost
    __builtin_printf("[PASS] %s\n", name);
}

#define RUN(fn) do { fn(); print_pass(#fn); } while (0)

} // namespace

int main() {
    // InlineDoFs
    RUN(test_inline_default);
    RUN(test_inline_resize_and_access);
    RUN(test_inline_assign);
    RUN(test_inline_iterate);
    RUN(test_inline_span);

    // DynamicDoFStorage
    RUN(test_dynamic_default);
    RUN(test_dynamic_resize_and_access);
    RUN(test_dynamic_assign);
    RUN(test_dynamic_unlimited_capacity);

    // SmallDoFs SBO
    RUN(test_sbo_default);
    RUN(test_sbo_inline_path);
    RUN(test_sbo_heap_fallback);
    RUN(test_sbo_copy);
    RUN(test_sbo_copy_heap);
    RUN(test_sbo_move);
    RUN(test_sbo_move_heap);
    RUN(test_sbo_assign_operator);
    RUN(test_sbo_iterate);

    // Concept
    RUN(test_concept_satisfaction);

    // Node with various storages
    RUN(test_node_construction);
    RUN(test_node_set_num_dof);
    RUN(test_node_set_dof_index_single);
    RUN(test_node_set_dof_index_range);
    RUN(test_node_fix_dof_idempotent);
    RUN(test_node_sieve_id);
    RUN(test_node_set_dof_interface_noop);
    RUN(test_node_with_inline_storage);
    RUN(test_node_with_dynamic_storage);
    RUN(test_node_concept);
    RUN(test_node_dof_span_contiguous);
    RUN(test_node_copy_value_semantics);

    __builtin_printf("\n=== All %d Node/DoF tests PASSED ===\n", 31);
    return 0;
}
