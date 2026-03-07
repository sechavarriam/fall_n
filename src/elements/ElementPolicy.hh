#ifndef FALL_N_ELEMENT_POLICY_HH
#define FALL_N_ELEMENT_POLICY_HH

// =============================================================================
//  Element Storage Policies for Model
// =============================================================================
//
//  Control how Model stores its element collection:
//
//  SingleElementPolicy<E>
//    Homogeneous mesh — all elements share the same concrete type E.
//    Storage: std::vector<E>.  Zero overhead (no type erasure, no vtable).
//    Use when the entire model consists of a single element type.
//
//  MultiElementPolicy
//    Heterogeneous mesh — elements can be any type satisfying FiniteElement.
//    Storage: std::vector<FEM_Element>.  One virtual dispatch per call.
//    Use when mixing element types (continuum + beam, etc.).
//
//  Usage sketch in Model:
//
//    // Default: homogeneous, fastest path
//    Model<MP, ndofs>
//      →  SingleElementPolicy<ContinuumElement<MP, ndofs>>
//      →  std::vector<ContinuumElement<MP, ndofs>>
//
//    // Heterogeneous: mixed element types
//    Model<MP, ndofs, MultiElementPolicy>
//      →  std::vector<FEM_Element>
//
//  ElementPolicyLike concept constrains the policy parameter on Model,
//  ensuring the policy exposes element_type, container_type, and
//  is_homogeneous.
//
// =============================================================================

#include <vector>
#include <concepts>

#include "FiniteElementConcept.hh"
#include "FEM_Element.hh"
#include "StructuralElement.hh"


// ── Concept for element policies ─────────────────────────────────────────────

template <typename P>
concept ElementPolicyLike = requires {
    typename P::element_type;
    typename P::container_type;
    { P::is_homogeneous } -> std::convertible_to<bool>;
} && FiniteElement<typename P::element_type>;


// ── SingleElementPolicy ──────────────────────────────────────────────────────
//
//  Stores elements as vector<ElementT> — direct, no indirection.
//  ElementT must satisfy FiniteElement.

template <FiniteElement ElementT>
struct SingleElementPolicy {
    using element_type   = ElementT;
    using container_type = std::vector<ElementT>;
    static constexpr bool is_homogeneous = true;
};


// ── MultiElementPolicy ───────────────────────────────────────────────────────
//
//  Stores elements as vector<FEM_Element> — type-erased, polymorphic.
//  Any FiniteElement can be emplaced into the container.

struct MultiElementPolicy {
    using element_type   = FEM_Element;
    using container_type = std::vector<FEM_Element>;
    static constexpr bool is_homogeneous = false;
};


// ── Static verification ──────────────────────────────────────────────────────

static_assert(ElementPolicyLike<MultiElementPolicy>,
    "MultiElementPolicy must satisfy ElementPolicyLike");

static_assert(ElementPolicyLike<SingleElementPolicy<StructuralElement>>,
    "SingleElementPolicy<StructuralElement> must satisfy ElementPolicyLike");

#endif // FALL_N_ELEMENT_POLICY_HH
