#ifndef FALL_STRUCTURAL_ELEMENT_ABSTRACTION
#define FALL_STRUCTURAL_ELEMENT_ABSTRACTION

// =============================================================================
//  StructuralElement — Type-erased wrapper for structural finite elements
// =============================================================================
//
//  Analogous to FEM_Element (which wraps *any* FiniteElement), but
//  scoped to the structural element family: beams, shells, cables, etc.
//
//  Use case:
//    SingleElementPolicy<StructuralElement>
//      → std::vector<StructuralElement>
//      → each wraps a concrete structural element (BeamElement, ShellElement…)
//
//  Why a separate type from FEM_Element?
//    1. Architectural clarity — structural vs continuum vs mixed.
//    2. Future: extended virtual interface for structural-specific queries
//       (section forces, beam frame, section geometry) without polluting
//       the base FiniteElement concept.
//    3. Type safety — prevents accidentally mixing continuum elements
//       into a "structural-only" model.
//
//  For truly heterogeneous meshes (continuum + structural), use
//  MultiElementPolicy with FEM_Element instead.
//
//  Design: External Polymorphism (Concept/Model/pimpl_), identical
//  pattern to FEM_Element. Value semantics via clone().
//
// =============================================================================

#include <memory>
#include <cstddef>
#include <type_traits>

#include <petsc.h>

#include "FiniteElementConcept.hh"


class StructuralElement {

    // ── Inner concept (virtual interface) ─────────────────────────

    struct Concept {
        virtual ~Concept() = default;
        virtual std::unique_ptr<Concept> clone() const = 0;

        // Assembly interface (mirrors FiniteElement concept)
        virtual void set_num_dof_in_nodes()                 = 0;
        virtual void inject_K(Mat K)                        = 0;
        virtual void compute_internal_forces(Vec u, Vec f)  = 0;
        virtual void inject_tangent_stiffness(Vec u, Mat K) = 0;
        virtual void commit_material_state(Vec u)           = 0;

        // Topology
        virtual std::size_t num_nodes()              const  = 0;
        virtual std::size_t num_integration_points() const  = 0;
        virtual PetscInt    sieve_id()               const  = 0;
    };

    // ── Inner model (type-specific bridge) ────────────────────────

    template <FiniteElement T>
    struct Model final : Concept {
        T element_;

        explicit Model(T elem) : element_(std::move(elem)) {}

        std::unique_ptr<Concept> clone() const override {
            return std::make_unique<Model>(*this);
        }

        void set_num_dof_in_nodes()                 override { element_.set_num_dof_in_nodes(); }
        void inject_K(Mat K)                        override { element_.inject_K(K); }
        void compute_internal_forces(Vec u, Vec f)  override { element_.compute_internal_forces(u, f); }
        void inject_tangent_stiffness(Vec u, Mat K) override { element_.inject_tangent_stiffness(u, K); }
        void commit_material_state(Vec u)           override { element_.commit_material_state(u); }

        std::size_t num_nodes()              const  override { return element_.num_nodes(); }
        std::size_t num_integration_points() const  override { return element_.num_integration_points(); }
        PetscInt    sieve_id()               const  override { return element_.sieve_id(); }
    };

    // ── Pimpl ─────────────────────────────────────────────────────

    std::unique_ptr<Concept> pimpl_;

public:

    // ── Construct from any FiniteElement (self-wrap guard) ────────

    template <FiniteElement T>
        requires (!std::same_as<std::remove_cvref_t<T>, StructuralElement>)
    StructuralElement(T element)                                        // NOLINT(google-explicit-constructor)
        : pimpl_(std::make_unique<Model<std::remove_cvref_t<T>>>(
              std::move(element))) {}

    // ── Value semantics ──────────────────────────────────────────

    StructuralElement(const StructuralElement& other)
        : pimpl_(other.pimpl_->clone()) {}

    StructuralElement& operator=(const StructuralElement& other) {
        StructuralElement tmp(other);
        pimpl_.swap(tmp.pimpl_);
        return *this;
    }

    StructuralElement(StructuralElement&&) noexcept = default;
    StructuralElement& operator=(StructuralElement&&) noexcept = default;
    ~StructuralElement() = default;

    // ── Forwarding interface (satisfies FiniteElement) ────────────

    void set_num_dof_in_nodes()                    { pimpl_->set_num_dof_in_nodes(); }
    void inject_K(Mat K)                           { pimpl_->inject_K(K); }
    void compute_internal_forces(Vec u, Vec f)     { pimpl_->compute_internal_forces(u, f); }
    void inject_tangent_stiffness(Vec u, Mat K)    { pimpl_->inject_tangent_stiffness(u, K); }
    void commit_material_state(Vec u)              { pimpl_->commit_material_state(u); }

    auto num_nodes()              const -> std::size_t { return pimpl_->num_nodes(); }
    auto num_integration_points() const -> std::size_t { return pimpl_->num_integration_points(); }
    auto sieve_id()               const -> PetscInt    { return pimpl_->sieve_id(); }
};


// ── Concept verification ─────────────────────────────────────────────────────

static_assert(FiniteElement<StructuralElement>,
    "StructuralElement must satisfy the FiniteElement concept");


#endif // FALL_STRUCTURAL_ELEMENT_ABSTRACTION