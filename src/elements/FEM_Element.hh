#ifndef FALL_N_FEM_ELEMENT_HH
#define FALL_N_FEM_ELEMENT_HH

// =============================================================================
//  FEM_Element — Type-erased finite element wrapper (External Polymorphism)
// =============================================================================
//
//  Wraps any type satisfying the FiniteElement concept behind a uniform
//  value-semantic interface.  Used by MultiElementPolicy to store
//  heterogeneous elements (e.g. continuum + beam) in a single container.
//
//  Design:
//    - External Polymorphism (Concept / Model / type-erased handle)
//    - Value semantics (deep copy via clone(), move = default)
//    - The wrapper itself satisfies FiniteElement (recursive erasure OK)
//    - Self-wrapping guard prevents accidental double indirection
//
//  Cost of the indirection (per element, per method call):
//    - One virtual dispatch + one pointer chase (unique_ptr)
//    - Negligible vs. the Eigen linear algebra inside each element
//
//  When homogeneous storage suffices (all elements share the same concrete
//  type), use SingleElementPolicy<E> instead — zero overhead.
//
// =============================================================================

#include <memory>
#include <cstddef>
#include <type_traits>
#include <vector>

#include <petsc.h>

#include "FiniteElementConcept.hh"
#include "../materials/InternalFieldSnapshot.hh"

class FEM_Element {

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
        virtual void revert_material_state()                = 0;

        // Topology
        virtual std::size_t num_nodes()              const  = 0;
        virtual std::size_t num_integration_points() const  = 0;
        virtual PetscInt    sieve_id()               const  = 0;

        // Mass matrix (optional — defaults to no-op for elements without mass)
        virtual double density()                     const  { return 0.0; }
        virtual void   set_density(double /*rho*/)          {}
        virtual void   inject_mass(Mat /*M*/)               {}

        // Post-processing: Gauss-point field export for VTK
        virtual std::vector<GaussFieldRecord>
            collect_gauss_fields(Vec /*u_local*/) const { return {}; }
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
        void revert_material_state()                override { element_.revert_material_state(); }

        std::size_t num_nodes()              const  override { return element_.num_nodes(); }
        std::size_t num_integration_points() const  override { return element_.num_integration_points(); }
        PetscInt    sieve_id()               const  override { return element_.sieve_id(); }

        // Mass matrix — forward if element supports it, else use defaults
        double density() const override {
            if constexpr (requires { element_.density(); })
                return element_.density();
            else
                return 0.0;
        }
        void set_density(double rho) override {
            if constexpr (requires { element_.set_density(rho); })
                element_.set_density(rho);
        }
        void inject_mass(Mat M) override {
            if constexpr (requires { element_.inject_mass(M); })
                element_.inject_mass(M);
        }

        std::vector<GaussFieldRecord>
        collect_gauss_fields(Vec u_local) const override {
            if constexpr (requires { element_.collect_gauss_fields(u_local); })
                return element_.collect_gauss_fields(u_local);
            else
                return {};
        }
    };

    // ── Pimpl ─────────────────────────────────────────────────────

    std::unique_ptr<Concept> pimpl_;

public:

    // ── Construct from any FiniteElement (self-wrap guard) ────────

    template <FiniteElement T>
        requires (!std::same_as<std::remove_cvref_t<T>, FEM_Element>)
    FEM_Element(T element)                                              // NOLINT(google-explicit-constructor)
        : pimpl_(std::make_unique<Model<std::remove_cvref_t<T>>>(
              std::move(element))) {}

    // ── Value semantics ──────────────────────────────────────────

    FEM_Element(const FEM_Element& other) : pimpl_(other.pimpl_->clone()) {}

    FEM_Element& operator=(const FEM_Element& other) {
        FEM_Element tmp(other);         // copy-and-swap
        pimpl_.swap(tmp.pimpl_);
        return *this;
    }

    FEM_Element(FEM_Element&&) noexcept = default;
    FEM_Element& operator=(FEM_Element&&) noexcept = default;
    ~FEM_Element() = default;

    // ── Forwarding interface (satisfies FiniteElement) ────────────

    void set_num_dof_in_nodes()                    { pimpl_->set_num_dof_in_nodes(); }
    void inject_K(Mat K)                           { pimpl_->inject_K(K); }
    void compute_internal_forces(Vec u, Vec f)     { pimpl_->compute_internal_forces(u, f); }
    void inject_tangent_stiffness(Vec u, Mat K)    { pimpl_->inject_tangent_stiffness(u, K); }
    void commit_material_state(Vec u)              { pimpl_->commit_material_state(u); }
    void revert_material_state()                   { pimpl_->revert_material_state(); }

    auto num_nodes()              const -> std::size_t { return pimpl_->num_nodes(); }
    auto num_integration_points() const -> std::size_t { return pimpl_->num_integration_points(); }
    auto sieve_id()               const -> PetscInt    { return pimpl_->sieve_id(); }

    // Mass matrix interface
    auto density()           const -> double   { return pimpl_->density(); }
    void set_density(double rho)               { pimpl_->set_density(rho); }
    void inject_mass(Mat M)                    { pimpl_->inject_mass(M); }

    // Post-processing: type-erased Gauss-point field export
    auto collect_gauss_fields(Vec u_local) const
        -> std::vector<GaussFieldRecord>
    {
        return pimpl_->collect_gauss_fields(u_local);
    }
};

// FEM_Element itself satisfies FiniteElement (recursive erasure is valid)
static_assert(FiniteElement<FEM_Element>,
    "FEM_Element must satisfy the FiniteElement concept");

#endif // FALL_N_FEM_ELEMENT_HH