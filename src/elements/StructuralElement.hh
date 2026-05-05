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
#include <typeinfo>
#include <vector>

#include <petsc.h>
#include <Eigen/Dense>

#include "FiniteElementConcept.hh"
#include "StructuralMassPolicy.hh"
#include "../materials/SectionConstitutiveSnapshot.hh"


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
        virtual void revert_material_state()                = 0;

        // Topology
        virtual std::size_t num_nodes()              const  = 0;
        virtual std::size_t num_integration_points() const  = 0;
        virtual PetscInt    sieve_id()               const  = 0;

        // Standalone-vector interface (for DynamicAnalysis parallel assembly)
        virtual Eigen::VectorXd extract_element_dofs(Vec /*u_local*/) const {
            return {};
        }
        virtual Eigen::VectorXd compute_internal_force_vector(const Eigen::VectorXd& /*u_e*/) {
            return {};
        }
        virtual Eigen::MatrixXd compute_tangent_stiffness_matrix(const Eigen::VectorXd& /*u_e*/) {
            return {};
        }
        virtual const std::vector<PetscInt>& get_dof_indices() {
            static const std::vector<PetscInt> empty;
            return empty;
        }

        // Mass matrix (optional, for dynamic analysis)
        virtual double density()                     const  { return 0.0; }
        virtual void   set_density(double /*rho*/)          {}
        virtual void   inject_mass(Mat /*M*/)               {}
        virtual void   set_structural_mass_policy(
            fall_n::StructuralMassPolicy /*policy*/) {}
        virtual fall_n::StructuralMassPolicy structural_mass_policy() const {
            return fall_n::StructuralMassPolicy::consistent;
        }

        // Introspection for structural-only post-processing/reconstruction.
        // This stays out of the FiniteElement concept and therefore out of
        // the solver hot path.
        virtual const std::type_info& concrete_type() const noexcept = 0;
        virtual const void* raw_ptr() const noexcept = 0;
        virtual void* raw_ptr() noexcept = 0;

        // Section constitutive snapshots (for observers, damage tracking, etc.)
        virtual std::vector<SectionConstitutiveSnapshot> section_snapshots() const {
            return {};
        }
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

        Eigen::VectorXd extract_element_dofs(Vec u_local) const override {
            if constexpr (requires { element_.extract_element_dofs(u_local); })
                return element_.extract_element_dofs(u_local);
            else return {};
        }
        Eigen::VectorXd compute_internal_force_vector(const Eigen::VectorXd& u_e) override {
            if constexpr (requires { element_.compute_internal_force_vector(u_e); })
                return element_.compute_internal_force_vector(u_e);
            else return {};
        }
        Eigen::MatrixXd compute_tangent_stiffness_matrix(const Eigen::VectorXd& u_e) override {
            if constexpr (requires { element_.compute_tangent_stiffness_matrix(u_e); })
                return element_.compute_tangent_stiffness_matrix(u_e);
            else return {};
        }
        const std::vector<PetscInt>& get_dof_indices() override {
            if constexpr (requires { element_.get_dof_indices(); })
                return element_.get_dof_indices();
            else { static const std::vector<PetscInt> empty; return empty; }
        }

        double density() const override {
            if constexpr (requires { element_.density(); })
                return element_.density();
            else return 0.0;
        }
        void set_density(double rho) override {
            if constexpr (requires { element_.set_density(rho); })
                element_.set_density(rho);
        }
        void inject_mass(Mat M) override {
            if constexpr (requires { element_.inject_mass(M); })
                element_.inject_mass(M);
        }
        void set_structural_mass_policy(fall_n::StructuralMassPolicy policy) override {
            if constexpr (requires { element_.set_structural_mass_policy(policy); })
                element_.set_structural_mass_policy(policy);
        }
        fall_n::StructuralMassPolicy structural_mass_policy() const override {
            if constexpr (requires { element_.structural_mass_policy(); })
                return element_.structural_mass_policy();
            else
                return fall_n::StructuralMassPolicy::consistent;
        }

        const std::type_info& concrete_type() const noexcept override { return typeid(T); }
        const void* raw_ptr() const noexcept override { return &element_; }
        void* raw_ptr() noexcept override { return &element_; }

        std::vector<SectionConstitutiveSnapshot> section_snapshots() const override {
            if constexpr (requires { element_.sections(); }) {
                const auto& secs = element_.sections();
                std::vector<SectionConstitutiveSnapshot> result;
                result.reserve(secs.size());
                for (const auto& s : secs)
                    result.push_back(s.section_snapshot());
                return result;
            } else {
                return {};
            }
        }
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
    void revert_material_state()                   { pimpl_->revert_material_state(); }

    auto num_nodes()              const -> std::size_t { return pimpl_->num_nodes(); }
    auto num_integration_points() const -> std::size_t { return pimpl_->num_integration_points(); }
    auto sieve_id()               const -> PetscInt    { return pimpl_->sieve_id(); }

    Eigen::VectorXd extract_element_dofs(Vec u_local) const { return pimpl_->extract_element_dofs(u_local); }
    Eigen::VectorXd compute_internal_force_vector(const Eigen::VectorXd& u_e) { return pimpl_->compute_internal_force_vector(u_e); }
    Eigen::MatrixXd compute_tangent_stiffness_matrix(const Eigen::VectorXd& u_e) { return pimpl_->compute_tangent_stiffness_matrix(u_e); }
    const std::vector<PetscInt>& get_dof_indices() { return pimpl_->get_dof_indices(); }

    auto density()           const -> double   { return pimpl_->density(); }
    void set_density(double rho)               { pimpl_->set_density(rho); }
    void inject_mass(Mat M)                    { pimpl_->inject_mass(M); }
    void set_structural_mass_policy(fall_n::StructuralMassPolicy policy)
    { pimpl_->set_structural_mass_policy(policy); }
    auto structural_mass_policy() const -> fall_n::StructuralMassPolicy
    { return pimpl_->structural_mass_policy(); }

    const std::type_info& concrete_type() const noexcept { return pimpl_->concrete_type(); }

    /// Section constitutive snapshots across all integration points.
    /// Empty if the element doesn't expose sections (e.g., pure continuum).
    [[nodiscard]] std::vector<SectionConstitutiveSnapshot> section_snapshots() const {
        return pimpl_->section_snapshots();
    }

    template <typename T>
    [[nodiscard]] const T* as() const noexcept {
        if (pimpl_->concrete_type() != typeid(T)) return nullptr;
        return static_cast<const T*>(pimpl_->raw_ptr());
    }

    template <typename T>
    [[nodiscard]] T* as() noexcept {
        if (pimpl_->concrete_type() != typeid(T)) return nullptr;
        return static_cast<T*>(pimpl_->raw_ptr());
    }
};


// ── Concept verification ─────────────────────────────────────────────────────

static_assert(FiniteElement<StructuralElement>,
    "StructuralElement must satisfy the FiniteElement concept");


#endif // FALL_STRUCTURAL_ELEMENT_ABSTRACTION
