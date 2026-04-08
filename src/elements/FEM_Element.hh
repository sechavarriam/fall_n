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

#include <Eigen/Core>

#include "FiniteElementConcept.hh"
#include "../materials/InternalFieldSnapshot.hh"


// =============================================================================
//  GaussPointSnapshot — per integration point state for multiscale queries
// =============================================================================
//
//  Returned by FEM_Element::gauss_point_snapshots() to expose material
//  state through the type-erased interface.  This is the bridge that allows
//  MixedModel (MultiElementPolicy) sub-models to report crack/damage data
//  without requiring typed access to ContinuumElement::material_points().
//
//  Design:
//    - Self-contained value type (no dangling references).
//    - Carries the fields needed for multiscale crack tracking:
//      position, stress, strain, damage, crack normals/openings.
//    - Elements without material points return an empty vector (default).
//
// =============================================================================

struct GaussPointSnapshot {
    Eigen::Vector3d position{Eigen::Vector3d::Zero()};
    Eigen::Vector<double, 6> stress{Eigen::Vector<double, 6>::Zero()};
    Eigen::Vector<double, 6> strain{Eigen::Vector<double, 6>::Zero()};

    double damage{0.0};
    bool damage_scalar_available{false};
    int    num_cracks{0};
    bool fracture_history_available{false};
    double sigma_o_max{0.0};
    double tau_o_max{0.0};

    std::array<Eigen::Vector3d, 3> crack_normals{
        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
    std::array<double, 3> crack_openings{0.0, 0.0, 0.0};
    std::array<bool, 3>   crack_closed{true, true, true};
};


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

        // Standalone-vector interface for local linearisation / assembly.
        virtual Eigen::VectorXd extract_element_dofs(Vec /*u_local*/) {
            return {};
        }
        virtual Eigen::VectorXd
        compute_internal_force_vector(const Eigen::VectorXd& /*u_e*/) {
            return {};
        }
        virtual Eigen::MatrixXd
        compute_tangent_stiffness_matrix(const Eigen::VectorXd& /*u_e*/) {
            return {};
        }
        virtual const std::vector<PetscInt>& get_dof_indices() {
            static const std::vector<PetscInt> empty;
            return empty;
        }

        // Mass matrix (optional — defaults to no-op for elements without mass)
        virtual double density()                     const  { return 0.0; }
        virtual void   set_density(double /*rho*/)          {}
        virtual void   inject_mass(Mat /*M*/)               {}

        // Post-processing: Gauss-point field export for VTK
        virtual std::vector<GaussFieldRecord>
            collect_gauss_fields(Vec /*u_local*/) const { return {}; }

        // Multiscale: per-GP state snapshot (crack data, damage, stress/strain)
        virtual std::vector<GaussPointSnapshot>
            gauss_point_snapshots(Vec /*u_local*/) const { return {}; }
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

        Eigen::VectorXd extract_element_dofs(Vec u_local) override {
            if constexpr (requires { element_.extract_element_dofs(u_local); })
                return element_.extract_element_dofs(u_local);
            else
                return {};
        }
        Eigen::VectorXd
        compute_internal_force_vector(const Eigen::VectorXd& u_e) override {
            if constexpr (requires { element_.compute_internal_force_vector(u_e); })
                return element_.compute_internal_force_vector(u_e);
            else
                return {};
        }
        Eigen::MatrixXd
        compute_tangent_stiffness_matrix(const Eigen::VectorXd& u_e) override {
            if constexpr (requires { element_.compute_tangent_stiffness_matrix(u_e); })
                return element_.compute_tangent_stiffness_matrix(u_e);
            else
                return {};
        }
        const std::vector<PetscInt>& get_dof_indices() override {
            if constexpr (requires { element_.get_dof_indices(); })
                return element_.get_dof_indices();
            else {
                static const std::vector<PetscInt> empty;
                return empty;
            }
        }

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

        std::vector<GaussPointSnapshot>
        gauss_point_snapshots([[maybe_unused]] Vec u_local) const override {
            // Elements that expose material_points() (ContinuumElement) can
            // be queried for per-GP crack/damage state.  Others return {}.
            if constexpr (requires { element_.material_points(); }) {
                std::vector<GaussPointSnapshot> out;
                const auto& mps = element_.material_points();
                out.reserve(mps.size());

                // Access Gauss-point coordinates via the element geometry.
                // The geometry's integration_points() provides (x,y,z) for each GP.
                [[maybe_unused]] bool has_geometry = false;
                const void* geom_ptr = nullptr;
                if constexpr (requires { element_.get_geometry(); }) {
                    geom_ptr = element_.get_geometry();
                    has_geometry = true;
                }

                std::size_t gp_idx = 0;
                for (const auto& mp : mps) {
                    GaussPointSnapshot snap;

                    // Position from geometry integration points
                    if constexpr (requires { element_.get_geometry(); }) {
                        const auto& gpts = element_.get_geometry()->integration_points();
                        if (gp_idx < gpts.size()) {
                            snap.position = Eigen::Vector3d{
                                gpts[gp_idx].coord(0),
                                gpts[gp_idx].coord(1),
                                gpts[gp_idx].coord(2)};
                        }
                    }

                    // Stress/strain from current material state
                    const auto& state = mp.current_state();
                    auto response = mp.compute_response(state);
                    const auto& sv = response.components();
                    const auto& ev = state.components();
                    for (int i = 0; i < std::min<int>(6, sv.size()); ++i)
                        snap.stress[i] = sv[i];
                    for (int i = 0; i < std::min<int>(6, ev.size()); ++i)
                        snap.strain[i] = ev[i];

                    // Crack/damage from internal field snapshot
                    auto ifs = mp.internal_field_snapshot();
                    snap.damage     = ifs.damage.value_or(0.0);
                    snap.damage_scalar_available = ifs.has_damage();
                    snap.num_cracks = ifs.num_cracks.value_or(0);
                    snap.fracture_history_available =
                        ifs.has_fracture_history();
                    snap.sigma_o_max = ifs.sigma_o_max.value_or(0.0);
                    snap.tau_o_max   = ifs.tau_o_max.value_or(0.0);

                    auto to_vec = [](const auto& opt) -> Eigen::Vector3d {
                        if (!opt) return Eigen::Vector3d::Zero();
                        return Eigen::Vector3d{(*opt)[0], (*opt)[1], (*opt)[2]};
                    };
                    snap.crack_normals[0] = to_vec(ifs.crack_normal_1);
                    snap.crack_normals[1] = to_vec(ifs.crack_normal_2);
                    snap.crack_normals[2] = to_vec(ifs.crack_normal_3);
                    snap.crack_openings[0] = ifs.crack_strain_1.value_or(0.0);
                    snap.crack_openings[1] = ifs.crack_strain_2.value_or(0.0);
                    snap.crack_openings[2] = ifs.crack_strain_3.value_or(0.0);
                    snap.crack_closed[0] = ifs.crack_closed_1.value_or(1.0) > 0.5;
                    snap.crack_closed[1] = ifs.crack_closed_2.value_or(1.0) > 0.5;
                    snap.crack_closed[2] = ifs.crack_closed_3.value_or(1.0) > 0.5;

                    out.push_back(snap);
                    ++gp_idx;
                }
                return out;
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

    Eigen::VectorXd extract_element_dofs(Vec u_local)
    { return pimpl_->extract_element_dofs(u_local); }
    Eigen::VectorXd compute_internal_force_vector(const Eigen::VectorXd& u_e)
    { return pimpl_->compute_internal_force_vector(u_e); }
    Eigen::MatrixXd compute_tangent_stiffness_matrix(const Eigen::VectorXd& u_e)
    { return pimpl_->compute_tangent_stiffness_matrix(u_e); }
    const std::vector<PetscInt>& get_dof_indices()
    { return pimpl_->get_dof_indices(); }

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

    // Multiscale: per-GP state snapshot through type-erased interface
    auto gauss_point_snapshots(Vec u_local) const
        -> std::vector<GaussPointSnapshot>
    {
        return pimpl_->gauss_point_snapshots(u_local);
    }
};

// FEM_Element itself satisfies FiniteElement (recursive erasure is valid)
static_assert(FiniteElement<FEM_Element>,
    "FEM_Element must satisfy the FiniteElement concept");

#endif // FALL_N_FEM_ELEMENT_HH
