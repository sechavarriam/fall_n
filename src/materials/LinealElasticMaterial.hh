#ifndef FALL_N_MATERIAL_INSTANCE_HH
#define FALL_N_MATERIAL_INSTANCE_HH

#include <memory>

#include "ConstitutiveRelation.hh"
#include "MaterialState.hh"
#include "MaterialStatePolicy.hh"

// Specific relations — needed by convenience aliases at the bottom
#include "constitutive_models/lineal/IsotropicRelation.hh"
#include "constitutive_models/lineal/TimoshenkoBeamSection.hh"
#include "constitutive_models/lineal/MindlinShellSection.hh"
#include "constitutive_models/non_lineal/InelasticRelation.hh"

// =============================================================================
//  MaterialInstance<Relation, StatePolicy>
// =============================================================================
//
//  Universal material-instance template.  Wraps a constitutive relation
//  (shared, flyweight) together with per-point kinematic state (owned).
//
//  ─── Template parameters ────────────────────────────────────────────────
//
//    Relation     Constrained by ConstitutiveRelation (Level 1).
//                 Provides KinematicT, ConjugateT, TangentT, and
//                 MaterialPolicyT (the trait bag needed by the
//                 Material<> type-erasure layer).
//
//    StatePolicy  Storage strategy for the kinematic state variable.
//                 Defaults to ElasticState (current-value-only).
//                 Use MemoryState for path-dependent materials.
//
//  ─── Conditional API ────────────────────────────────────────────────────
//
//    The interface adapts at compile time according to the concept level
//    satisfied by Relation:
//
//      ConstitutiveRelation (Level 1 — always):
//        compute_response(k) → ConjugateT
//        tangent(k)          → TangentT
//
//      ElasticConstitutiveRelation (Level 2a — if satisfied):
//        C()       → TangentT        (constant tangent, no argument)
//        tangent() → TangentT        (     ""          ""         )
//
//      InelasticConstitutiveRelation (Level 2b — if satisfied):
//        update(k)        — evolve internal variables
//        internal_state() → const InternalVariablesT&
//
//  ─── Relation access ────────────────────────────────────────────────────
//
//    Instead of leaking ad-hoc wrappers (set_elasticity, etc.), the
//    underlying relation is accessible via constitutive_relation():
//
//      mat.constitutive_relation().update_elasticity(210.0, 0.25);
//      mat.constitutive_relation().update_section_stiffness(E, G, A, I, J);
//
//  ─── Universality ──────────────────────────────────────────────────────
//
//    Works uniformly for solids and structural elements:
//
//      MaterialInstance<ContinuumIsotropicRelation>       3D isotropic solid
//      MaterialInstance<ElasticRelation<UniaxialMaterial>> uniaxial bar
//      MaterialInstance<TimoshenkoBeamSection3D>          3D beam section
//      MaterialInstance<VonMisesRelation, MemoryState>    future plasticity
//
// =============================================================================

template <ConstitutiveRelation Relation, template<typename> class StatePolicy = ElasticState>
    requires MemoryPolicyFor<StatePolicy, typename Relation::KinematicT>
class MaterialInstance {

public:
    // --- Aliases derived from concepts (not from legacy inheritance) ----------
    //
    //  MaterialPolicy  — the trait bag (SolidMaterial<6>, BeamMaterial<6,3>, …)
    //                    Required by Material.hh type-erasure layer, which
    //                    accesses MaterialType::MaterialPolicy.
    //
    //  KinematicT      — kinematic measure     (Strain<6>, BeamGeneralizedStrain<6,3>, …)
    //  ConjugateT      — conjugate response    (Stress<6>, BeamSectionForces<6>, …)
    //  TangentT        — tangent operator       (Eigen::Matrix<double, n, n>)
    //  StateVariableT  — what is stored per point (≡ KinematicT for now)
    //

    using MaterialPolicy  = typename Relation::MaterialPolicyT;
    using KinematicT      = typename Relation::KinematicT;
    using ConjugateT      = typename Relation::ConjugateT;
    using TangentT        = typename Relation::TangentT;
    using StateVariableT  = KinematicT;

    static constexpr std::size_t dim            = KinematicT::dim;
    static constexpr std::size_t num_components = KinematicT::num_components;

private:
    // Per-point kinematic state.  ElasticState stores only the current value;
    // MemoryState accumulates a full loading history.
    MaterialState<StatePolicy, KinematicT> state_{};

    // Shared constitutive relation (flyweight pattern):
    // all integration points in the same material zone share E, ν, section
    // properties, etc.  Copies of MaterialInstance intentionally share
    // this pointer — that is by design.
    std::shared_ptr<Relation> relation_;

public:
    // ─── Universal interface (ConstitutiveRelation — Level 1) ────────────

    // Compute the conjugate response σ = f(ε) given a kinematic state.
    [[nodiscard]] ConjugateT compute_response(const KinematicT& k) const {
        return relation_->compute_response(k);
    }

    // Tangent operator ∂σ/∂ε evaluated at a given kinematic state.
    // For linear-elastic materials the argument is ignored, but the signature
    // remains general so that nonlinear materials work transparently.
    [[nodiscard]] TangentT tangent(const KinematicT& k) const {
        return relation_->tangent(k);
    }

    // ─── Elastic-only interface (Level 2a) ──────────────────────────────

    // Constant tangent operator (no kinematic argument).
    // Available only when Relation satisfies ElasticConstitutiveRelation.
    [[nodiscard]] TangentT C() const
        requires ElasticConstitutiveRelation<Relation>
    {
        return relation_->tangent();
    }

    [[nodiscard]] TangentT tangent() const
        requires ElasticConstitutiveRelation<Relation>
    {
        return relation_->tangent();
    }

    // ─── Inelastic-only interface (Level 2b) ─────────────────────────────

    // Evolve internal state variables given the current kinematic state.
    // This is the "return-mapping" update step:
    //   1. compute_response(k) delivers the trial response   (const)
    //   2. update(k) commits the irreversible state change   (non-const)
    void update(const KinematicT& k)
        requires InelasticConstitutiveRelation<Relation>
    {
        relation_->update(k);
        state_.update(k);   // record in history (MemoryState)
    }

    [[nodiscard]] const auto& internal_state() const
        requires InelasticConstitutiveRelation<Relation>
    {
        return relation_->internal_state();
    }

    // ─── Per-point state access ──────────────────────────────────────────

    [[nodiscard]] const KinematicT& current_state() const noexcept {
        return state_.current_value();
    }

    void update_state(const KinematicT& e) { state_.update(e); }
    void update_state(KinematicT&& e)      { state_.update(std::move(e)); }

    // ─── Constitutive relation access ────────────────────────────────────
    //
    //  Provides direct access to the underlying relation for parameter
    //  queries or updates that are specific to a particular relation type.
    //
    //    mat.constitutive_relation().update_elasticity(210.0, 0.25);
    //    mat.constitutive_relation().print_constitutive_parameters();
    //    mat.constitutive_relation().update_section_stiffness(E, G, A, Iy, Iz, J);
    //
    //  This replaces ad-hoc wrappers like set_elasticity(), which leaked
    //  relation-specific details into the material interface.

    [[nodiscard]] const Relation& constitutive_relation() const noexcept { return *relation_; }
    [[nodiscard]]       Relation& constitutive_relation()       noexcept { return *relation_; }

    [[nodiscard]] std::shared_ptr<const Relation> shared_relation() const noexcept {
        return relation_;
    }

    // ─── Constructors ────────────────────────────────────────────────────

    // Perfect-forwarding constructor: builds the Relation in-place.
    //   e.g. MaterialInstance<ContinuumIsotropicRelation>{200.0, 0.3}
    //        MaterialInstance<TimoshenkoBeamSection3D>{E, G, A, Iy, Iz, J}
    //
    // The constraint `constructible_from` both SFINAE-guards against
    // hijacking copy/move and documents the constructor contract.
    template <typename... Args>
        requires std::constructible_from<Relation, Args...>
    explicit MaterialInstance(Args&&... args)
        : relation_{std::make_shared<Relation>(std::forward<Args>(args)...)}
    {}

    // Construct from an existing shared relation (sharing across points).
    //   auto rel = std::make_shared<ContinuumIsotropicRelation>(200.0, 0.3);
    //   MaterialInstance<ContinuumIsotropicRelation> m1{rel};
    //   MaterialInstance<ContinuumIsotropicRelation> m2{rel}; // shares rel
    explicit MaterialInstance(std::shared_ptr<Relation> relation)
        : relation_{std::move(relation)}
    {}

    // ─── Special members (rule of zero) ──────────────────────────────────

    MaterialInstance()                                      = default;
    ~MaterialInstance()                                     = default;

    MaterialInstance(const MaterialInstance&)                = default;
    MaterialInstance(MaterialInstance&&) noexcept            = default;
    MaterialInstance& operator=(const MaterialInstance&)     = default;
    MaterialInstance& operator=(MaterialInstance&&) noexcept = default;
};


// =============================================================================
//  Convenience constrained aliases
// =============================================================================

// Elastic material: ElasticState storage, ElasticConstitutiveRelation required
template <ElasticConstitutiveRelation R>
using ElasticMaterial = MaterialInstance<R, ElasticState>;

// Inelastic material: MemoryState storage, InelasticConstitutiveRelation required
template <InelasticConstitutiveRelation R>
using InelasticMaterial = MaterialInstance<R, MemoryState>;


// =============================================================================
//  Named material aliases — Solids
// =============================================================================

using ContinuumIsotropicElasticMaterial = ElasticMaterial<ContinuumIsotropicRelation>;
using UniaxialIsotropicElasticMaterial  = ElasticMaterial<UniaxialIsotropicRelation>;


// =============================================================================
//  Named material aliases — Beams
// =============================================================================

using TimoshenkoBeamMaterial3D = ElasticMaterial<TimoshenkoBeamSection3D>;
using TimoshenkoBeamMaterial2D = ElasticMaterial<TimoshenkoBeamSection2D>;


// =============================================================================
//  Named material aliases — Shells
// =============================================================================

using MindlinShellMaterial = ElasticMaterial<MindlinShellSection>;


// =============================================================================
//  Named material aliases — Inelastic (J2 Plasticity)
// =============================================================================

using J2PlasticMaterial3D  = InelasticMaterial<J2PlasticityRelation<ThreeDimensionalMaterial>>;
using J2PlasticMaterial2D  = InelasticMaterial<J2PlasticityRelation<PlaneMaterial>>;
using J2PlasticMaterial1D  = InelasticMaterial<J2PlasticityRelation<UniaxialMaterial>>;


#endif // FALL_N_MATERIAL_INSTANCE_HH