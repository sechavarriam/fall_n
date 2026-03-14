#ifndef FALL_N_MATERIAL_INSTANCE_HH
#define FALL_N_MATERIAL_INSTANCE_HH

#include <memory>

#include "ConstitutiveRelation.hh"
#include "ConstitutiveState.hh"
#include "MaterialStatePolicy.hh"

// Specific relations — needed by convenience aliases at the bottom
#include "constitutive_models/lineal/IsotropicRelation.hh"
#include "constitutive_models/lineal/TimoshenkoBeamSection.hh"
#include "constitutive_models/lineal/MindlinShellSection.hh"
#include "constitutive_models/non_lineal/InelasticRelation.hh"

namespace constitutive_site_detail {

template <typename Relation, bool HasExternalAlgorithmicState =
    ExternallyStateDrivenConstitutiveRelation<Relation>>
struct AlgorithmicStateHolder {
};

template <typename Relation>
struct AlgorithmicStateHolder<Relation, true> {
    using AlgorithmicStateT = typename Relation::InternalVariablesT;
    AlgorithmicStateT algorithmic_state_{};

    [[nodiscard]] constexpr const AlgorithmicStateT& algorithmic_state() const noexcept {
        return algorithmic_state_;
    }

    [[nodiscard]] constexpr AlgorithmicStateT& algorithmic_state() noexcept {
        return algorithmic_state_;
    }
};

} // namespace constitutive_site_detail

// =============================================================================
//  MaterialInstance<Relation, StatePolicy> / ConstitutiveSite<...>
// =============================================================================
//
//  Universal stateful constitutive-site template.  It wraps:
//    1. a constitutive relation handle,
//    2. per-site kinematic storage,
//    3. optional access to internal variables for inelastic models.
//
//  The legacy class name `MaterialInstance` is kept because it is already used
//  widely across the library.  Semantically, however, this object already
//  lives at the constitutive-site level rather than at the immutable material
//  definition level.
//
//  ─── Template parameters ────────────────────────────────────────────────
//
//    Relation     Constrained by ConstitutiveRelation (Level 1).
//                 Provides KinematicT, ConjugateT, TangentT, and
//                 MaterialPolicyT (the trait bag needed by the
//                 Material<> type-erasure layer).
//
//    StatePolicy  Storage strategy for the kinematic state variable.
//                 Defaults to ElasticState / CommittedState
//                 (single committed/current value).
//                 Use HistoryState for unbounded explicit history, or
//                 CircularHistoryPolicy<N>::Policy for a fixed compile-time
//                 sliding window backed by a circular buffer.
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
//      MaterialInstance<VonMisesRelation, HistoryState> explicit unbounded history
//      MaterialInstance<VonMisesRelation,
//                       CircularHistoryPolicy<4>::Policy> fixed 4-step window
//
// =============================================================================

template <ConstitutiveRelation Relation, template<typename> class StatePolicy = ElasticState>
    requires StateStoragePolicyFor<StatePolicy, typename Relation::KinematicT>
class MaterialInstance
    : private constitutive_site_detail::AlgorithmicStateHolder<Relation> {

public:
    // --- Aliases derived from concepts (not from legacy inheritance) ----------
    //
    //  ConstitutiveSpace — the measure-space descriptor
    //                     (SolidMaterial<6>, BeamMaterial<6,3>, …)
    //                     Required by Material.hh type-erasure layer.
    //
    //  MaterialPolicy    — legacy alias retained for compatibility.
    //
    //  KinematicT      — kinematic measure     (Strain<6>, BeamGeneralizedStrain<6,3>, …)
    //  ConjugateT      — conjugate response    (Stress<6>, BeamSectionForces<6>, …)
    //  TangentT        — tangent operator       (Eigen::Matrix<double, n, n>)
    //  StateVariableT  — what is stored per point (≡ KinematicT for now)
    //

    using ConstitutiveSpace = typename Relation::MaterialPolicyT;
    using MaterialPolicy    = ConstitutiveSpace;
    using KinematicT        = typename Relation::KinematicT;
    using ConjugateT        = typename Relation::ConjugateT;
    using TangentT          = typename Relation::TangentT;
    using StateVariableT    = KinematicT;
    using ConstitutiveStateT = ConstitutiveState<StatePolicy, KinematicT>;
    using AlgorithmicStateHolderT =
        constitutive_site_detail::AlgorithmicStateHolder<Relation>;

    static constexpr std::size_t dim            = KinematicT::dim;
    static constexpr std::size_t num_components = KinematicT::num_components;

private:
    // Per-site kinematic constitutive state.  This semantic layer sits above
    // the low-level storage adapter and is the next staging point toward an
    // eventual explicit separation of law/state/integration algorithm.
    ConstitutiveStateT constitutive_state_{};

    // Relation handle.
    //
    // The relation may be shared when the user explicitly builds the site from
    // an external `std::shared_ptr<Relation>`.  Ordinary copy construction of
    // MaterialInstance deep-copies the relation to preserve constitutive-site
    // independence.  That distinction is important: explicit sharing is
    // allowed, implicit sharing by copy is not.
    std::shared_ptr<Relation> relation_;

public:
    // ─── Universal interface (ConstitutiveRelation — Level 1) ────────────

    // Compute the conjugate response σ = f(ε) given a kinematic state.
    [[nodiscard]] ConjugateT compute_response(const KinematicT& k) const {
        if constexpr (ExternallyStateDrivenConstitutiveRelation<Relation>) {
            return relation_->compute_response(k, this->algorithmic_state());
        } else {
            return relation_->compute_response(k);
        }
    }

    // Tangent operator ∂σ/∂ε evaluated at a given kinematic state.
    // For linear-elastic materials the argument is ignored, but the signature
    // remains general so that nonlinear materials work transparently.
    [[nodiscard]] TangentT tangent(const KinematicT& k) const {
        if constexpr (ExternallyStateDrivenConstitutiveRelation<Relation>) {
            return relation_->tangent(k, this->algorithmic_state());
        } else {
            return relation_->tangent(k);
        }
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
        if constexpr (ExternallyStateDrivenConstitutiveRelation<Relation>) {
            relation_->commit(this->algorithmic_state(), k);
        } else {
            relation_->update(k);
        }
        constitutive_state_.update(k);
        if constexpr (requires { constitutive_state_.commit_trial(); }) {
            constitutive_state_.commit_trial();
        }
    }

    [[nodiscard]] const auto& internal_state() const
        requires InelasticConstitutiveRelation<Relation>
    {
        if constexpr (ExternallyStateDrivenConstitutiveRelation<Relation>) {
            return this->algorithmic_state();
        } else {
            return relation_->internal_state();
        }
    }

    [[nodiscard]] const auto& algorithmic_state() const noexcept
        requires ExternallyStateDrivenConstitutiveRelation<Relation>
    {
        return AlgorithmicStateHolderT::algorithmic_state();
    }

    [[nodiscard]] auto& algorithmic_state() noexcept
        requires ExternallyStateDrivenConstitutiveRelation<Relation>
    {
        return AlgorithmicStateHolderT::algorithmic_state();
    }

    // ─── Per-point state access ──────────────────────────────────────────

    [[nodiscard]] const KinematicT& current_state() const noexcept {
        return constitutive_state_.current_value();
    }

    [[nodiscard]] const ConstitutiveStateT& constitutive_state() const noexcept {
        return constitutive_state_;
    }

    [[nodiscard]] ConstitutiveStateT& constitutive_state() noexcept {
        return constitutive_state_;
    }

    void update_state(const KinematicT& e) {
        constitutive_state_.update(e);
        if constexpr (requires { constitutive_state_.commit_trial(); }) {
            constitutive_state_.commit_trial();
        }
    }
    void update_state(KinematicT&& e) {
        constitutive_state_.update(std::move(e));
        if constexpr (requires { constitutive_state_.commit_trial(); }) {
            constitutive_state_.commit_trial();
        }
    }

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

    [[nodiscard]] std::shared_ptr<const Relation> shared_relation_handle() const noexcept {
        return relation_;
    }

    // ─── Constructors ────────────────────────────────────────────────────

    // Perfect-forwarding constructor: builds a private relation in-place for
    // this constitutive site.
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

    // Construct from an existing shared relation handle.
    // This is the explicit opt-in path for sharing a read-mostly relation
    // object across several constitutive sites.
    //   auto rel = std::make_shared<ContinuumIsotropicRelation>(200.0, 0.3);
    //   MaterialInstance<ContinuumIsotropicRelation> m1{rel};
    //   MaterialInstance<ContinuumIsotropicRelation> m2{rel}; // shares rel
    explicit MaterialInstance(std::shared_ptr<Relation> relation)
        : relation_{std::move(relation)}
    {}

    // ─── Special members ─────────────────────────────────────────────────
    //
    //  Copying deep-clones the relation so that the new constitutive site owns
    //  an independent runtime object.  This keeps copy semantics semantically
    //  local and avoids surprising cross-site coupling through implicit
    //  sharing.  Shared relations remain available through the explicit
    //  `std::shared_ptr<Relation>` constructor above.

    MaterialInstance()                                      = default;
    ~MaterialInstance()                                     = default;

    MaterialInstance(const MaterialInstance& other)
        : AlgorithmicStateHolderT(static_cast<const AlgorithmicStateHolderT&>(other))
        , constitutive_state_(other.constitutive_state_)
        , relation_(other.relation_
            ? std::make_shared<Relation>(*other.relation_)
            : nullptr)
    {}

    MaterialInstance(MaterialInstance&&) noexcept            = default;

    MaterialInstance& operator=(const MaterialInstance& other) {
        if (this != &other) {
            static_cast<AlgorithmicStateHolderT&>(*this) =
                static_cast<const AlgorithmicStateHolderT&>(other);
            constitutive_state_ = other.constitutive_state_;
            relation_ = other.relation_
                ? std::make_shared<Relation>(*other.relation_)
                : nullptr;
        }
        return *this;
    }

    MaterialInstance& operator=(MaterialInstance&&) noexcept = default;
};


// =============================================================================
//  Convenience constrained aliases
// =============================================================================

template <ConstitutiveRelation R, template<typename> class StatePolicy = ElasticState>
using ConstitutiveSite = MaterialInstance<R, StatePolicy>;

template <ElasticConstitutiveRelation R>
using ElasticConstitutiveSite = ConstitutiveSite<R, CommittedState>;

template <InelasticConstitutiveRelation R>
using InelasticConstitutiveSite = ConstitutiveSite<R, CommittedState>;

template <InelasticConstitutiveRelation R>
using HistoryTrackingConstitutiveSite = ConstitutiveSite<R, HistoryState>;

template <InelasticConstitutiveRelation R, std::size_t Capacity>
using CircularHistoryConstitutiveSite =
    MaterialInstance<R, CircularHistoryPolicy<Capacity>::template Policy>;

template <InelasticConstitutiveRelation R>
using TrialConstitutiveSite = ConstitutiveSite<R, TrialCommittedState>;

// Legacy aliases kept for backward compatibility.
// Elastic material: single committed/current state
template <ElasticConstitutiveRelation R>
using ElasticMaterial = ElasticConstitutiveSite<R>;

// Inelastic material: by default only the committed/current state is stored.
template <InelasticConstitutiveRelation R>
using InelasticMaterial = InelasticConstitutiveSite<R>;

template <InelasticConstitutiveRelation R>
using HistoryTrackedInelasticMaterial = HistoryTrackingConstitutiveSite<R>;

template <InelasticConstitutiveRelation R, std::size_t Capacity>
using CircularHistoryInelasticMaterial = CircularHistoryConstitutiveSite<R, Capacity>;


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
