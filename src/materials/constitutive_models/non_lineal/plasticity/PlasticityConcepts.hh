#ifndef FN_PLASTICITY_CONCEPTS_HH
#define FN_PLASTICITY_CONCEPTS_HH

#include <concepts>
#include <cstddef>
#include <cmath>

#include <Eigen/Dense>

// =============================================================================
//  TrialState<N> — Elastic predictor result
// =============================================================================
//
//  Gathers the quantities produced by the elastic trial step before yield
//  checking.  Consumed by YieldCriterion, FlowRule, and the consistent
//  tangent computation.
//
//  N = number of Voigt components (1, 3, or 6).
//
//  The deviatoric decomposition and Voigt-weighted norm are computed once
//  by the PlasticityRelation and cached here so that yield criteria and
//  flow rules do not re-derive them.
//
// =============================================================================

template <std::size_t N>
struct TrialState {
    Eigen::Vector<double, N> stress;              // σ_trial = C_e · (ε − ε^p)
    Eigen::Vector<double, N> deviatoric;          // s_trial = dev(σ_trial)
    double                   deviatoric_norm = 0; // ‖s_trial‖  (Voigt-weighted)
    double                   hydrostatic     = 0; // p = tr(σ_trial) / 3
};

// =============================================================================
//  PlasticInternalVariables<N, HardeningStateT>
// =============================================================================
//
//  Composes the plastic strain tensor with the hardening law's state.
//  This is the α vector in standard return-mapping formulations:
//
//      α = { ε^p , hardening_state }
//
//  The HardeningStateT is opaque here; its shape depends on the hardening
//  law chosen by the user:
//
//    ┌─────────────────────────────────┬───────────────────────────────────┐
//    │  HardeningStateT                │  Physical meaning                 │
//    ├─────────────────────────────────┼───────────────────────────────────┤
//    │  IsotropicHardeningState        │  scalar ε̄^p  (yield surface       │
//    │                                 │  expands uniformly)               │
//    ├─────────────────────────────────┼───────────────────────────────────┤
//    │  KinematicHardeningState<N>     │  backstress tensor β (yield       │
//    │  (future)                       │  surface translates — Bauschinger │
//    │                                 │  effect for hysteretic models)    │
//    ├─────────────────────────────────┼───────────────────────────────────┤
//    │  MixedHardeningState<N>         │  {ε̄^p, β}  (both isotropic        │
//    │  (future)                       │  expansion + kinematic shift)     │
//    ├─────────────────────────────────┼───────────────────────────────────┤
//    │  DamageHardeningState<N>        │  {ε̄^p, d, β}  (hysteretic with    │
//    │  (future)                       │  stiffness/strength degradation)  │
//    └─────────────────────────────────┴───────────────────────────────────┘
//
// =============================================================================

template <std::size_t N, typename HardeningStateT>
struct PlasticInternalVariables {
    static constexpr std::size_t num_components = N;

    Eigen::Vector<double, N> plastic_strain = Eigen::Vector<double, N>::Zero();
    HardeningStateT          hardening_state{};

    // ─── Named accessors ─────────────────────────────────────────────────

    [[nodiscard]] const auto& eps_p() const noexcept { return plastic_strain; }

    // Backward-compat: equivalent plastic strain (when state carries one)
    [[nodiscard]] double eps_bar_p() const noexcept
        requires requires(const HardeningStateT& h) { h.equivalent_plastic_strain; }
    {
        return hardening_state.equivalent_plastic_strain;
    }

    // ─── Rule of five (all defaulted) ────────────────────────────────────

    PlasticInternalVariables()                                              = default;
    ~PlasticInternalVariables()                                             = default;
    PlasticInternalVariables(const PlasticInternalVariables&)               = default;
    PlasticInternalVariables(PlasticInternalVariables&&) noexcept           = default;
    PlasticInternalVariables& operator=(const PlasticInternalVariables&)    = default;
    PlasticInternalVariables& operator=(PlasticInternalVariables&&) noexcept = default;
};


// =============================================================================
//  YieldCriterion<Y, N> — Concept
// =============================================================================
//
//  Defines a yield surface geometry in stress space.
//
//    equivalent_stress(trial) → scalar stress measure
//        e.g. σ_eq = √(3J₂) for von Mises
//             q = α·I₁ + √J₂  for Drucker-Prager
//
//    gradient(trial) → ∂f/∂σ ∈ ℝ^N  (unit normal, Voigt notation)
//        Used as the associated flow direction and in the consistent tangent.
//
//  The yield function value  f = q − σ_y  is computed externally by the
//  PlasticityRelation, where σ_y comes from the HardeningLaw.
//
//  The YieldCriterion is stateless for simple criteria (von Mises, Tresca).
//  Pressure-dependent criteria (Drucker-Prager, Mohr-Coulomb) will store
//  material-level parameters (friction angle, cohesion).
//
// =============================================================================

template <typename Y, std::size_t N>
concept YieldCriterion = requires(const Y& y, const TrialState<N>& trial) {
    { y.equivalent_stress(trial) } -> std::convertible_to<double>;
    { y.gradient(trial)          } -> std::same_as<Eigen::Vector<double, N>>;
};


// =============================================================================
//  HardeningLaw<H> — Concept
// =============================================================================
//
//  Governs the evolution of the yield surface size (isotropic), position
//  (kinematic), or both (mixed).
//
//    StateT                     — type representing the hardening state
//    yield_stress(state)        → current σ_y(α)
//    modulus(state)             → H_iso = dσ_y/d(ε̄^p)
//    evolve(state, Δγ)          → updated state (functional, no mutation)
//
//  ─── Functional style ──────────────────────────────────────────────────
//
//  evolve() returns a NEW state rather than mutating in-place.  This keeps
//  the return-mapping algorithm clean: a tentative update is computed and
//  only committed after global Newton convergence.
//
//  ─── Extensibility for hysteretic / dynamic models ─────────────────────
//
//  Future hardening laws will add richer StateT types:
//
//    KinematicHardening (Armstrong-Frederick, Chaboche):
//      StateT carries backstress β.  The PlasticityRelation shifts the
//      effective deviatoric stress  s_eff = s − β  before evaluating the
//      yield criterion → Bauschinger effect for cyclic loading.
//
//    NonlinearIsotropicHardening (Voce, Ramberg-Osgood):
//      StateT remains scalar, but yield_stress() is nonlinear in ε̄^p.
//      Defines the monotonic backbone curve.
//
//    DamageHardening:
//      StateT augmented with damage variable d ∈ [0,1].
//      yield_stress() → (1−d)·σ_y  (degraded yield).
//      Interacts with backbone curves for hysteretic models.
//
// =============================================================================

template <typename H>
concept HardeningLaw = requires {
    typename H::StateT;
} && requires(const H& h, const typename H::StateT& state, double delta_gamma) {
    { h.yield_stress(state)        } -> std::convertible_to<double>;
    { h.modulus(state)             } -> std::convertible_to<double>;
    { h.evolve(state, delta_gamma) } -> std::same_as<typename H::StateT>;
} &&
    std::default_initializable<typename H::StateT> &&
    std::copyable<typename H::StateT>;


// =============================================================================
//  FlowRule<F, N, Y> — Concept
// =============================================================================
//
//  Determines the direction of plastic strain increment:
//
//    direction(yield, trial) → m̂ ∈ ℝ^N  (unit flow vector, Voigt notation)
//
//  The YieldCriterion is passed as argument (not stored by reference) so
//  that associated flow can delegate to yield.gradient() without lifetime
//  issues, and non-associated flow can ignore it.
//
//  ─── Associated plasticity ─────────────────────────────────────────────
//
//    m̂ = ∂f/∂σ   (plastic potential g ≡ yield function f)
//    → Symmetric consistent tangent, maximum dissipation, Drucker stability.
//
//  ─── Non-associated plasticity (future) ────────────────────────────────
//
//    m̂ = ∂g/∂σ   where g(σ) ≠ f(σ)
//    → Drucker-Prager dilation, non-symmetric tangent.
//    Implemented as NonAssociatedFlow<PotentialF>, which stores its own
//    potential function and ignores the yield argument.
//
// =============================================================================

template <typename F, std::size_t N, typename Y>
concept FlowRule = requires(
    const F& f,
    const Y& yield,
    const TrialState<N>& trial)
{
    { f.direction(yield, trial) } -> std::same_as<Eigen::Vector<double, N>>;
};


#endif // FN_PLASTICITY_CONCEPTS_HH
