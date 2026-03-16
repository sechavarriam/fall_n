#ifndef FN_CONSTITUTIVE_RELATION
#define FN_CONSTITUTIVE_RELATION

#include <concepts>
#include <cstddef>
#include <type_traits>

#include <Eigen/Dense>

// =============================================================================
//  LEVEL 0 — Fundamental measure concepts
// =============================================================================
//
//  En mecánica de medios continuos y estructural, toda formulación variacional
//  parte de un par de variables conjugadas energéticamente:
//
//      δW_int = ∫ σ : δε  dV        (sólidos 3D)
//      δW_int = ∫ (M·δκ + V·δγ) dx  (vigas)
//      δW_int = ∫ (N·δε + M·δκ) dx  (vigas Euler-Bernoulli)
//
//  La "causa" cinemática (ε, κ, γ, ...) es la KinematicMeasure.
//  El "efecto" tensional conjugado (σ, M, V, N, ...) es el TensionalConjugate.
//
//  Ambos conceptos exigen la misma interfaz mínima: un vector de componentes
//  indexable y un tamaño conocido en tiempo de compilación. Esto permite que
//  la relación constitutiva opere genéricamente sobre cualquier par, sin
//  conocer si son tensores de Voigt, fuerzas generalizadas de viga, etc.
//
// -----------------------------------------------------------------------------

// KinematicMeasure: the "cause" in the constitutive relation.
//
// Any type K that models a kinematic measure must provide:
//   - K::num_components  — compile-time size (number of independent components)
//   - K::dim             — spatial dimension associated with the measure
//   - k[i]              — read access to the i-th component (returns double)
//   - k.components()    — read access to the full component vector
//                          (the return type is intentionally unconstrained:
//                           it can be Eigen::Ref, a span, a raw double, etc.)
//   - K must be default-constructible and copyable so it can be stored in
//     state containers (MaterialState, vectors, etc.)
//
// Current types that satisfy this concept:
//   Strain<1>, Strain<3>, Strain<6>   (via VoigtVector<N>)
//
// Future types that will satisfy it:
//   BeamCurvature<N>, BeamDeformation<N>, ShellStrain<N>, ...

template <typename K>
concept KinematicMeasure = requires {
    // Compile-time constants
    { K::num_components } -> std::convertible_to<std::size_t>;
    { K::dim }            -> std::convertible_to<std::size_t>;
} && requires(const K k, std::size_t i) {
    // Component access
    { k[i]            } -> std::convertible_to<double>;
    { k.components()  };  // unconstrained return type (Eigen::Ref, double, span...)
} &&
    // Must be value-semantic (storable in containers and state policies)
    std::default_initializable<K> &&
    std::copyable<K>;


// TensionalConjugate: the "effect" in the constitutive relation.
//
// Any type T that models a tensional conjugate must provide:
//   - T::num_components  — compile-time size (must match its kinematic pair)
//   - t[i]              — read access to the i-th component
//   - t.components()    — read access to the full component vector
//   - set_components(v) — write the full component vector from an expression
//                          (Eigen expression, scalar, etc.)
//   - T must be default-constructible and copyable.
//
// The additional requirement `set_components` exists because the constitutive
// relation needs to *produce* the conjugate (σ = C·ε). The kinematic measure,
// on the other hand, is an *input* — it only needs to be read.
//
// Current types that satisfy this concept:
//   Stress<1>, Stress<3>, Stress<6>   (via VoigtVector<N>)
//
// Future types that will satisfy it:
//   BeamInternalForces<N>, ShellResultants<N>, ...

template <typename T>
concept TensionalConjugate = requires {
    { T::num_components } -> std::convertible_to<std::size_t>;
} && requires(const T t, std::size_t i) {
    { t[i]           } -> std::convertible_to<double>;
    { t.components() };
} &&
    std::default_initializable<T> &&
    std::copyable<T>;


// ConjugatePair: binds a kinematic measure with its tensional conjugate.
//
// The fundamental constraint is dimensional consistency: both must have the
// same number of independent components. This guarantees that the constitutive
// matrix C is square (n × n) and the inner product σ·δε is well-defined.
//
// Example:
//   ConjugatePair<Strain<6>, Stress<6>>  ✓  (6 == 6)
//   ConjugatePair<Strain<3>, Stress<6>>  ✗  (3 != 6)

template <typename K, typename T>
concept ConjugatePair =
    KinematicMeasure<K> &&
    TensionalConjugate<T> &&
    (K::num_components == T::num_components);


// =============================================================================
//  LEVEL 1 — ConstitutiveRelation concept
// =============================================================================
//
//  This is the universal contract for ANY constitutive relation, whether
//  linear elastic, hyperelastic, elastoplastic, viscoelastic, damage, etc.
//
//  Every constitutive relation R must:
//
//    1. Declare its conjugate pair via nested type aliases:
//         R::KinematicT    — the kinematic measure type (e.g. Strain<6>)
//         R::ConjugateT    — the tensional conjugate type (e.g. Stress<6>)
//         R::TangentT      — the tangent operator type (typically Eigen::Matrix)
//
//    2. The pair (KinematicT, ConjugateT) must satisfy ConjugatePair.
//
//    3. Provide two operations on a const instance given a kinematic state k:
//         r.compute_response(k) → ConjugateT   (computes σ given ε)
//         r.tangent(k)          → TangentT      (computes ∂σ/∂ε at ε)
//
//  Note: the tangent takes the kinematic state as argument because, in
//  general (nonlinear materials), C_t = ∂σ/∂ε depends on the current strain.
//  For linear elastic materials, the argument is simply ignored.
//
//  Note: we require these on `const R` because computing a response should
//  NOT mutate the relation's parameters. State evolution (for inelastic
//  materials) is handled by a separate `update` step, not by `compute_response`.
//
// -----------------------------------------------------------------------------

template <typename R>
concept ConstitutiveRelation = requires {
    typename R::KinematicT;
    typename R::ConjugateT;
    typename R::TangentT;
} &&
    ConjugatePair<typename R::KinematicT, typename R::ConjugateT> &&
    requires(const R r, const typename R::KinematicT& k) {
        { r.compute_response(k) } -> std::same_as<typename R::ConjugateT>;
        { r.tangent(k)          } -> std::same_as<typename R::TangentT>;
    };


// =============================================================================
//  LEVEL 2a — ElasticConstitutiveRelation (refinement)
// =============================================================================
//
//  An elastic relation is path-independent: the response depends ONLY on the
//  current kinematic state, never on the loading history.
//
//  Consequence: the tangent operator can be evaluated WITHOUT a kinematic
//  argument (it is either constant, or depends only on material parameters).
//  This is the key semantic distinction vs. inelastic relations.
//
//  This additional overload `r.tangent()` (no argument) enables optimizations:
//  — the stiffness matrix can be assembled once and reused
//  — no need to store/track state history
//
// -----------------------------------------------------------------------------

template <typename R>
concept ElasticConstitutiveRelation =
    ConstitutiveRelation<R> &&
    requires(const R r) {
        { r.tangent() } -> std::same_as<typename R::TangentT>;
    };


// =============================================================================
//  LEVEL 2b — InelasticConstitutiveRelation (refinement, for future use)
// =============================================================================
//
//  An inelastic relation is path-dependent: the response depends on the
//  current kinematic state AND on internal (history) variables α.
//
//  Additional requirements beyond ConstitutiveRelation:
//    - R::InternalVariablesT  — type of the internal state (α)
//    - r.update(k)            — evolve internal variables given a load step
//    - r.internal_state()     — read access to current α
//
//  The separation between compute_response() (const, no mutation) and
//  update() (non-const, mutates α) follows the standard return-mapping
//  algorithm pattern:
//    1. Trial state:   σ_trial = compute_response(ε)    [const]
//    2. Check yield:   if f(σ_trial, α) > 0 → plastic correction
//    3. Update state:  update(ε)                        [mutates α]
//
// -----------------------------------------------------------------------------

template <typename R>
concept InelasticConstitutiveRelation =
    ConstitutiveRelation<R> &&
    requires {
        typename R::InternalVariablesT;
    } &&
    requires(R r, const R cr, const typename R::KinematicT& k) {
        { r.update(k) };
        { cr.internal_state() } -> std::same_as<const typename R::InternalVariablesT&>;
    };


// =============================================================================
//  LEVEL 3 — ExternallyStateDrivenConstitutiveRelation
// =============================================================================
//
//  Transitional concept for inelastic relations that can be evaluated and
//  committed against an explicit external algorithmic-state object instead of
//  mutating only an internally owned state.
//
//  This is the first architectural step toward a full split:
//
//      ConstitutiveRelation + ConstitutiveState + ConstitutiveIntegrator
//
//  Required interface:
//    - compute_response(k, alpha)  -> ConjugateT
//    - tangent(k, alpha)           -> TangentT
//    - commit(alpha, k)            -> mutates alpha
//
//  The legacy embedded-state API is still retained via
//  InelasticConstitutiveRelation.
//
// -----------------------------------------------------------------------------

template <typename R>
concept ExternallyStateDrivenConstitutiveRelation =
    InelasticConstitutiveRelation<R> &&
    requires(const R cr, typename R::InternalVariablesT& alpha,
             const typename R::InternalVariablesT& calpha,
             const typename R::KinematicT& k) {
        { cr.compute_response(k, calpha) } -> std::same_as<typename R::ConjugateT>;
        { cr.tangent(k, calpha) } -> std::same_as<typename R::TangentT>;
        { cr.commit(alpha, k) };
    };


// =============================================================================
//  Convenience alias: standard tangent matrix type for a conjugate pair
// =============================================================================
//
//  Given a ConjugatePair<K,T>, the tangent operator C is an n×n matrix where
//  n = K::num_components = T::num_components.
//
//  This alias saves boilerplate when defining new constitutive relations:
//    using TangentT = TangentMatrix<Strain<6>, Stress<6>>;
//    // equivalent to Eigen::Matrix<double, 6, 6>

template <typename K, typename T> requires ConjugatePair<K, T>
using TangentMatrix = Eigen::Matrix<double, K::num_components, T::num_components>;


// =============================================================================
//  State separation utility — Phase 6
// =============================================================================
//
//  Deficiency #4: StateVariableT ≡ KinematicT in MaterialInstance —
//  the constitutive "state" stored per integration point is just the last
//  kinematic measure, not the actual internal variables (plastic strain,
//  damage parameter, back-stress, etc.).
//
//  Solution: introduce a compile-time trait that extracts the separated
//  internal-state type from a constitutive relation.  For elastic relations
//  (which have no internal state), the trait resolves to an empty struct.
//  For inelastic relations it resolves to R::InternalVariablesT.
//
//  This is backward-compatible: no existing code changes behavior.
//
// -----------------------------------------------------------------------------

/// Empty placeholder for relations with no internal state.
struct NoInternalState {};

/// Primary template: elastic relations have no internal state.
template <typename R, typename = void>
struct InternalStateOf {
    using type = NoInternalState;
};

/// Specialization: inelastic relations expose InternalVariablesT.
template <typename R>
struct InternalStateOf<R, std::void_t<typename R::InternalVariablesT>> {
    using type = typename R::InternalVariablesT;
};

/// Convenience alias.
template <typename R>
using InternalStateOf_t = typename InternalStateOf<R>::type;

/// Concept: does R carry separated internal state?
template <typename R>
concept HasSeparatedInternalState =
    !std::same_as<InternalStateOf_t<R>, NoInternalState>;


#endif // FN_CONSTITUTIVE_RELATION
