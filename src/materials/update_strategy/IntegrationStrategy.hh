#ifndef FN_INTEGRATION_STRATEGY_HH
#define FN_INTEGRATION_STRATEGY_HH

#include <concepts>

// =============================================================================
//  IntegrationStrategy — concepts and concrete algorithm classes
// =============================================================================
//
//  An IntegrationStrategy mediates ALL constitutive calls through the
//  type-erasure layer (Material<Policy>), decoupling the physical material
//  model from the computational algorithm used to integrate it.
//
//  Contract:
//    compute_response(material, ε) → σ      [const — evaluate stress]
//    tangent(material, ε)          → C_t    [const — evaluate tangent]
//    commit(material, ε)                    [non-const — commit state]
//
//  Concrete strategies:
//    ElasticUpdate    — trivial pass-through for path-independent materials
//    InelasticUpdate  — delegates to the material's built-in integration
//
//  Future strategies (to be implemented):
//    ReturnMapping    — generic backward-Euler return mapping
//    CuttingPlane     — semi-explicit cutting-plane algorithm
//    Substepping<S>   — sub-incremental decorator wrapping any strategy
//
// =============================================================================

/// Concept documenting the IntegrationStrategy interface.
/// Checked implicitly at OwningMaterialModel instantiation time.
template <typename S, typename M>
concept IntegrationStrategyConcept = requires(
    S& s, const S& cs,
    M& model, const M& cmodel,
    const typename M::KinematicT& k)
{
    { cs.compute_response(cmodel, k) } -> std::same_as<typename M::ConjugateT>;
    { cs.tangent(cmodel, k)          } -> std::same_as<typename M::TangentT>;
    { s.commit(model, k)             };
};


// ─── ElasticUpdate ───────────────────────────────────────────────────────────
//
//  Trivial pass-through for path-independent (elastic) materials.
//  compute_response and tangent delegate directly to the material.
//  commit is a no-op since there is no internal state to evolve.
//
// ─────────────────────────────────────────────────────────────────────────────

struct ElasticUpdate {

    template <typename MaterialType>
    [[nodiscard]] auto compute_response(
        const MaterialType& mat,
        const typename MaterialType::KinematicT& k) const
        -> typename MaterialType::ConjugateT
    {
        return mat.compute_response(k);
    }

    template <typename MaterialType>
    [[nodiscard]] auto tangent(
        const MaterialType& mat,
        const typename MaterialType::KinematicT& k) const
        -> typename MaterialType::TangentT
    {
        return mat.tangent(k);
    }

    template <typename MaterialType>
    void commit(MaterialType&, const typename MaterialType::KinematicT&) const {
        // No-op: elastic materials have no internal state to commit.
    }
};


// ─── InelasticUpdate ─────────────────────────────────────────────────────────
//
//  Delegates to the material's own integration algorithm.
//
//  The material's compute_response(ε) and tangent(ε) already perform the
//  return-mapping (or whatever algorithm is built in). commit() calls
//  update(ε) to evolve internal variables after global convergence.
//
//  This is a pragmatic bridge: future strategies (ReturnMapping<>,
//  CuttingPlane<>) will extract primitives (yield_function, flow_direction,
//  hardening_modulus) from the material model and implement their own
//  integration loop, allowing the SAME physical model to be integrated
//  with DIFFERENT algorithms.
//
// ─────────────────────────────────────────────────────────────────────────────

struct InelasticUpdate {

    template <typename MaterialType>
    [[nodiscard]] auto compute_response(
        const MaterialType& mat,
        const typename MaterialType::KinematicT& k) const
        -> typename MaterialType::ConjugateT
    {
        return mat.compute_response(k);
    }

    template <typename MaterialType>
    [[nodiscard]] auto tangent(
        const MaterialType& mat,
        const typename MaterialType::KinematicT& k) const
        -> typename MaterialType::TangentT
    {
        return mat.tangent(k);
    }

    template <typename MaterialType>
    void commit(MaterialType& mat, const typename MaterialType::KinematicT& k) {
        mat.update(k);  // evolve internal variables (return-mapping + state history)
    }
};


#endif // FN_INTEGRATION_STRATEGY_HH
