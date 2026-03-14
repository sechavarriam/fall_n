#ifndef FALL_N_CONSTITUTIVE_INTEGRATOR_HH
#define FALL_N_CONSTITUTIVE_INTEGRATOR_HH

#include <concepts>

// =============================================================================
//  ConstitutiveIntegrator — local integration algorithm at a constitutive site
// =============================================================================
//
//  A ConstitutiveIntegrator is the algorithmic policy injected at the
//  Material<> / ConstitutiveHandle<> boundary.  It acts on a constitutive site
//  (relation + constitutive state) and is therefore the architectural bridge
//  toward a full split:
//
//      ConstitutiveLaw  +  ConstitutiveState  +  ConstitutiveIntegrator
//
//  The current constitutive relations still embed most of their own local
//  integration logic.  This header makes that dependency explicit and keeps the
//  injection point stable so that future return-mapping / substepping /
//  cutting-plane algorithms can be introduced without changing the type-erased
//  Material<> API.
//
//  Contract:
//    compute_response(site, k) -> conjugate response
//    tangent(site, k)          -> algorithmic tangent
//    commit(site, k)           -> evolve local state after convergence
//
// =============================================================================

template <typename Integrator, typename ConstitutiveSiteT>
concept ConstitutiveIntegratorConcept = requires(
    Integrator& integrator, const Integrator& cintegrator,
    ConstitutiveSiteT& site, const ConstitutiveSiteT& csite,
    const typename ConstitutiveSiteT::KinematicT& k)
{
    { cintegrator.compute_response(csite, k) }
        -> std::same_as<typename ConstitutiveSiteT::ConjugateT>;
    { cintegrator.tangent(csite, k) }
        -> std::same_as<typename ConstitutiveSiteT::TangentT>;
    { integrator.commit(site, k) };
};

namespace constitutive_integrators {

template <typename ConstitutiveSiteT>
inline void commit_constitutive_state(
    ConstitutiveSiteT& site,
    const typename ConstitutiveSiteT::KinematicT& k)
{
    site.constitutive_state().update(k);

    if constexpr (requires { site.constitutive_state().commit_trial(); }) {
        site.constitutive_state().commit_trial();
    }
}

} // namespace constitutive_integrators

// ─────────────────────────────────────────────────────────────────────────────
// PassThroughIntegrator
// ─────────────────────────────────────────────────────────────────────────────
//
// Path-independent local evaluation.  The constitutive response and tangent are
// read directly from the constitutive relation; commit is intentionally a no-op
// for backward compatibility with the existing elastic path.
//
// ─────────────────────────────────────────────────────────────────────────────

struct PassThroughIntegrator {

    template <typename ConstitutiveSiteT>
    [[nodiscard]] auto compute_response(
        const ConstitutiveSiteT& site,
        const typename ConstitutiveSiteT::KinematicT& k) const
        -> typename ConstitutiveSiteT::ConjugateT
    {
        if constexpr (requires {
            site.algorithmic_state();
            site.constitutive_relation().compute_response(k, site.algorithmic_state());
        }) {
            return site.constitutive_relation().compute_response(k, site.algorithmic_state());
        } else {
            return site.constitutive_relation().compute_response(k);
        }
    }

    template <typename ConstitutiveSiteT>
    [[nodiscard]] auto tangent(
        const ConstitutiveSiteT& site,
        const typename ConstitutiveSiteT::KinematicT& k) const
        -> typename ConstitutiveSiteT::TangentT
    {
        if constexpr (requires {
            site.algorithmic_state();
            site.constitutive_relation().tangent(k, site.algorithmic_state());
        }) {
            return site.constitutive_relation().tangent(k, site.algorithmic_state());
        } else {
            return site.constitutive_relation().tangent(k);
        }
    }

    template <typename ConstitutiveSiteT>
    void commit(
        ConstitutiveSiteT&,
        const typename ConstitutiveSiteT::KinematicT&) const
    {
        // No-op by design: preserves the legacy elastic workflow.
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// EmbeddedRelationIntegrator
// ─────────────────────────────────────────────────────────────────────────────
//
// Compatibility bridge for the current inelastic path.  The constitutive law
// still owns its local integration/update routine; this integrator simply makes
// that fact explicit and coordinates it with the external constitutive-state
// carrier.
//
// ─────────────────────────────────────────────────────────────────────────────

struct EmbeddedRelationIntegrator {

    template <typename ConstitutiveSiteT>
    [[nodiscard]] auto compute_response(
        const ConstitutiveSiteT& site,
        const typename ConstitutiveSiteT::KinematicT& k) const
        -> typename ConstitutiveSiteT::ConjugateT
    {
        if constexpr (requires {
            site.algorithmic_state();
            site.constitutive_relation().compute_response(k, site.algorithmic_state());
        }) {
            return site.constitutive_relation().compute_response(k, site.algorithmic_state());
        } else {
            return site.constitutive_relation().compute_response(k);
        }
    }

    template <typename ConstitutiveSiteT>
    [[nodiscard]] auto tangent(
        const ConstitutiveSiteT& site,
        const typename ConstitutiveSiteT::KinematicT& k) const
        -> typename ConstitutiveSiteT::TangentT
    {
        if constexpr (requires {
            site.algorithmic_state();
            site.constitutive_relation().tangent(k, site.algorithmic_state());
        }) {
            return site.constitutive_relation().tangent(k, site.algorithmic_state());
        } else {
            return site.constitutive_relation().tangent(k);
        }
    }

    template <typename ConstitutiveSiteT>
    void commit(
        ConstitutiveSiteT& site,
        const typename ConstitutiveSiteT::KinematicT& k) const
    {
        if constexpr (requires {
            site.algorithmic_state();
            site.constitutive_relation().commit(site.algorithmic_state(), k);
        }) {
            site.constitutive_relation().commit(site.algorithmic_state(), k);
        } else if constexpr (requires { site.constitutive_relation().update(k); }) {
            site.constitutive_relation().update(k);
        }

        constitutive_integrators::commit_constitutive_state(site, k);
    }
};

// Semantic names
using ElasticConstitutiveIntegrator = PassThroughIntegrator;
using EmbeddedInelasticConstitutiveIntegrator = EmbeddedRelationIntegrator;

// Legacy aliases retained for compatibility across the codebase.
using ElasticUpdate = PassThroughIntegrator;
using InelasticUpdate = EmbeddedRelationIntegrator;

#endif // FALL_N_CONSTITUTIVE_INTEGRATOR_HH
