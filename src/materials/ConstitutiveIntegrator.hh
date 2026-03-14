#ifndef FALL_N_CONSTITUTIVE_INTEGRATOR_HH
#define FALL_N_CONSTITUTIVE_INTEGRATOR_HH

#include <concepts>
#include <type_traits>
#include <utility>

#include "../continuum/ConstitutiveKinematics.hh"
#include "local_problem/ContinuumLocalProblem.hh"

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

namespace constitutive_integrators {

template <typename ConstitutiveSiteT>
using RelationType = std::remove_cvref_t<decltype(std::declval<const ConstitutiveSiteT&>().constitutive_relation())>;

template <typename ConstitutiveSiteT, std::size_t dim>
using ContinuumKinematicsT = continuum::ConstitutiveKinematics<dim>;

template <typename ConstitutiveSiteT, std::size_t dim>
[[nodiscard]] inline auto fallback_continuum_response(
    const ConstitutiveSiteT& site,
    const ContinuumKinematicsT<ConstitutiveSiteT, dim>& kin)
    -> typename ConstitutiveSiteT::ConjugateT
{
    if constexpr (requires {
        site.algorithmic_state();
        site.constitutive_relation().compute_response(kin, site.algorithmic_state());
    }) {
        return site.constitutive_relation().compute_response(kin, site.algorithmic_state());
    } else if constexpr (requires {
        site.constitutive_relation().compute_response(kin);
    }) {
        return site.constitutive_relation().compute_response(kin);
    } else {
        return site.constitutive_relation().compute_response(
            continuum::make_kinematic_measure<typename ConstitutiveSiteT::KinematicT>(kin));
    }
}

template <typename ConstitutiveSiteT, std::size_t dim>
[[nodiscard]] inline auto fallback_continuum_tangent(
    const ConstitutiveSiteT& site,
    const ContinuumKinematicsT<ConstitutiveSiteT, dim>& kin)
    -> typename ConstitutiveSiteT::TangentT
{
    if constexpr (requires {
        site.algorithmic_state();
        site.constitutive_relation().tangent(kin, site.algorithmic_state());
    }) {
        return site.constitutive_relation().tangent(kin, site.algorithmic_state());
    } else if constexpr (requires {
        site.constitutive_relation().tangent(kin);
    }) {
        return site.constitutive_relation().tangent(kin);
    } else {
        return site.constitutive_relation().tangent(
            continuum::make_kinematic_measure<typename ConstitutiveSiteT::KinematicT>(kin));
    }
}

template <typename ConstitutiveSiteT>
inline void fallback_small_strain_commit(
    ConstitutiveSiteT& site,
    const typename ConstitutiveSiteT::KinematicT& k)
{
    if constexpr (requires {
        site.algorithmic_state();
        site.constitutive_relation().commit(site.algorithmic_state(), k);
    }) {
        site.constitutive_relation().commit(site.algorithmic_state(), k);
    } else if constexpr (requires { site.constitutive_relation().update(k); }) {
        site.constitutive_relation().update(k);
    }

    commit_constitutive_state(site, k);
}

template <typename ConstitutiveSiteT, std::size_t dim>
inline void fallback_continuum_commit(
    ConstitutiveSiteT& site,
    const ContinuumKinematicsT<ConstitutiveSiteT, dim>& kin)
{
    if constexpr (requires {
        site.algorithmic_state();
        site.constitutive_relation().commit(site.algorithmic_state(), kin);
    }) {
        site.constitutive_relation().commit(site.algorithmic_state(), kin);
    } else if constexpr (requires { site.constitutive_relation().update(kin); }) {
        site.constitutive_relation().update(kin);
    } else {
        auto k = continuum::make_kinematic_measure<typename ConstitutiveSiteT::KinematicT>(kin);
        fallback_small_strain_commit(site, k);
        return;
    }

    commit_constitutive_state(
        site,
        continuum::make_kinematic_measure<typename ConstitutiveSiteT::KinematicT>(kin));
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

    template <typename ConstitutiveSiteT, std::size_t dim>
        requires (ConstitutiveSiteT::MaterialPolicy::is_continuum_space &&
                  ConstitutiveSiteT::dim == dim)
    [[nodiscard]] auto compute_response(
        const ConstitutiveSiteT& site,
        const continuum::ConstitutiveKinematics<dim>& kin) const
        -> typename ConstitutiveSiteT::ConjugateT
    {
        return constitutive_integrators::fallback_continuum_response(site, kin);
    }

    template <typename ConstitutiveSiteT, std::size_t dim>
        requires (ConstitutiveSiteT::MaterialPolicy::is_continuum_space &&
                  ConstitutiveSiteT::dim == dim)
    [[nodiscard]] auto tangent(
        const ConstitutiveSiteT& site,
        const continuum::ConstitutiveKinematics<dim>& kin) const
        -> typename ConstitutiveSiteT::TangentT
    {
        return constitutive_integrators::fallback_continuum_tangent(site, kin);
    }

    template <typename ConstitutiveSiteT>
    void commit(
        ConstitutiveSiteT&,
        const typename ConstitutiveSiteT::KinematicT&) const
    {
        // No-op by design: preserves the legacy elastic workflow.
    }

    template <typename ConstitutiveSiteT, std::size_t dim>
        requires (ConstitutiveSiteT::MaterialPolicy::is_continuum_space &&
                  ConstitutiveSiteT::dim == dim)
    void commit(
        ConstitutiveSiteT&,
        const continuum::ConstitutiveKinematics<dim>&) const
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

    template <typename ConstitutiveSiteT, std::size_t dim>
        requires (ConstitutiveSiteT::MaterialPolicy::is_continuum_space &&
                  ConstitutiveSiteT::dim == dim)
    [[nodiscard]] auto compute_response(
        const ConstitutiveSiteT& site,
        const continuum::ConstitutiveKinematics<dim>& kin) const
        -> typename ConstitutiveSiteT::ConjugateT
    {
        return constitutive_integrators::fallback_continuum_response(site, kin);
    }

    template <typename ConstitutiveSiteT, std::size_t dim>
        requires (ConstitutiveSiteT::MaterialPolicy::is_continuum_space &&
                  ConstitutiveSiteT::dim == dim)
    [[nodiscard]] auto tangent(
        const ConstitutiveSiteT& site,
        const continuum::ConstitutiveKinematics<dim>& kin) const
        -> typename ConstitutiveSiteT::TangentT
    {
        return constitutive_integrators::fallback_continuum_tangent(site, kin);
    }

    template <typename ConstitutiveSiteT>
    void commit(
        ConstitutiveSiteT& site,
        const typename ConstitutiveSiteT::KinematicT& k) const
    {
        constitutive_integrators::fallback_small_strain_commit(site, k);
    }

    template <typename ConstitutiveSiteT, std::size_t dim>
        requires (ConstitutiveSiteT::MaterialPolicy::is_continuum_space &&
                  ConstitutiveSiteT::dim == dim)
    void commit(
        ConstitutiveSiteT& site,
        const continuum::ConstitutiveKinematics<dim>& kin) const
    {
        constitutive_integrators::fallback_continuum_commit(site, kin);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// ContinuumLocalProblemIntegrator
// ─────────────────────────────────────────────────────────────────────────────
//
// Compile-time bridge between:
//   - a continuum-aware constitutive site,
//   - a continuum-local nonlinear problem,
//   - an injected local nonlinear solver.
//
// This is the first constitutive-integrator path that consumes the generic
// `ContinuumLocalProblemPolicy` infrastructure at the real `Material<>`
// boundary. If the local problem is not compatible with the relation, the
// integrator falls back to the direct continuum path so that the same wrapper
// can still host elastic or legacy inelastic relations.
//
// ─────────────────────────────────────────────────────────────────────────────

template <
    typename LocalProblemT,
    typename LocalSolverT = NewtonLocalSolver<>
>
struct ContinuumLocalProblemIntegrator {
    [[no_unique_address]] LocalProblemT local_problem_{};
    [[no_unique_address]] LocalSolverT local_solver_{};

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

    template <typename ConstitutiveSiteT, std::size_t dim>
        requires (ConstitutiveSiteT::MaterialPolicy::is_continuum_space &&
                  ConstitutiveSiteT::dim == dim)
    [[nodiscard]] auto compute_response(
        const ConstitutiveSiteT& site,
        const continuum::ConstitutiveKinematics<dim>& kin) const
        -> typename ConstitutiveSiteT::ConjugateT
    {
        using RelationT = typename ConstitutiveSiteT::RelationT;
        if constexpr (ContinuumLocalProblemPolicy<LocalProblemT, RelationT> &&
                      LocalNonlinearSolverPolicy<
                          LocalSolverT,
                          LocalProblemT,
                          RelationT,
                          typename LocalProblemT::template ContextT<RelationT>>) {
            auto result = ContinuumLocalIntegrationAlgorithm<LocalProblemT, LocalSolverT>{
                local_problem_, local_solver_
            }.integrate(site.constitutive_relation(), kin, site.algorithmic_state());
            return result.response;
        } else {
            return constitutive_integrators::fallback_continuum_response(site, kin);
        }
    }

    template <typename ConstitutiveSiteT, std::size_t dim>
        requires (ConstitutiveSiteT::MaterialPolicy::is_continuum_space &&
                  ConstitutiveSiteT::dim == dim)
    [[nodiscard]] auto tangent(
        const ConstitutiveSiteT& site,
        const continuum::ConstitutiveKinematics<dim>& kin) const
        -> typename ConstitutiveSiteT::TangentT
    {
        using RelationT = typename ConstitutiveSiteT::RelationT;
        if constexpr (ContinuumLocalProblemPolicy<LocalProblemT, RelationT> &&
                      LocalNonlinearSolverPolicy<
                          LocalSolverT,
                          LocalProblemT,
                          RelationT,
                          typename LocalProblemT::template ContextT<RelationT>>) {
            auto result = ContinuumLocalIntegrationAlgorithm<LocalProblemT, LocalSolverT>{
                local_problem_, local_solver_
            }.integrate(site.constitutive_relation(), kin, site.algorithmic_state());
            return result.tangent;
        } else {
            return constitutive_integrators::fallback_continuum_tangent(site, kin);
        }
    }

    template <typename ConstitutiveSiteT>
    void commit(
        ConstitutiveSiteT& site,
        const typename ConstitutiveSiteT::KinematicT& k) const
    {
        constitutive_integrators::fallback_small_strain_commit(site, k);
    }

    template <typename ConstitutiveSiteT, std::size_t dim>
        requires (ConstitutiveSiteT::MaterialPolicy::is_continuum_space &&
                  ConstitutiveSiteT::dim == dim)
    void commit(
        ConstitutiveSiteT& site,
        const continuum::ConstitutiveKinematics<dim>& kin) const
    {
        using RelationT = typename ConstitutiveSiteT::RelationT;
        if constexpr (ContinuumLocalProblemPolicy<LocalProblemT, RelationT> &&
                      LocalNonlinearSolverPolicy<
                          LocalSolverT,
                          LocalProblemT,
                          RelationT,
                          typename LocalProblemT::template ContextT<RelationT>>) {
            auto result = ContinuumLocalIntegrationAlgorithm<LocalProblemT, LocalSolverT>{
                local_problem_, local_solver_
            }.integrate(site.constitutive_relation(), kin, site.algorithmic_state());
            site.algorithmic_state() = result.algorithmic_state;
            constitutive_integrators::commit_constitutive_state(
                site,
                continuum::make_kinematic_measure<typename ConstitutiveSiteT::KinematicT>(kin));
        } else {
            constitutive_integrators::fallback_continuum_commit(site, kin);
        }
    }
};

// Semantic names
using ElasticConstitutiveIntegrator = PassThroughIntegrator;
using EmbeddedInelasticConstitutiveIntegrator = EmbeddedRelationIntegrator;

// Legacy aliases retained for compatibility across the codebase.
using ElasticUpdate = PassThroughIntegrator;
using InelasticUpdate = EmbeddedRelationIntegrator;

#endif // FALL_N_CONSTITUTIVE_INTEGRATOR_HH
