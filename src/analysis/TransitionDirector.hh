#ifndef FALL_N_SRC_ANALYSIS_TRANSITION_DIRECTOR_HH
#define FALL_N_SRC_ANALYSIS_TRANSITION_DIRECTOR_HH

// =============================================================================
//  TransitionDirector — Condition-based phase transition for multiscale
//                       linear → nonlinear dynamic analysis
// =============================================================================
//
//  Provides StepDirector factories that pause a running dynamic analysis
//  when a demand metric exceeds a user-supplied threshold.  The resulting
//  AnalysisState can then be injected into a different analysis engine
//  (e.g. a refined nonlinear model) via AnalysisDirector phases.
//
//  Two built-in strategies:
//
//    1. Displacement threshold — pause when the max component of the
//       global displacement vector exceeds a limit.  Suitable for
//       continuum (hex/tet) models.
//
//    2. Damage criterion threshold — pause when any element's damage
//       index (evaluated via a DamageCriterion) exceeds a limit.
//       Suitable for structural models with fiber sections.
//
//  A TransitionReport captures the trigger details (time, step, metric
//  value, element index for the damage variant) so the caller knows
//  exactly what caused the phase transition.
//
//  Usage:
//
//    // Continuum model: pause when max |u| > 0.005
//    auto [director, report] = make_displacement_threshold_director<ModelT>(0.005);
//    dyn.step_to(10.0, director);
//    if (report.triggered) { /* inspect report.trigger_time, etc. */ }
//
//    // Structural model: pause when damage > 1.0
//    MaxStrainDamageCriterion criterion(eps_y);
//    auto [director, report] = make_damage_threshold_director<ModelT>(criterion, 1.0);
//    dyn.step_to(10.0, director);
//
// =============================================================================

#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <petscvec.h>

#include "StepDirector.hh"
#include "DamageCriterion.hh"


namespace fall_n {


// =============================================================================
//  TransitionReport — what triggered the phase transition
// =============================================================================

struct TransitionReport {
    bool         triggered{false};     ///< True if the threshold was exceeded
    double       trigger_time{0.0};    ///< Time at which the trigger fired
    PetscInt     trigger_step{0};      ///< Step at which the trigger fired
    double       metric_value{0.0};    ///< Value of the metric when triggered
    double       threshold{0.0};       ///< Configured threshold
    std::string  criterion_name;       ///< Name of the criterion used

    // Structural-specific detail (damage threshold only)
    std::size_t  critical_element{0};  ///< Element that triggered (damage only)
    std::size_t  critical_gp{0};       ///< Gauss point within element
    std::size_t  critical_fiber{0};    ///< Fiber within GP
};


// =============================================================================
//  Displacement threshold — pause when max |u_i| > limit
// =============================================================================
//
//  Uses PETSc VecNorm(NORM_INFINITY) on the global displacement vector
//  from the StepEvent.  This is cheap and does not require model access.
//
//  Returns a (StepDirector, shared_ptr<TransitionReport>) pair.  The
//  report is populated the first time the threshold fires.

template <typename ModelT>
auto make_displacement_threshold_director(double u_max)
    -> std::pair<StepDirector<ModelT>, std::shared_ptr<TransitionReport>>
{
    auto report = std::make_shared<TransitionReport>();
    report->threshold      = u_max;
    report->criterion_name = "DisplacementThreshold";

    auto director = [report, u_max]
                    (const StepEvent& ev,
                     [[maybe_unused]] const ModelT& /*model*/) -> StepVerdict
    {
        if (report->triggered) return StepVerdict::Continue;  // fire once

        PetscReal norm_inf;
        VecNorm(ev.displacement, NORM_INFINITY, &norm_inf);

        if (norm_inf > u_max) {
            report->triggered    = true;
            report->trigger_time = ev.time;
            report->trigger_step = ev.step;
            report->metric_value = norm_inf;
            return StepVerdict::Pause;
        }
        return StepVerdict::Continue;
    };

    return {std::move(director), report};
}


// =============================================================================
//  Damage criterion threshold — pause when max damage_index > limit
// =============================================================================
//
//  Evaluates a DamageCriterion on every element after each step.  Heavier
//  than the displacement variant but provides fiber-level detail.
//
//  NOTE: The ModelT must expose structural elements via elements() and
//  provide state_vector() for the u_local required by the criterion.

template <typename ModelT>
auto make_damage_threshold_director(const DamageCriterion& criterion,
                                    double damage_limit)
    -> std::pair<StepDirector<ModelT>, std::shared_ptr<TransitionReport>>
{
    auto report = std::make_shared<TransitionReport>();
    report->threshold      = damage_limit;
    report->criterion_name = criterion.name();

    auto crit_clone = criterion.clone();  // own a copy
    // shared_ptr makes the lambda copy-constructible (required by std::function)
    auto crit_shared = std::shared_ptr<DamageCriterion>(std::move(crit_clone));

    auto director = [report, damage_limit,
                     crit = crit_shared]
                    (const StepEvent& ev,
                     const ModelT& model) mutable -> StepVerdict
    {
        if (report->triggered) return StepVerdict::Continue;

        Vec u_local = model.state_vector();

        double max_damage = 0.0;
        ElementDamageInfo worst{};

        for (std::size_t e = 0; e < model.elements().size(); ++e) {
            const auto& elem = model.elements()[e];

            // Only evaluate structural elements (those with section_snapshots)
            if constexpr (requires { elem.section_snapshots(); }) {
                auto info = crit->evaluate_element(elem, e, u_local);
                if (info.damage_index > max_damage) {
                    max_damage = info.damage_index;
                    worst      = info;
                }
            }
        }

        if (max_damage > damage_limit) {
            report->triggered        = true;
            report->trigger_time     = ev.time;
            report->trigger_step     = ev.step;
            report->metric_value     = max_damage;
            report->critical_element = worst.element_index;
            report->critical_gp      = worst.critical_gp;
            report->critical_fiber   = worst.critical_fiber;
            return StepVerdict::Pause;
        }
        return StepVerdict::Continue;
    };

    return {std::move(director), report};
}


// =============================================================================
//  ExceedanceReport — node-level displacement exceedance summary
// =============================================================================
//
//  After a phase transition, identifies which nodes have displacements
//  exceeding a given fraction of the trigger value.  Useful for deciding
//  which region of the model needs refinement.

struct NodeExceedance {
    std::size_t  node_id;
    double       displacement_norm;   ///< ‖u_node‖
};

struct ExceedanceReport {
    double                      threshold;
    std::vector<NodeExceedance> nodes;      ///< Sorted by displacement (descending)
};

/// Build an ExceedanceReport from a global displacement vector.
///
/// Walks all nodes in the domain, computes the Euclidean norm of each
/// node's displacement DOFs, and collects those exceeding `u_threshold`.
/// Uses DMGlobalToLocal to properly map constrained (BC) DOFs.
template <typename DomainT>
ExceedanceReport compute_exceedance_report(
    const DomainT& domain, Vec u_global, double u_threshold)
{
    ExceedanceReport report;
    report.threshold = u_threshold;

    // Scatter global → local so we can use local DOF indices
    DM dm;
    VecGetDM(u_global, &dm);

    Vec u_local;
    DMGetLocalVector(dm, &u_local);
    VecSet(u_local, 0.0);
    DMGlobalToLocal(dm, u_global, INSERT_VALUES, u_local);

    const PetscScalar* u_array;
    VecGetArrayRead(u_local, &u_array);

    PetscInt vec_size;
    VecGetLocalSize(u_local, &vec_size);

    for (const auto& node : domain.nodes()) {
        double norm_sq = 0.0;
        for (auto idx : node.dof_index()) {
            if (idx >= 0 && idx < vec_size) {
                double val = u_array[idx];
                norm_sq += val * val;
            }
        }
        double norm = std::sqrt(norm_sq);
        if (norm > u_threshold) {
            report.nodes.push_back({
                static_cast<std::size_t>(node.id()),
                norm
            });
        }
    }

    VecRestoreArrayRead(u_local, &u_array);
    DMRestoreLocalVector(dm, &u_local);

    // Sort descending by displacement norm
    std::sort(report.nodes.begin(), report.nodes.end(),
              [](const auto& a, const auto& b) {
                  return a.displacement_norm > b.displacement_norm;
              });

    return report;
}


// =============================================================================
//  inject_dynamic_state — Transfer kinematic state into a DynamicAnalysis
// =============================================================================
//
//  After a phase transition (StepVerdict::Pause), the caller may want to
//  continue the simulation on a different (higher-fidelity) DynamicAnalysis
//  instance, starting from the same kinematic state that caused the pause.
//
//  This function performs the injection:
//
//    1. Copies `u_global` into the target's initial/live displacement vector.
//    2. Copies `v_global` into the target's initial/live velocity vector.
//    3. Sets the PETSc TS time to `t` so that the target begins at the
//       correct time position within the earthquake record.
//
//  Works both BEFORE and AFTER target.setup():
//    - Before setup: sets the initial-condition vectors used by TS2SetSolution.
//    - After  setup: writes directly into the live TS solution vectors (safe
//      because TS2SetSolution stores a reference, not a copy).
//
//  Acceleration cannot be injected directly — PETSc's α₂/Newmark integrators
//  compute it internally.  The first time step will produce a fresh tangent
//  prediction; for dynamics this is usually acceptable.
//
//  Usage:
//
//    auto [director, report] = make_displacement_threshold_director<ModelA>(u_lim);
//    dyn_a.step_to(t_end, director);
//
//    if (report.triggered) {
//        auto ev = report;  // has trigger_time, metric_value …
//        fall_n::inject_dynamic_state(dyn_b,
//                                     ev_step.displacement,
//                                     ev_step.velocity,
//                                     ev_step.time);
//        dyn_b.solve(ev_step.time, t_end, dt_fine);
//    }

template <typename DynamicAnalysisT>
void inject_dynamic_state(DynamicAnalysisT& target,
                           Vec u_global,
                           Vec v_global,
                           double t = 0.0)
{
    // set_initial_displacement / set_initial_velocity handle both the
    // pre-setup case (create vector, copy) and the post-setup case
    // (vector already exists, plain VecCopy updates the live TS solver).
    target.set_initial_displacement(u_global);
    target.set_initial_velocity(v_global);

    // Advance the TS clock so that time functions (loads, ground motion)
    // evaluate at the correct time after the restart.
    if (t != 0.0) {
        TSSetTime(target.get_ts(), static_cast<PetscReal>(t));
    }
}


} // namespace fall_n

#endif // FALL_N_SRC_ANALYSIS_TRANSITION_DIRECTOR_HH
