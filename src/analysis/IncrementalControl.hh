#ifndef FALL_N_SRC_ANALYSIS_INCREMENTAL_CONTROL_HH
#define FALL_N_SRC_ANALYSIS_INCREMENTAL_CONTROL_HH

#include <cstddef>
#include <utility>
#include <vector>
#include <petscvec.h>

// =============================================================================
//  IncrementalControlPolicy — concept for incremental stepping strategies
// =============================================================================
//
//  A control policy determines the system state for a given control
//  parameter p ∈ [0, 1].  The contract is:
//
//    scheme.apply(p, f_full, f_ext, model)
//
//  where:
//    p      — control parameter (0 at start, 1 at full load/displacement)
//    f_full — reference external force vector (unscaled, read-only)
//    f_ext  — external force vector to fill (consumed by SNES residual)
//    model  — the Model object (to modify imposed_solution, etc.)
//
//  INVARIANT: apply() must be ABSOLUTE (idempotent for a given p).
//  Calling apply(p, ...) at any time fully determines the system state.
//  This makes bisection rollback trivial — no save/restore needed.
//
//  Three concrete policies are provided:
//    - LoadControl            (default — pure load scaling)
//    - DisplacementControl    (prescribed DOF(s) + optional proportional load)
//    - CustomControl<F>       (user-supplied callable — maximum flexibility)
//
// =============================================================================

template <typename S, typename ModelT>
concept IncrementalControlPolicy = requires(
    S& scheme, double p, Vec f_full, Vec f_ext, ModelT* model)
{
    scheme.apply(p, f_full, f_ext, model);
};

// =============================================================================
//  LoadControl — pure proportional load scaling (default)
// =============================================================================
//
//  At parameter p:  f_ext = p · f_full
//
//  This reproduces the original hardcoded behaviour of advance_to_lambda_.

struct LoadControl {
    template <typename ModelT>
    void apply(double p, Vec f_full, Vec f_ext,
               [[maybe_unused]] ModelT*) const
    {
        VecCopy(f_full, f_ext);
        VecScale(f_ext, p);
    }
};

// =============================================================================
//  DisplacementControl — prescribed DOF(s) + optional proportional load
// =============================================================================
//
//  Controls one or more DOFs linearly with the parameter p:
//
//      imposed_value(dof_i) = p · target_i
//      f_ext                = p · load_scale · f_full
//
//  The controlled DOFs MUST have been pre-constrained in the Model
//  (via constrain_dof) before setup() — this policy only updates
//  the imposed value, not the DM section.
//
//  The bisection engine works because apply() is absolute: for any p,
//  the DOF values and load level are fully determined.
//
//  Usage:
//    // Single DOF — pushover at node 42, dof 0, target 0.01
//    DisplacementControl dc{42, 0, 0.01};
//
//    // Multi-DOF with proportional load
//    DisplacementControl dc{{{10, 0, 0.01}, {11, 0, 0.01}}, 0.5};

struct DisplacementControl {
    struct PrescribedDOF {
        std::size_t node;
        std::size_t dof;
        double      target;
    };

    std::vector<PrescribedDOF> controlled_dofs;
    double load_scale{0.0};

    // Single DOF convenience constructor
    DisplacementControl(std::size_t node, std::size_t dof,
                        double target, double scale = 0.0)
        : controlled_dofs{{node, dof, target}}, load_scale{scale} {}

    // Multi-DOF constructor
    DisplacementControl(std::vector<PrescribedDOF> dofs, double scale = 0.0)
        : controlled_dofs{std::move(dofs)}, load_scale{scale} {}

    template <typename ModelT>
    void apply(double p, Vec f_full, Vec f_ext, ModelT* model) {
        // Scale external forces (may be zero if load_scale == 0)
        VecCopy(f_full, f_ext);
        VecScale(f_ext, p * load_scale);

        // Update each controlled DOF to p · target
        for (const auto& [node, dof, target] : controlled_dofs) {
            model->update_imposed_value(node, dof, p * target);
        }
    }
};

// =============================================================================
//  CustomControl<F> — user-supplied callable for maximum flexibility
// =============================================================================
//
//  Wraps any callable with signature:
//    void(double p, Vec f_full, Vec f_ext, ModelT* model)
//
//  Usage:
//    auto scheme = make_control([](double p, Vec f_full, Vec f_ext, auto* m) {
//        VecCopy(f_full, f_ext);
//        VecScale(f_ext, std::sqrt(p));
//        m->update_imposed_value(42, 0, p * 0.01);
//    });

template <typename F>
struct CustomControl {
    F apply_fn;

    template <typename ModelT>
    void apply(double p, Vec f_full, Vec f_ext, ModelT* model) {
        apply_fn(p, f_full, f_ext, model);
    }
};

template <typename F>
CustomControl(F) -> CustomControl<F>;

template <typename F>
auto make_control(F&& f) {
    return CustomControl<std::decay_t<F>>{std::forward<F>(f)};
}

// =============================================================================
//  ArcLengthControl — stub (not yet implemented)
// =============================================================================
//
//  Arc-length (Riks) requires augmenting the SNES system to (u, λ) and
//  a constraint equation ‖Δu‖² + Δλ²‖f‖² = Δs².  The parameter of
//  control becomes the arc-length s, not p.  This needs structural
//  changes to the bisection engine (reinterpret stepping parameter as s).
//
//  TODO: Implement when snap-through/snap-back analysis is needed.


#endif // FALL_N_SRC_ANALYSIS_INCREMENTAL_CONTROL_HH
