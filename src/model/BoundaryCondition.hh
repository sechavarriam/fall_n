#ifndef FN_BOUNDARY_CONDITION_HH
#define FN_BOUNDARY_CONDITION_HH

// =============================================================================
//  BoundaryCondition.hh — Time-dependent boundary condition infrastructure
// =============================================================================
//
//  Provides time-parametric functions and boundary condition descriptors
//  for both static and dynamic analyses.
//
//  ─── TimeFunction ───────────────────────────────────────────────────────
//
//  A TimeFunction maps a real scalar t ∈ ℝ to a value v ∈ ℝ.  It is the
//  fundamental building block for time-varying loads, prescribed
//  displacements, and ground motions.
//
//  Factory functions in namespace time_fn create common patterns:
//    constant(v)              →  v
//    linear_ramp(t0,t1,v0,v1) →  linear interpolation, clamped
//    harmonic(A, ω, φ)        →  A·sin(ω·t + φ)
//    cosine(A, ω, φ)          →  A·cos(ω·t + φ)
//    step(t_step, v0, v1)     →  v0 for t < t_step, v1 otherwise
//    pulse(t0, t1, val)       →  val in [t0, t1], 0 elsewhere
//    piecewise_linear({ti,vi}) →  interpolated from sorted data
//    exponential_decay(A, α)  →  A·exp(−α·t)
//
//  Combinators:
//    product(f, g)  → f(t)·g(t)
//    sum(f, g)      → f(t)+g(t)
//    scale(c, f)    → c·f(t)
//
//  ─── Boundary condition types ───────────────────────────────────────────
//
//  NodalForceBC<dim>    — time-varying nodal force (Neumann)
//  PrescribedBC         — time-varying imposed displacement (Dirichlet)
//  GroundMotionBC       — uniform base acceleration (earthquake input)
//
//  ─── Initial conditions ─────────────────────────────────────────────────
//
//  NodalInitialCondition<dim> — per-node initial displacement/velocity
//  ElementInitialStress<dim>  — per-element initial stress (prestress)
//
//  ─── BoundaryConditionSet<dim> ──────────────────────────────────────────
//
//  Aggregates all BCs and provides evaluators:
//    bc_set.evaluate_forces(t, f_local, model)
//    bc_set.evaluate_prescribed(t, u_imposed, model)
//    bc_set.apply_initial_displacement(u_local, model)
//    bc_set.apply_initial_velocity(v_local, model)
//
//  DynamicAnalysis uses these evaluators in its PETSc TS callbacks.
//
// =============================================================================

#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>
#include <array>
#include <algorithm>
#include <stdexcept>
#include <utility>
#include <string>
#include <iostream>

#include <petsc.h>


// =============================================================================
//  TimeFunction — A scalar function of time
// =============================================================================
//
//  Wraps std::function<double(double)> with value semantics.
//  A null/empty TimeFunction evaluates to 0.0 for any t.

using TimeFunction = std::function<double(double)>;


// =============================================================================
//  time_fn — Factory functions for common time patterns
// =============================================================================

namespace time_fn {

/// Constant:  v(t) = val  ∀t
inline TimeFunction constant(double val) {
    return [val](double /*t*/) -> double { return val; };
}

/// Zero:  v(t) = 0  ∀t
inline TimeFunction zero() {
    return [](double /*t*/) -> double { return 0.0; };
}

/// Linear ramp:  v(t) = v0 + (v1−v0)·(t−t0)/(t1−t0),  clamped
///
///   t ≤ t0  →  v0
///   t ≥ t1  →  v1
///   else    →  linear interpolation
inline TimeFunction linear_ramp(double t0, double t1, double v0, double v1) {
    return [=](double t) -> double {
        if (t <= t0) return v0;
        if (t >= t1) return v1;
        return v0 + (v1 - v0) * (t - t0) / (t1 - t0);
    };
}

/// Harmonic (sine):  v(t) = A·sin(ω·t + φ)
inline TimeFunction harmonic(double amplitude, double omega, double phase = 0.0) {
    return [=](double t) -> double {
        return amplitude * std::sin(omega * t + phase);
    };
}

/// Cosine:  v(t) = A·cos(ω·t + φ)
inline TimeFunction cosine(double amplitude, double omega, double phase = 0.0) {
    return [=](double t) -> double {
        return amplitude * std::cos(omega * t + phase);
    };
}

/// Step:  v(t) = v_before  for t < t_step,   v_after  for t ≥ t_step
inline TimeFunction step(double t_step, double v_before = 0.0, double v_after = 1.0) {
    return [=](double t) -> double {
        return (t < t_step) ? v_before : v_after;
    };
}

/// Pulse:  v(t) = val  for t ∈ [t_start, t_end],  0 otherwise
inline TimeFunction pulse(double t_start, double t_end, double val = 1.0) {
    return [=](double t) -> double {
        return (t >= t_start && t <= t_end) ? val : 0.0;
    };
}

/// Piecewise linear:  sorted data points {(t_i, v_i)},  clamped at extremes
///
/// Requires at least 2 data points.  Sorts automatically.
inline TimeFunction piecewise_linear(std::vector<std::pair<double, double>> data) {
    if (data.size() < 2)
        throw std::invalid_argument("time_fn::piecewise_linear requires ≥ 2 data points");
    std::sort(data.begin(), data.end());
    return [data = std::move(data)](double t) -> double {
        if (t <= data.front().first) return data.front().second;
        if (t >= data.back().first)  return data.back().second;
        auto it = std::lower_bound(data.begin(), data.end(), t,
            [](const auto& p, double val) { return p.first < val; });
        if (it == data.begin()) return data.front().second;
        auto prev = std::prev(it);
        double alpha = (t - prev->first) / (it->first - prev->first);
        return (1.0 - alpha) * prev->second + alpha * it->second;
    };
}

/// Exponential decay:  v(t) = A·exp(−α·t)
inline TimeFunction exponential_decay(double amplitude, double decay_rate) {
    return [=](double t) -> double {
        return amplitude * std::exp(-decay_rate * t);
    };
}

/// Multiply:  (f·g)(t) = f(t)·g(t)
inline TimeFunction product(TimeFunction f, TimeFunction g) {
    return [f = std::move(f), g = std::move(g)](double t) -> double {
        return f(t) * g(t);
    };
}

/// Sum:  (f+g)(t) = f(t)+g(t)
inline TimeFunction sum(TimeFunction f, TimeFunction g) {
    return [f = std::move(f), g = std::move(g)](double t) -> double {
        return f(t) + g(t);
    };
}

/// Scale:  (c·f)(t) = c·f(t)
inline TimeFunction scale(double c, TimeFunction f) {
    return [c, f = std::move(f)](double t) -> double {
        return c * f(t);
    };
}

} // namespace time_fn


// =============================================================================
//  NodalForceBC<dim> — Time-dependent nodal force
// =============================================================================
//
//  Specifies a time-varying force applied to a single node.
//  Each DOF component has its own TimeFunction.
//
//  Example (3D harmonic force in x):
//    NodalForceBC<3> bc;
//    bc.node_id = 10;
//    bc.components[0] = time_fn::harmonic(1000.0, 2*M_PI*5.0);  // 5 Hz
//    bc.components[1] = time_fn::zero();
//    bc.components[2] = time_fn::zero();

template <std::size_t dim>
struct NodalForceBC {
    std::size_t node_id;
    std::array<TimeFunction, dim> components;

    /// Evaluate all components at time t
    std::array<double, dim> evaluate(double t) const {
        std::array<double, dim> f{};
        for (std::size_t i = 0; i < dim; ++i)
            f[i] = components[i] ? components[i](t) : 0.0;
        return f;
    }
};


// =============================================================================
//  PrescribedBC — Time-dependent imposed displacement (Dirichlet)
// =============================================================================
//
//  Imposes u_{node, dof}(t) = displacement(t) on a single DOF.
//  The DOF must be constrained (via fix_node or fix_dof) in the model.
//
//  For dynamics, velocity() and acceleration() can optionally be provided
//  for correct mass/damping coupling with prescribed motion.  If not
//  given, they are assumed zero (quasi-static BC application).
//
//  For ground motion input, use GroundMotionBC instead.

struct PrescribedBC {
    std::size_t  node_id;
    std::size_t  local_dof;      // 0-based within the node

    TimeFunction displacement;    // u_D(t)
    TimeFunction velocity;        // u̇_D(t)  — optional
    TimeFunction acceleration;    // ü_D(t)  — optional

    double eval_displacement(double t) const {
        return displacement ? displacement(t) : 0.0;
    }
    double eval_velocity(double t) const {
        return velocity ? velocity(t) : 0.0;
    }
    double eval_acceleration(double t) const {
        return acceleration ? acceleration(t) : 0.0;
    }
};


// =============================================================================
//  GroundMotionBC — Uniform base acceleration (earthquake input)
// =============================================================================
//
//  Applies a uniform ground acceleration in a given direction.
//  Equivalent to adding pseudo-forces:
//
//    f_pseudo = −M · 1̂_dir · a_g(t)
//
//  where 1̂_dir is the influence vector (ones in the given DOF direction,
//  zeros elsewhere) and M is the mass matrix.
//
//  When used with DynamicAnalysis, the pseudo-force is applied automatically
//  during mass matrix multiplication.
//
//  Direction: 0 = x,  1 = y,  2 = z

struct GroundMotionBC {
    std::size_t  direction;       // DOF direction (0, 1, or 2)
    TimeFunction acceleration;    // a_g(t) — ground acceleration

    double eval(double t) const {
        return acceleration ? acceleration(t) : 0.0;
    }
};


// =============================================================================
//  NodalInitialCondition<dim> — Per-node initial state
// =============================================================================
//
//  Specifies initial displacement and/or velocity at a node.
//  Nodes without an explicit IC start from zero.

template <std::size_t dim>
struct NodalInitialCondition {
    std::size_t node_id;
    std::array<double, dim> displacement{};
    std::array<double, dim> velocity{};
};


// =============================================================================
//  ElementInitialStress<dim> — Per-element initial stress
// =============================================================================
//
//  Provides initial stress tensors for all Gauss points of an element.
//  These represent pre-existing stresses (geostatic, prestress, residual)
//  that exist before any displacement occurs:
//
//    f_int(u) = ∫ Bᵀ·(σ(ε(u)) + σ₀) dV
//
//  where σ₀ is the initial stress at each Gauss point.

template <std::size_t dim>
struct ElementInitialStress {
    std::size_t element_index;    // 0-based in model's element container

    static constexpr std::size_t nvoigt = dim * (dim + 1) / 2;

    /// Stress per Gauss point (Voigt notation)
    std::vector<std::array<double, nvoigt>> gauss_point_stresses;
};


// =============================================================================
//  BoundaryConditionSet<dim> — Aggregate BC container with evaluators
// =============================================================================
//
//  Collects all boundary conditions and provides methods to evaluate
//  them at a given time t, filling PETSc vectors compatible with the
//  Model/Analysis framework.
//
//  This class is the primary interface between the BC specification
//  and the analysis solver.  DynamicAnalysis<...> accepts a
//  BoundaryConditionSet<dim> and uses it in TS callbacks.
//
//  Usage:
//    BoundaryConditionSet<3> bcs;
//    bcs.add_force({node_id, {time_fn::harmonic(P, w), zero(), zero()}});
//    bcs.add_initial_condition({tip_node, {0.01, 0, 0}, {0, 0, 0}});
//    DynamicAnalysis analysis(model, bcs);

template <std::size_t dim>
class BoundaryConditionSet {

    std::vector<NodalForceBC<dim>>          forces_;
    std::vector<PrescribedBC>               prescribed_;
    std::vector<GroundMotionBC>             ground_motions_;
    std::vector<NodalInitialCondition<dim>> initial_conditions_;
    std::vector<ElementInitialStress<dim>>  initial_stresses_;

    public:

    // ── Add boundary conditions ──────────────────────────────────────

    void add_force(NodalForceBC<dim> bc)          { forces_.push_back(std::move(bc)); }
    void add_prescribed(PrescribedBC bc)           { prescribed_.push_back(std::move(bc)); }

    void add_ground_motion(GroundMotionBC gm, double factor = 1.0) {
        if (factor != 1.0) {
            gm.acceleration = time_fn::scale(factor, gm.acceleration);
        }
        ground_motions_.push_back(std::move(gm));
    }

    void add_initial_condition(NodalInitialCondition<dim> ic) { initial_conditions_.push_back(std::move(ic)); }
    void add_initial_stress(ElementInitialStress<dim> is)     { initial_stresses_.push_back(std::move(is)); }


    // ── Accessors ────────────────────────────────────────────────────

    const auto& forces()             const noexcept { return forces_; }
    const auto& prescribed()         const noexcept { return prescribed_; }
    const auto& ground_motions()     const noexcept { return ground_motions_; }
    const auto& initial_conditions() const noexcept { return initial_conditions_; }
    const auto& initial_stresses()   const noexcept { return initial_stresses_; }

    bool has_ground_motion() const noexcept { return !ground_motions_.empty(); }
    bool has_prescribed()    const noexcept { return !prescribed_.empty(); }
    bool has_initial_stress() const noexcept { return !initial_stresses_.empty(); }

    bool empty() const noexcept {
        return forces_.empty() && prescribed_.empty() &&
               ground_motions_.empty() && initial_conditions_.empty();
    }


    // ── Evaluate time-dependent nodal forces → PETSc local vector ────
    //
    //  Fills f_local with the sum of all time-dependent nodal forces
    //  evaluated at time t.
    //
    //  ModelT must provide get_domain().node(id), with DOF index access.

    template <typename ModelT>
    void evaluate_forces(double t, Vec f_local, ModelT& model) const
    {
        VecSet(f_local, 0.0);

        for (const auto& bc : forces_) {
            const auto& node = model.get_domain().node(bc.node_id);
            auto f = bc.evaluate(t);
            auto num_dofs = static_cast<PetscInt>(node.num_dof());
            const auto* dofs = node.dof_data();

            std::array<PetscScalar, dim> vals{};
            for (std::size_t i = 0; i < dim; ++i)
                vals[i] = f[i];

            VecSetValuesLocal(f_local, num_dofs, dofs, vals.data(), ADD_VALUES);
        }

        VecAssemblyBegin(f_local);
        VecAssemblyEnd(f_local);
    }


    // ── Evaluate prescribed displacements → imposed solution vector ──
    //
    //  Fills u_imposed (local) with the non-zero prescribed displacement
    //  values at time t.  Constrained DOFs should already be registered
    //  in the Model via fix_node/fix_dof.

    template <typename ModelT>
    void evaluate_prescribed(double t, Vec u_imposed, ModelT& model) const
    {
        VecSet(u_imposed, 0.0);

        for (const auto& bc : prescribed_) {
            const auto& node = model.get_domain().node(bc.node_id);
            PetscInt dof_idx = node.dof_index()[bc.local_dof];
            PetscScalar val = bc.eval_displacement(t);

            VecSetValueLocal(u_imposed, dof_idx, val, INSERT_VALUES);
        }

        VecAssemblyBegin(u_imposed);
        VecAssemblyEnd(u_imposed);
    }


    // ── Apply initial displacement to a local vector ─────────────────

    template <typename ModelT>
    void apply_initial_displacement(Vec u_local, ModelT& model) const
    {
        for (const auto& ic : initial_conditions_) {
            const auto& node = model.get_domain().node(ic.node_id);
            auto num_dofs = static_cast<PetscInt>(node.num_dof());
            const auto* dofs = node.dof_data();

            std::array<PetscScalar, dim> vals{};
            for (std::size_t i = 0; i < dim; ++i)
                vals[i] = ic.displacement[i];

            VecSetValuesLocal(u_local, num_dofs, dofs, vals.data(), INSERT_VALUES);
        }

        VecAssemblyBegin(u_local);
        VecAssemblyEnd(u_local);
    }


    // ── Apply initial velocity to a local vector ─────────────────────

    template <typename ModelT>
    void apply_initial_velocity(Vec v_local, ModelT& model) const
    {
        for (const auto& ic : initial_conditions_) {
            const auto& node = model.get_domain().node(ic.node_id);
            auto num_dofs = static_cast<PetscInt>(node.num_dof());
            const auto* dofs = node.dof_data();

            std::array<PetscScalar, dim> vals{};
            for (std::size_t i = 0; i < dim; ++i)
                vals[i] = ic.velocity[i];

            VecSetValuesLocal(v_local, num_dofs, dofs, vals.data(), INSERT_VALUES);
        }

        VecAssemblyBegin(v_local);
        VecAssemblyEnd(v_local);
    }



};


#endif // FN_BOUNDARY_CONDITION_HH
