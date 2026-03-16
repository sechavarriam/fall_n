#ifndef FALL_N_FINITE_ELEMENT_CONCEPT_HH
#define FALL_N_FINITE_ELEMENT_CONCEPT_HH

// =============================================================================
//  FiniteElement — Assembly-level concept for all finite element types
// =============================================================================
//
//  Minimal interface consumed by the global assembly / nonlinear-solve loop.
//
//  Consumers:
//    • Model<…>              — inject_K, set_num_dof_in_nodes
//    • NonlinearAnalysis<…>  — compute_internal_forces, inject_tangent_stiffness,
//                              commit_material_state
//
//  What is intentionally excluded:
//    • B(), H()                 — internal kinematics, each element type owns these
//    • material_points()        — post-processing, depends on MaterialPolicy
//    • set_material_point_state — ditto
//
//  Each concrete element type defines its own kinematic operators:
//
//    ContinuumElement   →  B = ∇ˢ (symmetric gradient, Voigt notation)
//    BeamElement         →  B = beam kinematics (axial + bending + shear + twist)
//    ShellElement        →  B = membrane + bending  (future)
//    ContinuumMixed      →  separate B for u and p  (future)
//
//  The solver/assembler only sees the inject/compute interface below.
//
// =============================================================================

#include <concepts>
#include <cstddef>
#include <petsc.h>

template <typename E>
concept FiniteElement = requires(E e, E const ce, Mat K, Vec u, Vec f) {

    // ── Topology (const queries) ──────────────────────────────────
    { ce.num_nodes()              } -> std::convertible_to<std::size_t>;
    { ce.num_integration_points() } -> std::convertible_to<std::size_t>;
    { ce.sieve_id()               } -> std::convertible_to<PetscInt>;

    // ── DOF setup ─────────────────────────────────────────────────
    e.set_num_dof_in_nodes();

    // ── Linear assembly ───────────────────────────────────────────
    e.inject_K(K);

    // ── Nonlinear assembly (SNES callbacks) ───────────────────────
    e.compute_internal_forces(u, f);
    e.inject_tangent_stiffness(u, K);
    e.commit_material_state(u);
    e.revert_material_state();
};

#endif // FALL_N_FINITE_ELEMENT_CONCEPT_HH
