#ifndef FALL_N_POST_PROCESSING_STATE_QUERY_HH
#define FALL_N_POST_PROCESSING_STATE_QUERY_HH

// =============================================================================
//  StateQuery — Lightweight accessors for nodal / element response quantities
// =============================================================================
//
//  Pure function templates that extract response quantities from a Model.
//  Used by observers, recorders, and post-processing code.
//
//  Design: header-only, no state, no allocation (returns small Eigen types
//  or scalars).  All functions take a const model reference so they are safe
//  to call from observers.
//
// =============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>

#include <petsc.h>
#include <Eigen/Dense>


namespace fall_n::query {


// ─────────────────────────────────────────────────────────────────────────────
//  Nodal displacement extraction
// ─────────────────────────────────────────────────────────────────────────────

/// Extract the translational displacement at a node (first 3 DOFs).
template <typename ModelT>
Eigen::Vector3d nodal_translation(const ModelT& model, std::size_t node_id) {
    const auto& node = model.get_domain().node(node_id);
    const auto  dofs = node.dof_index();

    const PetscScalar* values = nullptr;
    VecGetArrayRead(model.state_vector(), &values);

    Eigen::Vector3d u = Eigen::Vector3d::Zero();
    for (std::size_t d = 0; d < 3 && d < dofs.size(); ++d)
        u[static_cast<Eigen::Index>(d)] = values[dofs[d]];

    VecRestoreArrayRead(model.state_vector(), &values);
    return u;
}

/// Extract a single DOF value at a node from a PETSc Vec (typically u or v).
template <typename ModelT>
double nodal_dof_value(const ModelT& model, Vec vec, std::size_t node_id, std::size_t dof) {
    const auto& node = model.get_domain().node(node_id);
    const auto  dofs = node.dof_index();
    if (dof >= dofs.size()) return 0.0;

    PetscScalar val;
    PetscInt idx = dofs[dof];
    VecGetValues(vec, 1, &idx, &val);
    return static_cast<double>(val);
}


// ─────────────────────────────────────────────────────────────────────────────
//  Envelope queries (max over all nodes)
// ─────────────────────────────────────────────────────────────────────────────

/// Maximum Euclidean norm of the translational displacement over all nodes.
template <typename ModelT>
double max_translation_norm(const ModelT& model) {
    double max_u = 0.0;
    for (const auto& node : model.get_domain().nodes())
        max_u = std::max(max_u, nodal_translation(model, node.id()).norm());
    return max_u;
}

/// Maximum absolute value of a single displacement component over all nodes.
template <typename ModelT>
double max_component_abs(const ModelT& model, std::size_t comp) {
    double max_u = 0.0;
    for (const auto& node : model.get_domain().nodes()) {
        max_u = std::max(
            max_u,
            std::abs(nodal_translation(model, node.id())[static_cast<Eigen::Index>(comp)]));
    }
    return max_u;
}


} // namespace fall_n::query

#endif // FALL_N_POST_PROCESSING_STATE_QUERY_HH
