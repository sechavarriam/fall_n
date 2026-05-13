#ifndef FALL_N_XFEM_DOF_MANAGER_HH
#define FALL_N_XFEM_DOF_MANAGER_HH

#include <cstddef>
#include <stdexcept>
#include <string>

#include <petscis.h>
#include <petscsection.h>

#include "../domain/Domain.hh"

namespace fall_n::xfem {

// Shifted-Heaviside XFEM stores the ordinary displacement vector first and
// appends one enriched displacement vector per discontinuity plane to each
// enriched host node:
//
//   q_I = [u_x, u_y, u_z, a1_x, a1_y, a1_z, a2_x, ...]
//
// The single-plane helpers below preserve the historical API.  The overloads
// that accept a plane index are used by MultiPlaneXFEM kernels.

template <std::size_t Dim>
struct ShiftedHeavisideDofLayout {
    static constexpr std::size_t standard_dofs = Dim;
    static constexpr std::size_t enriched_offset = Dim;
    static constexpr std::size_t enriched_dofs = Dim;
    static constexpr std::size_t total_dofs = 2 * Dim;
};

template <std::size_t Dim>
[[nodiscard]] constexpr std::size_t
shifted_heaviside_enriched_component(std::size_t displacement_component)
{
    return ShiftedHeavisideDofLayout<Dim>::enriched_offset +
           displacement_component;
}

template <std::size_t Dim>
[[nodiscard]] constexpr std::size_t shifted_heaviside_enriched_component(
    std::size_t plane_index,
    std::size_t displacement_component)
{
    return ShiftedHeavisideDofLayout<Dim>::standard_dofs +
           plane_index * ShiftedHeavisideDofLayout<Dim>::enriched_dofs +
           displacement_component;
}

template <std::size_t Dim>
[[nodiscard]] constexpr std::size_t
shifted_heaviside_total_dofs(std::size_t plane_count)
{
    return ShiftedHeavisideDofLayout<Dim>::standard_dofs +
           plane_count * ShiftedHeavisideDofLayout<Dim>::enriched_dofs;
}

template <std::size_t Dim>
[[nodiscard]] constexpr bool
node_has_shifted_heaviside_enrichment(const Node<Dim>& node) noexcept
{
    return node.num_dof() >= ShiftedHeavisideDofLayout<Dim>::total_dofs;
}

template <std::size_t Dim, typename EnrichedMask>
std::size_t apply_shifted_heaviside_dof_layout(
    Domain<Dim>& domain,
    const EnrichedMask& enriched_mask)
{
    if (enriched_mask.size() != domain.num_nodes()) {
        throw std::invalid_argument(
            "XFEM enriched-node mask size does not match the domain node count.");
    }

    std::size_t enriched_count = 0;
    for (std::size_t node_id = 0; node_id < domain.num_nodes(); ++node_id) {
        if (!static_cast<bool>(enriched_mask[node_id])) {
            continue;
        }

        auto& node = domain.node(node_id);
        if (node.num_dof() < ShiftedHeavisideDofLayout<Dim>::standard_dofs) {
            throw std::runtime_error(
                "XFEM enrichment must be applied after standard element DOFs are declared.");
        }
        if (node.num_dof() < ShiftedHeavisideDofLayout<Dim>::total_dofs) {
            node.set_num_dof(ShiftedHeavisideDofLayout<Dim>::total_dofs);
        }
        ++enriched_count;
    }
    return enriched_count;
}

[[nodiscard]] inline PetscInt petsc_local_dof_index(
    PetscSection local_section,
    PetscInt sieve_point,
    PetscInt component) noexcept
{
    PetscInt local_dof = 0;
    PetscSectionGetDof(local_section, sieve_point, &local_dof);
    if (component < 0 || component >= local_dof) {
        return -1;
    }

    PetscInt local_offset = 0;
    PetscSectionGetOffset(local_section, sieve_point, &local_offset);
    if (local_offset < 0) {
        return -1;
    }
    return local_offset + component;
}

[[nodiscard]] inline PetscInt petsc_global_dof_index(
    PetscSection local_section,
    ISLocalToGlobalMapping local_to_global,
    PetscInt sieve_point,
    PetscInt component) noexcept
{
    const PetscInt local_index =
        petsc_local_dof_index(local_section, sieve_point, component);
    if (local_index < 0) {
        return -1;
    }

    PetscInt global_index = -1;
    ISLocalToGlobalMappingApply(
        local_to_global, 1, &local_index, &global_index);
    return global_index;
}

} // namespace fall_n::xfem

#endif // FALL_N_XFEM_DOF_MANAGER_HH
