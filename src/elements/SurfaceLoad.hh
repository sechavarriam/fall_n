#ifndef FN_SURFACE_LOAD_HH
#define FN_SURFACE_LOAD_HH

// ═══════════════════════════════════════════════════════════════════════════════
//  SurfaceLoad — Consistent nodal forces from distributed surface traction
// ═══════════════════════════════════════════════════════════════════════════════
//
//  Given a uniform traction vector  t ∈ ℝ^dim  on a boundary surface Γ,
//  the consistent nodal force vector is:
//
//      f_I = ∫_Γ  N_I(ξ,η) · t · dA
//
//  where N_I are the (scalar) shape functions of the surface element and
//  dA = ‖∂x/∂ξ × ∂x/∂η‖ dξ dη  is the physical area element.
//
//  This is computed via Gauss quadrature on each surface ElementGeometry<3>
//  (e.g., LagrangeElement<3,3,3> = quad9 in 3D space) and assembled into a
//  PETSc local vector through the Node DOF indices.
//
//  Usage:
//
//    // Option 1: Given pre-built surface geometries
//    surface_load::apply_traction(surface_geometries, traction, nodal_forces);
//
//    // Option 2: Integrated into Model (via GmshDomainBuilder)
//    model.apply_surface_traction("Load", 0.0, 0.0, 100.0);
//
// ═══════════════════════════════════════════════════════════════════════════════

#include <array>
#include <cstddef>
#include <span>
#include <vector>

#include <Eigen/Dense>
#include <petsc.h>

#include "element_geometry/ElementGeometry.hh"


namespace surface_load {

// ─────────────────────────────────────────────────────────────────────────────
//  Core: compute consistent nodal force vector for ONE surface element.
//
//  Returns a flat vector of size (dim × num_nodes) with the shape:
//    [ f_{1x}, f_{1y}, f_{1z},  f_{2x}, f_{2y}, f_{2z},  ... ]
//
//  traction is the uniform traction vector  t ∈ ℝ^dim.
// ─────────────────────────────────────────────────────────────────────────────
template <std::size_t dim>
Eigen::VectorXd consistent_nodal_forces(
    const ElementGeometry<dim>& surface_geo,
    const std::array<double, dim>& traction)
{
    const auto n_nodes = surface_geo.num_nodes();
    const auto n_gp    = surface_geo.num_integration_points();
    const auto total   = dim * n_nodes;

    Eigen::VectorXd f_e = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(total));

    for (std::size_t gp = 0; gp < n_gp; ++gp) {
        auto   xi   = surface_geo.reference_integration_point(gp);
        double w    = surface_geo.weight(gp);
        double dA   = surface_geo.differential_measure(xi);   // ‖J₁×J₂‖

        double wdA = w * dA;

        for (std::size_t I = 0; I < n_nodes; ++I) {
            double N_I = surface_geo.H(I, xi);
            for (std::size_t d = 0; d < dim; ++d) {
                f_e(static_cast<Eigen::Index>(I * dim + d)) += N_I * traction[d] * wdA;
            }
        }
    }
    return f_e;
}


// ─────────────────────────────────────────────────────────────────────────────
//  Assemble consistent nodal forces from a collection of surface elements
//  into a PETSc local vector (ADD_VALUES).
//
//  Each surface element's nodes must have their DOF indices already set
//  (i.e., the model's set_sieve_layout + setup must have been called).
// ─────────────────────────────────────────────────────────────────────────────
template <std::size_t dim>
void apply_traction(
    std::span<ElementGeometry<dim>> surface_elements,
    const std::array<double, dim>& traction,
    Vec nodal_forces_local)
{
    for (auto& surf_geo : surface_elements) {
        Eigen::VectorXd f_e = consistent_nodal_forces<dim>(surf_geo, traction);

        const auto n_nodes = surf_geo.num_nodes();

        // Gather DOF indices from surface nodes
        std::vector<PetscInt> dof_indices;
        dof_indices.reserve(dim * n_nodes);
        for (std::size_t I = 0; I < n_nodes; ++I) {
            for (const auto idx : surf_geo.node_p(I).dof_index()) {
                dof_indices.push_back(idx);
            }
        }

        VecSetValuesLocal(nodal_forces_local,
                          static_cast<PetscInt>(dof_indices.size()),
                          dof_indices.data(),
                          f_e.data(),
                          ADD_VALUES);
    }

    VecAssemblyBegin(nodal_forces_local);
    VecAssemblyEnd  (nodal_forces_local);
}


// ─────────────────────────────────────────────────────────────────────────────
//  Convenience: compute total surface area of a set of surface elements.
//  Useful for verification:  Area of face = L^2  for a cube of side L.
// ─────────────────────────────────────────────────────────────────────────────
template <std::size_t dim>
double compute_surface_area(std::span<ElementGeometry<dim>> surface_elements)
{
    double total_area = 0.0;
    for (auto& surf_geo : surface_elements) {
        total_area += surf_geo.integrate([](auto /*xi*/) { return 1.0; });
    }
    return total_area;
}

} // namespace surface_load

#endif // FN_SURFACE_LOAD_HH
