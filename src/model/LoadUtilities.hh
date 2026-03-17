#ifndef FN_LOAD_UTILITIES_HH
#define FN_LOAD_UTILITIES_HH

// =============================================================================
//  LoadUtilities.hh — Common load-application helpers for structural models
// =============================================================================

#include "../elements/element_geometry/ElementGeometry.hh"

#include <Eigen/Dense>
#include <vector>

namespace fall_n {

/// Apply a uniform surface traction to a set of shell element geometries.
///
/// Integrates  ∫_Ω N_a(ξ) · t · dA  via the element's own quadrature rule,
/// producing consistent nodal forces that are added to the model's force
/// vector through `apply_node_force()`.
///
/// @param model             Structural model with `apply_node_force()`
/// @param shell_geometries  Pointers to the shell ElementGeometry objects
/// @param traction          Uniform traction vector (force/area), world frame
template <typename ModelT>
void apply_uniform_shell_surface_load(
    ModelT& model,
    const std::vector<const ElementGeometry<3>*>& shell_geometries,
    const Eigen::Vector3d& traction)
{
    for (const auto* geom : shell_geometries) {
        std::vector<Eigen::Vector3d> nodal_forces(
            geom->num_nodes(), Eigen::Vector3d::Zero());

        for (std::size_t gp = 0; gp < geom->num_integration_points(); ++gp) {
            const auto xi = geom->reference_integration_point(gp);
            const double wdA = geom->weight(gp) * geom->differential_measure(xi);

            for (std::size_t a = 0; a < geom->num_nodes(); ++a) {
                nodal_forces[a] += geom->H(a, xi) * traction * wdA;
            }
        }

        for (std::size_t a = 0; a < geom->num_nodes(); ++a) {
            model.apply_node_force(
                geom->node(a),
                nodal_forces[a][0],
                nodal_forces[a][1],
                nodal_forces[a][2],
                0.0, 0.0, 0.0);
        }
    }
}

} // namespace fall_n

#endif // FN_LOAD_UTILITIES_HH
