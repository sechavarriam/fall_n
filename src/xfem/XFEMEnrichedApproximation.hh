#ifndef FALL_N_XFEM_ENRICHED_APPROXIMATION_HH
#define FALL_N_XFEM_ENRICHED_APPROXIMATION_HH

#include "CrackKinematics.hh"
#include "XFEMEnrichment.hh"

#include <Eigen/Dense>

#include <cstdint>
#include <span>
#include <stdexcept>

namespace fall_n::xfem {

struct ShiftedHeavisideKinematics {
    Eigen::Vector3d displacement = Eigen::Vector3d::Zero();
    Eigen::Matrix3d displacement_gradient = Eigen::Matrix3d::Zero();
    Eigen::Matrix<double, 6, 1> engineering_strain =
        Eigen::Matrix<double, 6, 1>::Zero();
    Eigen::Vector3d crack_jump = Eigen::Vector3d::Zero();
    Eigen::Vector3d tangential_jump = Eigen::Vector3d::Zero();
    double normal_opening{0.0};
    std::size_t enriched_node_count{0};
};

[[nodiscard]] inline Eigen::Matrix<double, 6, 1>
engineering_strain_from_displacement_gradient(
    const Eigen::Matrix3d& grad_u) noexcept
{
    Eigen::Matrix<double, 6, 1> strain;
    strain << grad_u(0, 0),
              grad_u(1, 1),
              grad_u(2, 2),
              grad_u(1, 2) + grad_u(2, 1),
              grad_u(0, 2) + grad_u(2, 0),
              grad_u(0, 1) + grad_u(1, 0);
    return strain;
}

inline void require_shifted_heaviside_input_sizes(
    std::span<const double> shape_values,
    std::span<const Eigen::Vector3d> shape_gradients,
    std::span<const Eigen::Vector3d> node_positions,
    std::span<const Eigen::Vector3d> standard_dofs,
    std::span<const Eigen::Vector3d> enriched_dofs,
    std::span<const std::uint8_t> enriched_flags)
{
    const std::size_t n = shape_values.size();
    if (shape_gradients.size() != n || node_positions.size() != n ||
        standard_dofs.size() != n || enriched_dofs.size() != n ||
        enriched_flags.size() != n) {
        throw std::invalid_argument(
            "XFEM shifted-Heaviside kinematics requires all nodal arrays to "
            "have the same local size.");
    }
}

[[nodiscard]] inline ShiftedHeavisideKinematics
evaluate_shifted_heaviside_kinematics(
    std::span<const double> shape_values,
    std::span<const Eigen::Vector3d> shape_gradients,
    std::span<const Eigen::Vector3d> node_positions,
    std::span<const Eigen::Vector3d> standard_dofs,
    std::span<const Eigen::Vector3d> enriched_dofs,
    std::span<const std::uint8_t> enriched_flags,
    const PlaneCrackLevelSet& crack,
    const Eigen::Vector3d& evaluation_point,
    double tolerance = 1.0e-12)
{
    require_shifted_heaviside_input_sizes(
        shape_values,
        shape_gradients,
        node_positions,
        standard_dofs,
        enriched_dofs,
        enriched_flags);

    ShiftedHeavisideKinematics out{};
    const double h_x =
        signed_heaviside(crack.signed_distance(evaluation_point), tolerance);

    for (std::size_t a = 0; a < shape_values.size(); ++a) {
        const double Na = shape_values[a];
        const Eigen::Vector3d& grad_Na = shape_gradients[a];
        const Eigen::Vector3d& ua = standard_dofs[a];

        out.displacement += Na * ua;
        out.displacement_gradient += ua * grad_Na.transpose();

        if (enriched_flags[a] == 0) {
            continue;
        }

        const Eigen::Vector3d& aa = enriched_dofs[a];
        const double h_node =
            signed_heaviside(crack.signed_distance(node_positions[a]), tolerance);
        const double psi = h_x - h_node;

        // Bulk contribution away from the crack surface. The distributional
        // gradient of H(phi) is handled by the interface/cohesive term, not by
        // the regular volumetric strain operator.
        out.displacement += Na * psi * aa;
        out.displacement_gradient += psi * aa * grad_Na.transpose();

        // Trace jump at a point on the discontinuity:
        // [[u]] = (H+ - H-) sum_I N_I a_I = 2 sum_I N_I a_I.
        out.crack_jump += 2.0 * Na * aa;
        ++out.enriched_node_count;
    }

    out.engineering_strain =
        engineering_strain_from_displacement_gradient(out.displacement_gradient);
    const auto split = split_crack_jump(crack.normal, out.crack_jump);
    out.normal_opening = split.normal_opening;
    out.tangential_jump = split.tangential_jump;
    return out;
}

} // namespace fall_n::xfem

#endif // FALL_N_XFEM_ENRICHED_APPROXIMATION_HH
