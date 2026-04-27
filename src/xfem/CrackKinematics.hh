#ifndef FALL_N_XFEM_CRACK_KINEMATICS_HH
#define FALL_N_XFEM_CRACK_KINEMATICS_HH

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fall_n::xfem {

struct CrackJumpKinematics {
    Eigen::Vector3d normal = Eigen::Vector3d::UnitX();
    Eigen::Vector3d jump = Eigen::Vector3d::Zero();
    double normal_opening{0.0};
    Eigen::Vector3d tangential_jump = Eigen::Vector3d::Zero();
};

[[nodiscard]] inline Eigen::Vector3d normalized_crack_normal(
    const Eigen::Vector3d& crack_normal)
{
    const double n_norm = crack_normal.norm();
    if (n_norm <= 1.0e-14) {
        throw std::invalid_argument("Crack kinematics requires a non-zero normal.");
    }
    return crack_normal / n_norm;
}

[[nodiscard]] inline Eigen::Matrix3d normal_projector(
    const Eigen::Vector3d& crack_normal)
{
    const Eigen::Vector3d n = normalized_crack_normal(crack_normal);
    return n * n.transpose();
}

[[nodiscard]] inline Eigen::Matrix3d tangential_projector(
    const Eigen::Vector3d& crack_normal)
{
    return Eigen::Matrix3d::Identity() - normal_projector(crack_normal);
}

[[nodiscard]] inline CrackJumpKinematics split_crack_jump(
    const Eigen::Vector3d& crack_normal,
    const Eigen::Vector3d& jump)
{
    CrackJumpKinematics out{};
    out.normal = normalized_crack_normal(crack_normal);
    out.jump = jump;
    out.normal_opening = out.normal.dot(jump);
    out.tangential_jump = jump - out.normal_opening * out.normal;
    return out;
}

[[nodiscard]] inline double effective_mixed_mode_separation(
    double normal_opening,
    const Eigen::Vector3d& tangential_jump,
    double mode_mixity_weight)
{
    const double opening = std::max(normal_opening, 0.0);
    const double slip = tangential_jump.norm();
    return std::sqrt(opening * opening +
                     mode_mixity_weight * mode_mixity_weight * slip * slip);
}

[[nodiscard]] inline double crack_band_opening_mm(
    double equivalent_crack_strain,
    double characteristic_length_mm) noexcept
{
    return std::max(equivalent_crack_strain, 0.0) *
           std::max(characteristic_length_mm, 0.0);
}

} // namespace fall_n::xfem

#endif // FALL_N_XFEM_CRACK_KINEMATICS_HH
