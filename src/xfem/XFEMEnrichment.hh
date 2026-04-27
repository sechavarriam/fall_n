#ifndef FALL_N_XFEM_ENRICHMENT_HH
#define FALL_N_XFEM_ENRICHMENT_HH

#include <Eigen/Dense>

#include <algorithm>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

namespace fall_n::xfem {

enum class HeavisideSide {
    negative = -1,
    on_interface = 0,
    positive = 1
};

[[nodiscard]] constexpr double signed_heaviside(
    double phi,
    double tolerance = 1.0e-12) noexcept
{
    if (phi > tolerance) {
        return 1.0;
    }
    if (phi < -tolerance) {
        return -1.0;
    }
    return 0.0;
}

[[nodiscard]] constexpr HeavisideSide side_of(
    double phi,
    double tolerance = 1.0e-12) noexcept
{
    if (phi > tolerance) {
        return HeavisideSide::positive;
    }
    if (phi < -tolerance) {
        return HeavisideSide::negative;
    }
    return HeavisideSide::on_interface;
}

struct PlaneCrackLevelSet {
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
    Eigen::Vector3d normal = Eigen::Vector3d::UnitX();

    PlaneCrackLevelSet() = default;

    PlaneCrackLevelSet(Eigen::Vector3d point_on_crack,
                       Eigen::Vector3d crack_normal)
        : point{std::move(point_on_crack)}
    {
        const double n = crack_normal.norm();
        if (n <= 1.0e-14) {
            throw std::invalid_argument(
                "PlaneCrackLevelSet requires a non-zero crack normal.");
        }
        normal = crack_normal / n;
    }

    [[nodiscard]] double signed_distance(
        const Eigen::Vector3d& x) const noexcept
    {
        return normal.dot(x - point);
    }

    [[nodiscard]] HeavisideSide side(
        const Eigen::Vector3d& x,
        double tolerance = 1.0e-12) const noexcept
    {
        return side_of(signed_distance(x), tolerance);
    }
};

struct ShiftedHeavisideEnrichment {
    PlaneCrackLevelSet crack{};
    double tolerance{1.0e-12};

    [[nodiscard]] double operator()(
        const Eigen::Vector3d& x,
        const Eigen::Vector3d& enriched_node_position) const noexcept
    {
        // Shifted enrichment preserves the ordinary nodal interpolation value:
        // psi_I(x) = H(phi(x)) - H(phi(x_I)).
        return signed_heaviside(crack.signed_distance(x), tolerance) -
               signed_heaviside(
                   crack.signed_distance(enriched_node_position),
                   tolerance);
    }
};

[[nodiscard]] inline bool element_is_cut_by_crack(
    std::span<const Eigen::Vector3d> nodes,
    std::span<const std::size_t> connectivity,
    const PlaneCrackLevelSet& crack,
    double tolerance = 1.0e-12)
{
    bool has_positive = false;
    bool has_negative = false;
    bool has_interface_node = false;

    for (const std::size_t node_id : connectivity) {
        if (node_id >= nodes.size()) {
            throw std::out_of_range(
                "XFEM connectivity references a node outside the coordinate array.");
        }
        switch (crack.side(nodes[node_id], tolerance)) {
            case HeavisideSide::positive:
                has_positive = true;
                break;
            case HeavisideSide::negative:
                has_negative = true;
                break;
            case HeavisideSide::on_interface:
                has_interface_node = true;
                break;
        }
    }

    return (has_positive && has_negative) ||
           (has_interface_node && (has_positive || has_negative));
}

[[nodiscard]] inline std::vector<bool> mark_heaviside_enriched_nodes(
    std::span<const Eigen::Vector3d> nodes,
    std::span<const std::vector<std::size_t>> element_connectivity,
    const PlaneCrackLevelSet& crack,
    double tolerance = 1.0e-12)
{
    std::vector<bool> enriched(nodes.size(), false);

    for (const auto& element : element_connectivity) {
        if (!element_is_cut_by_crack(
                nodes,
                std::span<const std::size_t>{element.data(), element.size()},
                crack,
                tolerance)) {
            continue;
        }
        for (const std::size_t node_id : element) {
            if (node_id >= enriched.size()) {
                throw std::out_of_range(
                    "XFEM connectivity references a node outside the enrichment mask.");
            }
            enriched[node_id] = true;
        }
    }

    return enriched;
}

} // namespace fall_n::xfem

#endif // FALL_N_XFEM_ENRICHMENT_HH
