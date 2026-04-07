#ifndef FALL_N_SRC_RECONSTRUCTION_HOMOGENISATION_STRATEGY_HH
#define FALL_N_SRC_RECONSTRUCTION_HOMOGENISATION_STRATEGY_HH

// =============================================================================
//  HomogenisationStrategy -- legacy/internal upscaling selector
// =============================================================================
//
//  This file is retained as a validation utility for comparing force
//  extraction operators. It is not part of the normative public multiscale
//  API: the production FE2 path uses boundary reactions directly through
//  SectionHomogenizedResponse and reports the operator explicitly.
//
//  Concrete strategies:
//    - BoundaryReactionHomogenisation: f_int reactions at Face B (robust)
//    - VolumeAverageHomogenisation:    sigma·dA integral (comparison path)
//
//  The interface is intentionally independent of any solver class to
//  avoid circular header dependencies.
//
// =============================================================================

#include <string>

#include <Eigen/Dense>

namespace fall_n {

struct HomogenisationStrategy {
    virtual ~HomogenisationStrategy() = default;

    virtual Eigen::Vector<double, 6> select_forces(
        const Eigen::Vector<double, 6>& boundary_reaction,
        const Eigen::Vector<double, 6>& volume_average) const = 0;

    virtual std::string name() const = 0;
};

struct BoundaryReactionHomogenisation final : HomogenisationStrategy {

    Eigen::Vector<double, 6> select_forces(
        const Eigen::Vector<double, 6>& boundary_reaction,
        const Eigen::Vector<double, 6>& /*volume_average*/) const override
    {
        return boundary_reaction;
    }

    std::string name() const override { return "boundary-reaction"; }
};

struct VolumeAverageHomogenisation final : HomogenisationStrategy {

    Eigen::Vector<double, 6> select_forces(
        const Eigen::Vector<double, 6>& /*boundary_reaction*/,
        const Eigen::Vector<double, 6>& volume_average) const override
    {
        return volume_average;
    }

    std::string name() const override { return "volume-average"; }
};

}  // namespace fall_n

#endif // FALL_N_SRC_RECONSTRUCTION_HOMOGENISATION_STRATEGY_HH
