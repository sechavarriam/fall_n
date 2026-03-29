#ifndef FALL_N_SRC_RECONSTRUCTION_HOMOGENISATION_STRATEGY_HH
#define FALL_N_SRC_RECONSTRUCTION_HOMOGENISATION_STRATEGY_HH

// =============================================================================
//  HomogenisationStrategy — upscaling interface for section force extraction
// =============================================================================
//
//  Decouples the force-extraction method from the sub-model solver.
//  Two approaches are provided:
//    - BoundaryReactionHomogenisation: f_int reactions at Face B (default, robust)
//    - VolumeAverageHomogenisation:    stress-volume integral (original approach)
//
// =============================================================================

#include <string>

#include <Eigen/Dense>


namespace fall_n {


struct HomogenisationStrategy {
    virtual ~HomogenisationStrategy() = default;
    virtual Eigen::Vector<double, 6> homogenize_forces(
        class NonlinearSubModelEvolver& evolver,
        double width, double height) const = 0;
    virtual std::string name() const = 0;
};


}  // namespace fall_n

#endif // FALL_N_SRC_RECONSTRUCTION_HOMOGENISATION_STRATEGY_HH
