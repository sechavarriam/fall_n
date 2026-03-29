#ifndef FALL_N_SRC_RECONSTRUCTION_HOMOGENISATION_STRATEGY_HH
#define FALL_N_SRC_RECONSTRUCTION_HOMOGENISATION_STRATEGY_HH

// =============================================================================
//  HomogenisationStrategy — upscaling interface for section force extraction
// =============================================================================
//
//  Decouples the force-extraction method from the sub-model solver.
//  The sub-model solver computes section forces via two independent
//  methods; the strategy selects which result to use.
//
//  Concrete strategies:
//    - BoundaryReactionHomogenisation: f_int reactions at Face B (robust)
//    - VolumeAverageHomogenisation:    σ·dA integral (original approach)
//
//  The interface is intentionally independent of any solver class to
//  avoid circular header dependencies.
//
// =============================================================================

#include <string>

#include <Eigen/Dense>


namespace fall_n {


// =============================================================================
//  HomogenisationStrategy — abstract base
// =============================================================================

struct HomogenisationStrategy {
    virtual ~HomogenisationStrategy() = default;

    /// Select / blend from pre-computed force vectors.
    ///   boundary_reaction: from assembled f_int at Face B nodes
    ///   volume_average:    from σ·dA integration over the cross-section
    virtual Eigen::Vector<double, 6> select_forces(
        const Eigen::Vector<double, 6>& boundary_reaction,
        const Eigen::Vector<double, 6>& volume_average) const = 0;

    virtual std::string name() const = 0;
};


// =============================================================================
//  BoundaryReactionHomogenisation — default strategy
// =============================================================================
//
//  Returns section forces obtained from the boundary reactions at Face B
//  of the sub-model.  This is the most robust approach because it uses
//  the actual equilibrium forces (assembled f_int), which remain
//  consistent with the displacement field even after perturbation solves.

struct BoundaryReactionHomogenisation final : HomogenisationStrategy {

    Eigen::Vector<double, 6> select_forces(
        const Eigen::Vector<double, 6>& boundary_reaction,
        const Eigen::Vector<double, 6>& /*volume_average*/) const override
    {
        return boundary_reaction;
    }

    std::string name() const override { return "boundary-reaction"; }
};


// =============================================================================
//  VolumeAverageHomogenisation — stress-volume integral approach
// =============================================================================
//
//  Returns section forces obtained from integrating the Cauchy stress
//  field over the cross-section: N = ∫σ₁₁ dA, V = ∫τ dA, etc.
//  This is the classical Hill–Mandel approach, but it may lose accuracy
//  when the stored material state is stale (e.g. after perturbation).

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
