#ifndef FALL_N_BEAM_SECTION_FORCES_HH
#define FALL_N_BEAM_SECTION_FORCES_HH

// =============================================================================
//  BeamSectionForces<N>
// =============================================================================
//
//  Represents the internal force resultants at a beam cross-section.
//  These are the energy-conjugate quantities to BeamGeneralizedStrain<N,Dim>.
//
//   3D Timoshenko beam  (N=6):
//     s = { N,  M_y,  M_z,  V_y,  V_z,  T }
//          axial  bend_y  bend_z  shear_y  shear_z  torsion
//
//   2D Timoshenko beam  (N=3):
//     s = { N,  M,  V }
//          axial  moment  shear
//
//   Euler-Bernoulli 2D  (N=2):
//     s = { N,  M }
//          axial  moment
//
//  This type satisfies the TensionalConjugate concept defined in
//  ConstitutiveRelation.hh.
//
//  Provides set_components() for constitutive-relation output assignment,
//  and named accessors for physics-level readability.
//
// =============================================================================

#include <cstddef>
#include <Eigen/Dense>

template <std::size_t N>
    requires (N > 0)
class BeamSectionForces {

public:
    static constexpr std::size_t num_components{N};

    using VectorT = Eigen::Vector<double, N>;

private:
    VectorT components_ = VectorT::Zero();

public:
    // --- Component access (const) --------------------------------------------

    constexpr double  operator[](std::size_t i) const { return components_[i]; }
    constexpr double& operator[](std::size_t i)       { return components_[i]; }

    Eigen::Ref<const VectorT> components() const &  { return components_; }
    Eigen::Ref<const VectorT> components() const && = delete;

    const double* data() const &  { return components_.data(); }
    const double* data() const && = delete;

    // --- Component write -----------------------------------------------------

    template <typename Derived>
    constexpr void set_components(const Eigen::MatrixBase<Derived>& v) {
        components_ = v;
    }

    // --- Named accessors (physical meaning) ----------------------------------
    //
    //  3D Timoshenko (N=6):
    //    [0] axial_force     N       [3] shear_force_y  V_y
    //    [1] moment_y        M_y     [4] shear_force_z  V_z
    //    [2] moment_z        M_z     [5] torque         T
    //
    //  2D Timoshenko (N=3):
    //    [0] axial_force     N
    //    [1] moment          M
    //    [2] shear_force     V

    constexpr double axial_force()   const requires (N >= 2) { return components_[0]; }

    constexpr double moment_y()      const requires (N == 6) { return components_[1]; }
    constexpr double moment_z()      const requires (N == 6) { return components_[2]; }
    constexpr double shear_force_y() const requires (N == 6) { return components_[3]; }
    constexpr double shear_force_z() const requires (N == 6) { return components_[4]; }
    constexpr double torque()        const requires (N == 6) { return components_[5]; }

    constexpr double moment()        const requires (N == 3) { return components_[1]; }
    constexpr double shear_force()   const requires (N == 3) { return components_[2]; }

    // --- Constructors --------------------------------------------------------

    constexpr BeamSectionForces() = default;
    constexpr ~BeamSectionForces() = default;

    constexpr BeamSectionForces(const BeamSectionForces&) = default;
    constexpr BeamSectionForces(BeamSectionForces&&) noexcept = default;
    constexpr BeamSectionForces& operator=(const BeamSectionForces&) = default;
    constexpr BeamSectionForces& operator=(BeamSectionForces&&) noexcept = default;

    // Variadic constructor for direct initialization
    template <typename... S>
        requires (sizeof...(S) == N && (std::convertible_to<S, double> && ...))
    constexpr BeamSectionForces(S... s) : components_{ static_cast<double>(s)... } {}
};


// =============================================================================
//  Convenience aliases
// =============================================================================

using TimoshenkoBeamForces3D      = BeamSectionForces<6>;
using TimoshenkoBeamForces2D      = BeamSectionForces<3>;
using EulerBernoulliBeamForces2D  = BeamSectionForces<2>;


#endif // FALL_N_BEAM_SECTION_FORCES_HH
