#ifndef FALL_N_BEAM_GENERALIZED_STRAIN_HH
#define FALL_N_BEAM_GENERALIZED_STRAIN_HH

// =============================================================================
//  BeamGeneralizedStrain<N, Dim>
// =============================================================================
//
//  Represents the generalized strain vector at a beam cross-section.
//  Unlike VoigtVector (which encodes symmetric tensor components in Voigt
//  notation), this class stores *heterogeneous* kinematic quantities that
//  arise from beam kinematics:
//
//   3D Timoshenko beam (N=6, Dim=3):
//     e = { ε,  κ_y,  κ_z,  γ_y,  γ_z,  θ' }
//          axial  bend_y  bend_z  shear_y  shear_z  twist
//
//   2D Timoshenko beam (N=3, Dim=2):
//     e = { ε,  κ,  γ }
//          axial  bending  shear
//
//   Euler-Bernoulli 2D (N=2, Dim=2):
//     e = { ε,  κ }
//          axial  bending
//
//  This type satisfies the KinematicMeasure concept defined in
//  ConstitutiveRelation.hh.
//
//  Design rationale:
//  - NOT derived from VoigtVector: these are not symmetric tensor components.
//  - Uses Eigen::Vector internally for seamless integration with Eigen
//    matrix operations (tangent * strain → forces).
//  - Provides the same structural interface as VoigtVector (operator[],
//    components(), set_components(), data()) so that generic algorithms
//    work uniformly on both solids and beams.
//
// =============================================================================

#include <cstddef>
#include <Eigen/Dense>

template <std::size_t N, std::size_t Dim> requires (N > 0 && Dim > 0 && Dim <= 3)
class BeamGeneralizedStrain {

public:
    static constexpr std::size_t num_components{N};
    static constexpr std::size_t dim{Dim};

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
    //  These are provided for readability at call sites. Their availability
    //  depends on N; they are always safe because the template constraint
    //  limits N to specific values.
    //
    //  3D Timoshenko (N=6):
    //    [0] axial_strain   ε       [3] shear_strain_y  γ_y
    //    [1] curvature_y    κ_y     [4] shear_strain_z  γ_z
    //    [2] curvature_z    κ_z     [5] twist_rate      θ'
    //
    //  2D Timoshenko (N=3):
    //    [0] axial_strain   ε
    //    [1] curvature      κ
    //    [2] shear_strain   γ

    constexpr double axial_strain() const requires (N >= 2) { return components_[0]; }

    constexpr double curvature_y()  const requires (N == 6) { return components_[1]; }
    constexpr double curvature_z()  const requires (N == 6) { return components_[2]; }
    constexpr double shear_y()      const requires (N == 6) { return components_[3]; }
    constexpr double shear_z()      const requires (N == 6) { return components_[4]; }
    constexpr double twist_rate()   const requires (N == 6) { return components_[5]; }

    constexpr double curvature()    const requires (N == 3) { return components_[1]; }
    constexpr double shear()        const requires (N == 3) { return components_[2]; }

    // --- Constructors --------------------------------------------------------

    constexpr BeamGeneralizedStrain() = default;
    constexpr ~BeamGeneralizedStrain() = default;

    constexpr BeamGeneralizedStrain(const BeamGeneralizedStrain&) = default;
    constexpr BeamGeneralizedStrain(BeamGeneralizedStrain&&) noexcept = default;
    constexpr BeamGeneralizedStrain& operator=(const BeamGeneralizedStrain&) = default;
    constexpr BeamGeneralizedStrain& operator=(BeamGeneralizedStrain&&) noexcept = default;

    // Variadic constructor for direct initialization
    //   e.g. BeamGeneralizedStrain<6,3> e{eps, ky, kz, gy, gz, tw};
    template <typename... S>
        requires (sizeof...(S) == N && (std::convertible_to<S, double> && ...))
    constexpr BeamGeneralizedStrain(S... s) : components_{ static_cast<double>(s)... } {}
};


// =============================================================================
//  Convenience aliases for common beam types
// =============================================================================

// 3D Timoshenko beam: 6 generalized strains in 3D space
//   { ε, κ_y, κ_z, γ_y, γ_z, θ' }
using TimoshenkoBeamStrain3D = BeamGeneralizedStrain<6, 3>;

// 2D Timoshenko beam: 3 generalized strains in 2D space
//   { ε, κ, γ }
using TimoshenkoBeamStrain2D = BeamGeneralizedStrain<3, 2>;

// 2D Euler-Bernoulli beam: 2 generalized strains in 2D space (no shear)
//   { ε, κ }
using EulerBernoulliBeamStrain2D = BeamGeneralizedStrain<2, 2>;


#endif // FALL_N_BEAM_GENERALIZED_STRAIN_HH
