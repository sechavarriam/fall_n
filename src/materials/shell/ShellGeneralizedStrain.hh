#ifndef FALL_N_SHELL_GENERALIZED_STRAIN_HH
#define FALL_N_SHELL_GENERALIZED_STRAIN_HH

// =============================================================================
//  ShellGeneralizedStrain<N, Dim>
// =============================================================================
//
//  Represents the generalized strain vector at a shell mid-surface point.
//  These are kinematic quantities arising from Mindlin-Reissner shell theory.
//
//   Mindlin-Reissner shell in 3D (N=8, Dim=3):
//     e = { ε₁₁, ε₂₂, γ₁₂,   κ₁₁, κ₂₂, κ₁₂,   γ₁₃, γ₂₃ }
//           membrane (3)     |  bending (3)      | trans. shear (2)
//
//     Membrane:   ε_αβ  = ½(∂u_α/∂x_β + ∂u_β/∂x_α)
//     Bending:    κ_αβ  = ½(∂β_α/∂x_β + ∂β_β/∂x_α)
//     Shear:      γ_α3  = ∂w/∂x_α + β_α
//
//     where β₁ = θ₂,  β₂ = −θ₁  (right-hand rule convention).
//
//  This type satisfies the KinematicMeasure concept defined in
//  ConstitutiveRelation.hh.
//
// =============================================================================

#include <cstddef>
#include <type_traits>
#include <Eigen/Dense>

template <std::size_t N, std::size_t Dim>
    requires (N > 0 && Dim > 0 && Dim <= 3)
class ShellGeneralizedStrain {

public:
    static constexpr std::size_t num_components{N};
    static constexpr std::size_t dim{Dim};

    using VectorT = Eigen::Vector<double, N>;

private:
    VectorT components_ = VectorT::Zero();

public:
    // --- Component access ----------------------------------------------------

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

    // --- Named accessors (Mindlin-Reissner, N=8) -----------------------------
    //
    //  [0] ε₁₁  membrane strain 11
    //  [1] ε₂₂  membrane strain 22
    //  [2] γ₁₂  membrane shear strain
    //  [3] κ₁₁  curvature 11
    //  [4] κ₂₂  curvature 22
    //  [5] κ₁₂  twist curvature
    //  [6] γ₁₃  transverse shear strain 13
    //  [7] γ₂₃  transverse shear strain 23

    constexpr double membrane_11()    const requires (N == 8) { return components_[0]; }
    constexpr double membrane_22()    const requires (N == 8) { return components_[1]; }
    constexpr double membrane_12()    const requires (N == 8) { return components_[2]; }
    constexpr double curvature_11()   const requires (N == 8) { return components_[3]; }
    constexpr double curvature_22()   const requires (N == 8) { return components_[4]; }
    constexpr double curvature_12()   const requires (N == 8) { return components_[5]; }
    constexpr double shear_13()       const requires (N == 8) { return components_[6]; }
    constexpr double shear_23()       const requires (N == 8) { return components_[7]; }

    // --- Constructors --------------------------------------------------------

    constexpr ShellGeneralizedStrain() = default;
    constexpr ~ShellGeneralizedStrain() = default;

    constexpr ShellGeneralizedStrain(const ShellGeneralizedStrain&) = default;
    constexpr ShellGeneralizedStrain(ShellGeneralizedStrain&&) noexcept = default;
    constexpr ShellGeneralizedStrain& operator=(const ShellGeneralizedStrain&) = default;
    constexpr ShellGeneralizedStrain& operator=(ShellGeneralizedStrain&&) noexcept = default;

    template <typename... S>
        requires (sizeof...(S) == N && (std::convertible_to<S, double> && ...))
    constexpr ShellGeneralizedStrain(S... s) : components_{ static_cast<double>(s)... } {}
};


// =============================================================================
//  Convenience aliases
// =============================================================================

// Mindlin-Reissner shell: 8 generalized strains (3 membrane + 3 bending + 2 shear)
using MindlinReissnerShellStrain3D = ShellGeneralizedStrain<8, 3>;


#endif // FALL_N_SHELL_GENERALIZED_STRAIN_HH
