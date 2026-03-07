#ifndef FALL_N_SHELL_RESULTANTS_HH
#define FALL_N_SHELL_RESULTANTS_HH

// =============================================================================
//  ShellResultants<N>
// =============================================================================
//
//  Represents the internal force/moment resultants at a shell mid-surface point.
//  These are the energy-conjugate quantities to ShellGeneralizedStrain<N,Dim>.
//
//   Mindlin-Reissner shell (N=8):
//     S = { N₁₁, N₂₂, N₁₂,   M₁₁, M₂₂, M₁₂,   Q₁, Q₂ }
//           membrane forces   | bending moments  | trans. shear forces
//
//  This type satisfies the TensionalConjugate concept defined in
//  ConstitutiveRelation.hh.
//
// =============================================================================

#include <cstddef>
#include <type_traits>
#include <Eigen/Dense>

template <std::size_t N>
    requires (N > 0)
class ShellResultants {

public:
    static constexpr std::size_t num_components{N};

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
    //  [0] N₁₁   membrane force 11     [3] M₁₁   bending moment 11
    //  [1] N₂₂   membrane force 22     [4] M₂₂   bending moment 22
    //  [2] N₁₂   membrane force 12     [5] M₁₂   twisting moment
    //                                   [6] Q₁    transverse shear force 1
    //                                   [7] Q₂    transverse shear force 2

    constexpr double membrane_force_11()   const requires (N == 8) { return components_[0]; }
    constexpr double membrane_force_22()   const requires (N == 8) { return components_[1]; }
    constexpr double membrane_force_12()   const requires (N == 8) { return components_[2]; }
    constexpr double bending_moment_11()   const requires (N == 8) { return components_[3]; }
    constexpr double bending_moment_22()   const requires (N == 8) { return components_[4]; }
    constexpr double bending_moment_12()   const requires (N == 8) { return components_[5]; }
    constexpr double shear_force_1()       const requires (N == 8) { return components_[6]; }
    constexpr double shear_force_2()       const requires (N == 8) { return components_[7]; }

    // --- Constructors --------------------------------------------------------

    constexpr ShellResultants() = default;
    constexpr ~ShellResultants() = default;

    constexpr ShellResultants(const ShellResultants&) = default;
    constexpr ShellResultants(ShellResultants&&) noexcept = default;
    constexpr ShellResultants& operator=(const ShellResultants&) = default;
    constexpr ShellResultants& operator=(ShellResultants&&) noexcept = default;

    template <typename... S>
        requires (sizeof...(S) == N && (std::convertible_to<S, double> && ...))
    constexpr ShellResultants(S... s) : components_{ static_cast<double>(s)... } {}
};


// =============================================================================
//  Convenience aliases
// =============================================================================

using MindlinReissnerShellResultants = ShellResultants<8>;


#endif // FALL_N_SHELL_RESULTANTS_HH
