#ifndef FN_SECTION_PROPERTIES_HH
#define FN_SECTION_PROPERTIES_HH

// =============================================================================
//  SectionProperties.hh — Pure geometric/mechanical section utilities
// =============================================================================
//
//  Small constexpr helpers for cross-section property calculations.
//  No dependencies beyond <cmath> and <numbers>.
//

#include <cmath>
#include <numbers>

namespace fall_n {

/// Approximate St-Venant torsion constant for a solid rectangular section.
///
/// Uses the classical approximation:
///   J ≈ (b³·h / 3) · (1 − 0.63·b/h)
/// where b = min(width, height) and h = max(width, height).
constexpr double rectangular_torsion_constant(double width,
                                              double height) noexcept {
    const double b_min = std::min(width, height);
    const double h_max = std::max(width, height);
    return (b_min * b_min * b_min * h_max / 3.0)
         * (1.0 - 0.63 * b_min / h_max);
}

/// Area of a circular rebar from its diameter: A = π·d²/4.
constexpr double bar_area(double diameter) noexcept {
    return std::numbers::pi * diameter * diameter / 4.0;
}

/// Isotropic shear modulus: G = E / (2·(1+ν)).
constexpr double isotropic_shear_modulus(double E, double nu) noexcept {
    return E / (2.0 * (1.0 + nu));
}

/// Simplified initial elastic modulus of concrete: Ec = 1000·f'c.
///
/// This is a rough estimate used for shear/torsional stiffness in fiber
/// sections, where fiber axial stiffness governs the flexural response.
constexpr double concrete_initial_modulus(double fpc) noexcept {
    return 1000.0 * fpc;
}

} // namespace fall_n

#endif // FN_SECTION_PROPERTIES_HH
