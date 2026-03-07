#ifndef FALL_N_SECTION_GEOMETRY_HH
#define FALL_N_SECTION_GEOMETRY_HH

// ============================================================================
//  SectionGeometry  —  Cross-section geometric description
// ============================================================================
//
//  A SectionGeometry describes the **shape** of a structural member's
//  cross-section (rectangle, circle, I-beam, …).  It supplies geometric
//  properties needed by the section constitutive relation:
//
//    • area()           – A
//    • moment_y()       – I_y  (second moment about local y-axis)
//    • moment_z()       – I_z  (second moment about local z-axis)
//    • torsion_J()      – J    (St-Venant torsional constant)
//    • shear_factor_y() – k_y  (Timoshenko shear correction factor)
//    • shear_factor_z() – k_z
//
//  The concept `SectionGeometryLike` constrains templates that accept
//  any conforming cross-section type.
//
//  The concept hierarchy separates **geometry** from **material**:
//    SectionGeometry knows A, I, J, k  (shape)
//    TimoshenkoBeamSection3D knows E, G (material) — it *uses* a geometry
//
// ============================================================================

#include <concepts>
#include <cmath>
#include <numbers>

namespace section {

// ── Concept ─────────────────────────────────────────────────────────────────

template <typename G>
concept SectionGeometryLike = requires(const G& g) {
    { g.area()           } -> std::convertible_to<double>;
    { g.moment_y()       } -> std::convertible_to<double>;
    { g.moment_z()       } -> std::convertible_to<double>;
    { g.torsion_J()      } -> std::convertible_to<double>;
    { g.shear_factor_y() } -> std::convertible_to<double>;
    { g.shear_factor_z() } -> std::convertible_to<double>;
};

// ============================================================================
//  RectangularSection  —  solid rectangular cross-section (b × h)
// ============================================================================
//
//  Axes convention:  y is the "strong" axis (parallel to b), z is "weak" (parallel to h)
//  when h > b.  The user chooses which is which by passing (b, h).
//
//    I_y = b·h³/12     (bending about y)
//    I_z = h·b³/12     (bending about z)
//    J   ≈ b·h³ · [1/3 – 0.21·(h/b)·(1 – (h/b)⁴/12)]  (St-Venant approximation)
//    k   = 5/6         (Timoshenko shear factor for rectangles)

class RectangularSection {
    double b_;   // width
    double h_;   // height

    double area_;
    double Iy_;
    double Iz_;
    double J_;

    constexpr void compute() noexcept {
        area_ = b_ * h_;
        Iy_   = b_ * h_ * h_ * h_ / 12.0;
        Iz_   = h_ * b_ * b_ * b_ / 12.0;

        // St-Venant torsion for rectangle (a ≥ b convention: a = max, b = min)
        const double a = std::max(b_, h_);
        const double bb = std::min(b_, h_);
        const double r = bb / a;
        J_ = a * bb * bb * bb * (1.0 / 3.0 - 0.21 * r * (1.0 - r * r * r * r / 12.0));
    }

public:
    constexpr double area()           const noexcept { return area_; }
    constexpr double moment_y()       const noexcept { return Iy_; }
    constexpr double moment_z()       const noexcept { return Iz_; }
    constexpr double torsion_J()      const noexcept { return J_; }
    constexpr double shear_factor_y() const noexcept { return 5.0 / 6.0; }
    constexpr double shear_factor_z() const noexcept { return 5.0 / 6.0; }

    // Accessors
    constexpr double width()  const noexcept { return b_; }
    constexpr double height() const noexcept { return h_; }

    constexpr RectangularSection(double b, double h) : b_{b}, h_{h},
        area_{0}, Iy_{0}, Iz_{0}, J_{0}
    {
        compute();
    }

    constexpr RectangularSection() = default;
};

// ============================================================================
//  CircularSection  —  solid circular cross-section (radius r)
// ============================================================================
//
//    A   = π r²
//    I_y = I_z = π r⁴ / 4
//    J   = π r⁴ / 2             (polar moment = 2·I for circle)
//    k   ≈ 6(1+ν)/(7+6ν) ≈ 9/10 for ν=0.3; we use 9/10 as default

class CircularSection {
    double r_;

    double area_;
    double I_;
    double J_;

    constexpr void compute() noexcept {
        const double r2 = r_ * r_;
        const double r4 = r2 * r2;
        area_ = std::numbers::pi * r2;
        I_    = std::numbers::pi * r4 / 4.0;
        J_    = std::numbers::pi * r4 / 2.0;
    }

public:
    constexpr double area()           const noexcept { return area_; }
    constexpr double moment_y()       const noexcept { return I_; }
    constexpr double moment_z()       const noexcept { return I_; }
    constexpr double torsion_J()      const noexcept { return J_; }
    constexpr double shear_factor_y() const noexcept { return 9.0 / 10.0; }
    constexpr double shear_factor_z() const noexcept { return 9.0 / 10.0; }

    constexpr double radius() const noexcept { return r_; }

    constexpr CircularSection(double r) : r_{r}, area_{0}, I_{0}, J_{0} {
        compute();
    }

    constexpr CircularSection() = default;
};

// ============================================================================
//  GenericSection  —  user-supplied properties (no shape assumptions)
// ============================================================================
//
//  For I-beams, L-shapes, etc., where the user provides pre-computed
//  geometric properties directly.

class GenericSection {
    double A_{0};
    double Iy_{0};
    double Iz_{0};
    double J_{0};
    double ky_{5.0 / 6.0};
    double kz_{5.0 / 6.0};

public:
    constexpr double area()           const noexcept { return A_; }
    constexpr double moment_y()       const noexcept { return Iy_; }
    constexpr double moment_z()       const noexcept { return Iz_; }
    constexpr double torsion_J()      const noexcept { return J_; }
    constexpr double shear_factor_y() const noexcept { return ky_; }
    constexpr double shear_factor_z() const noexcept { return kz_; }

    constexpr GenericSection(double A, double Iy, double Iz, double J,
                             double ky = 5.0 / 6.0, double kz = 5.0 / 6.0)
        : A_{A}, Iy_{Iy}, Iz_{Iz}, J_{J}, ky_{ky}, kz_{kz} {}

    constexpr GenericSection() = default;
};

// ── Static assertions ───────────────────────────────────────────────────────

static_assert(SectionGeometryLike<RectangularSection>,
              "RectangularSection must satisfy SectionGeometryLike.");
static_assert(SectionGeometryLike<CircularSection>,
              "CircularSection must satisfy SectionGeometryLike.");
static_assert(SectionGeometryLike<GenericSection>,
              "GenericSection must satisfy SectionGeometryLike.");

} // namespace section

#endif // FALL_N_SECTION_GEOMETRY_HH
