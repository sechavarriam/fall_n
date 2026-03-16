#ifndef FALL_N_MINDLIN_SHELL_SECTION_HH
#define FALL_N_MINDLIN_SHELL_SECTION_HH

#include "ElasticRelation.hh"

// =============================================================================
//  MindlinShellSection  — Elastic section constitutive for Mindlin-Reissner shell
// =============================================================================
//
//  Derives from ElasticRelation<MindlinReissnerShell3D> and populates the
//  8×8 block-diagonal section stiffness matrix from material and geometric
//  properties (E, ν, t).
//
//  The section stiffness D_s is block-diagonal:
//
//       ┌                  ┐
//       │  A    0      0   │   A  = membrane stiffness    (3×3)
//   D = │  0    D_b    0   │   D_b = bending stiffness   (3×3)
//       │  0    0      S   │   S  = transverse shear     (2×2)
//       └                  ┘
//
//  For isotropic material with thickness t, Young's modulus E, Poisson's ν:
//
//    A = (E·t)/(1−ν²)  · [ 1   ν       0       ]
//                         [ ν   1       0       ]
//                         [ 0   0   (1−ν)/2     ]
//
//    D_b = (E·t³)/(12(1−ν²)) · [ same pattern as A ]
//
//    S = κ·G·t · I₂   where κ = 5/6,  G = E/(2(1+ν))
//
//  Strain ordering:
//    [0] ε₁₁    [3] κ₁₁    [6] γ₁₃
//    [1] ε₂₂    [4] κ₂₂    [7] γ₂₃
//    [2] γ₁₂    [5] κ₁₂
//
// =============================================================================

class MindlinShellSection : public ElasticRelation<MindlinReissnerShell3D> {

public:
    using Base = ElasticRelation<MindlinReissnerShell3D>;
    using Base::KinematicT;
    using Base::ConjugateT;
    using Base::TangentT;

private:
    double E_{0.0};               // Young's modulus
    double nu_{0.0};              // Poisson's ratio
    double t_{0.0};               // Shell thickness
    double kappa_{5.0 / 6.0};    // Transverse shear correction factor

public:
    // --- Parameter accessors -------------------------------------------------

    constexpr double young_modulus()          const { return E_;     }
    constexpr double poisson_ratio()         const { return nu_;    }
    constexpr double thickness()             const { return t_;     }
    constexpr double shear_correction()      const { return kappa_; }

    constexpr double shear_modulus()         const { return E_ / (2.0 * (1.0 + nu_)); }

    // Derived stiffnesses
    constexpr double membrane_modulus()      const { return E_ * t_ / (1.0 - nu_ * nu_); }
    constexpr double bending_modulus()       const { return E_ * t_ * t_ * t_ / (12.0 * (1.0 - nu_ * nu_)); }
    constexpr double shear_stiffness()       const { return kappa_ * shear_modulus() * t_; }

    // --- Build the section stiffness matrix ----------------------------------

    constexpr void update_section_stiffness() {
        stiffness_matrix_.setZero();

        const double Dm = membrane_modulus();
        const double Db = bending_modulus();
        const double Ds = shear_stiffness();

        // Membrane block A (rows/cols 0..2)
        stiffness_matrix_(0, 0) = Dm;
        stiffness_matrix_(0, 1) = Dm * nu_;
        stiffness_matrix_(1, 0) = Dm * nu_;
        stiffness_matrix_(1, 1) = Dm;
        stiffness_matrix_(2, 2) = Dm * (1.0 - nu_) / 2.0;

        // Bending block D_b (rows/cols 3..5)
        stiffness_matrix_(3, 3) = Db;
        stiffness_matrix_(3, 4) = Db * nu_;
        stiffness_matrix_(4, 3) = Db * nu_;
        stiffness_matrix_(4, 4) = Db;
        stiffness_matrix_(5, 5) = Db * (1.0 - nu_) / 2.0;

        // Transverse shear block S (rows/cols 6..7)
        stiffness_matrix_(6, 6) = Ds;
        stiffness_matrix_(7, 7) = Ds;
    }

    constexpr void update_section_stiffness(
        double E, double nu, double t, double kappa = 5.0 / 6.0)
    {
        E_ = E;  nu_ = nu;  t_ = t;  kappa_ = kappa;
        update_section_stiffness();
    }

    // --- Allow direct full-matrix assignment (for composites/laminates) ------

    void set_matrix(const TangentT& D) { stiffness_matrix_ = D; }

    // --- Constructors --------------------------------------------------------

    constexpr MindlinShellSection(
        double E, double nu, double t, double kappa = 5.0 / 6.0)
        : E_{E}, nu_{nu}, t_{t}, kappa_{kappa}
    {
        update_section_stiffness();
    }

    constexpr MindlinShellSection() = default;

    // --- Testing / debugging -------------------------------------------------

    void print_section_properties() const {
        std::cout << "=== Mindlin-Reissner Shell Section ===" << std::endl;
        std::cout << "E = " << E_ << "  nu = " << nu_ << "  t = " << t_ << std::endl;
        std::cout << "kappa = " << kappa_ << std::endl;
        std::cout << std::endl;
        std::cout << "Section stiffness matrix D_s:" << std::endl;
        std::cout << stiffness_matrix() << std::endl;
    }
};


#endif // FALL_N_MINDLIN_SHELL_SECTION_HH
