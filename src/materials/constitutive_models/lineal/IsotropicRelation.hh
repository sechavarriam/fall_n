
#ifndef FALL_N_CONSTITUTIVE_ISOTROPIC_LINEAL_RELATION
#define FALL_N_CONSTITUTIVE_ISOTROPIC_LINEAL_RELATION

#include "ElasticRelation.hh"

// =============================================================================
//  ContinuumIsotropicRelation — 3D isotropic linear-elastic relation
// =============================================================================
//
//  Derives from ElasticRelation<ThreeDimensionalMaterial> and populates the
//  tangent (stiffness) matrix using two parameters: E (Young's modulus) and
//  ν (Poisson's ratio).
//
//  Inherits full ConstitutiveRelation / ElasticConstitutiveRelation conformance
//  from the base class. The only responsibility here is parameter management.
//
// -----------------------------------------------------------------------------

class ContinuumIsotropicRelation : public ElasticRelation<ThreeDimensionalMaterial> {

public:
    using MaterialPolicy = ThreeDimensionalMaterial;

    // Inherit concept-conforming aliases from base
    using Base = ElasticRelation<ThreeDimensionalMaterial>;
    using Base::KinematicT;
    using Base::ConjugateT;
    using Base::TangentT;

    using MaterialStateT = Base::MaterialStateT;
    using StateVariableT = Base::StateVariableT;
    using StrainT        = Base::StrainT;
    using StressT        = Base::StressT;

private:
    double E_{0.0};
    double v_{0.0};

public:
    // --- Elastic parameter accessors -----------------------------------------

    constexpr void set_E(double E) { E_ = E; }
    constexpr void set_v(double v) { v_ = v; }

    constexpr double young_modulus()  const { return E_; }
    constexpr double poisson_ratio()  const { return v_; }

    constexpr double c11()    const { return E_ * (1.0 - v_) / ((1.0 + v_) * (1.0 - 2.0 * v_)); }
    constexpr double c12()    const { return E_ * v_         / ((1.0 + v_) * (1.0 - 2.0 * v_)); }
    constexpr double G()      const { return E_ / (2.0 * (1.0 + v_)); }
    constexpr double k()      const { return E_ / (3.0 * (1.0 - 2.0 * v_)); }
    constexpr double lambda() const { return E_ * v_ / ((1.0 + v_) * (1.0 - 2.0 * v_)); }
    constexpr double mu()     const { return E_ / (2.0 * (1.0 + v_)); }

    // --- Build the tangent matrix from E, ν ----------------------------------

    constexpr void update_elasticity() {
        stiffness_matrix_(0, 0) = c11();
        stiffness_matrix_(1, 1) = c11();
        stiffness_matrix_(2, 2) = c11();
        stiffness_matrix_(3, 3) = (c11() - c12()) / 2.0;
        stiffness_matrix_(4, 4) = (c11() - c12()) / 2.0;
        stiffness_matrix_(5, 5) = (c11() - c12()) / 2.0;
        stiffness_matrix_(0, 1) = c12();
        stiffness_matrix_(0, 2) = c12();
        stiffness_matrix_(1, 0) = c12();
        stiffness_matrix_(1, 2) = c12();
        stiffness_matrix_(2, 0) = c12();
        stiffness_matrix_(2, 1) = c12();
    }

    constexpr void update_elasticity(double young_modulus, double poisson_ratio) {
        E_ = young_modulus;
        v_ = poisson_ratio;
        update_elasticity();
    }

    // --- Testing -------------------------------------------------------------

    void print_constitutive_parameters() const {
        std::cout << "E = " << E_ << ", ν = " << v_ << std::endl;
    }

    // --- Constructors --------------------------------------------------------

    constexpr ContinuumIsotropicRelation(double young_modulus, double poisson_ratio)
        : E_{young_modulus}, v_{poisson_ratio}
    {
        update_elasticity();
    }

    constexpr ContinuumIsotropicRelation() = default;
};

using UniaxialIsotropicRelation = ElasticRelation<UniaxialMaterial>;

#endif // FALL_N_CONSTITUTIVE_ISOTROPIC_LINEAL_RELATION
