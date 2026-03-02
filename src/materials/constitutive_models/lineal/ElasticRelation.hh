
#ifndef FALL_N_CONSTITUTIVE_LINEAL_RELATION
#define FALL_N_CONSTITUTIVE_LINEAL_RELATION

#include <iostream>
#include <cstddef>
#include <type_traits>
#include <concepts>

#include "../../MaterialPolicy.hh"
#include "../../MaterialState.hh"
#include "../../ConstitutiveRelation.hh"

#include "../../../numerics/linear_algebra/Matrix.hh"
#include "../../../utils/index.hh"


// =============================================================================
//  ElasticRelation<MaterialPolicy>  —  General linear-elastic relation
// =============================================================================
//
//  This class implements a generic linear-elastic constitutive relation
//  parameterized by a MaterialPolicy (e.g. SolidMaterial<6>).
//
//  It satisfies both ConstitutiveRelation and ElasticConstitutiveRelation
//  concepts defined in ConstitutiveRelation.hh.
//
//  The relation is:   σ = C · ε   (linear, path-independent)
//
//  The tangent operator C is constant (does not depend on ε), so both
//  tangent() and tangent(k) return the same matrix.
//
// -----------------------------------------------------------------------------

template <class MaterialPolicy>
class ElasticRelation {

public:
    // --- Type aliases required by ConstitutiveRelation concept ----------------
    using KinematicT = typename MaterialPolicy::StrainT;
    using ConjugateT = typename MaterialPolicy::StressT;
    using TangentT   = TangentMatrix<KinematicT, ConjugateT>;

    // --- Legacy aliases (kept for backward compatibility) ---------------------
    using StrainT        = KinematicT;
    using StressT        = ConjugateT;
    using MatrixT        = TangentT;
    using MaterialStateT = MaterialState<ElasticState, StrainT>;
    using StateVariableT = typename MaterialStateT::StateVariableT;

    static constexpr std::size_t dim           = KinematicT::dim;
    static constexpr std::size_t num_strains_  = KinematicT::num_components;
    static constexpr std::size_t num_stresses_ = ConjugateT::num_components;

protected:
    // The tangent (compliance/stiffness) matrix. Protected so that derived
    // classes (e.g. ContinuumIsotropicRelation) can populate it directly.
    TangentT compliance_matrix_ = TangentT::Zero();

public:
    // --- ConstitutiveRelation interface --------------------------------------

    // compute_response(k): computes σ = C · ε
    ConjugateT compute_response(const KinematicT& strain) const {
        ConjugateT stress;
        stress.set_components(compliance_matrix_ * strain.components());
        return stress;
    }

    // tangent(k): returns ∂σ/∂ε evaluated at k. For linear elasticity,
    // the tangent is constant — the argument is ignored.
    TangentT tangent([[maybe_unused]] const KinematicT& k) const {
        return compliance_matrix_;
    }

    // --- ElasticConstitutiveRelation interface --------------------------------

    // tangent(): returns the constant tangent operator (no kinematic argument).
    TangentT tangent() const { return compliance_matrix_; }

    // --- Legacy interface (backward compatibility) ---------------------------

    // These delegate to the concept-conforming methods above.
    // TODO: Migrate call sites and remove these.

    StressT compute_stress(const StrainT& strain) const {
        return compute_response(strain);
    }

    void compute_stress(StressT& stress, const StrainT& strain) const {
        stress.set_components(compliance_matrix_ * strain.components());
    }

    // Direct read access to the matrix (for IsotropicRelation and similar).
    const TangentT& compliance_matrix() const { return compliance_matrix_; }

    // --- Constructors --------------------------------------------------------
    constexpr ElasticRelation() = default;
    constexpr ~ElasticRelation() = default;
};

// Static verification that the general template satisfies concepts
static_assert(
    ConstitutiveRelation<ElasticRelation<ThreeDimensionalMaterial>>,
    "ElasticRelation<ThreeDimensionalMaterial> must satisfy ConstitutiveRelation");
static_assert(
    ElasticConstitutiveRelation<ElasticRelation<ThreeDimensionalMaterial>>,
    "ElasticRelation<ThreeDimensionalMaterial> must satisfy ElasticConstitutiveRelation");


// =============================================================================
//  ElasticRelation<UniaxialMaterial>  —  Specialization for 1D (scalar E)
// =============================================================================
//
//  For uniaxial stress, the "compliance matrix" is just the Young's modulus E.
//  This specialization avoids matrix overhead entirely.
//
// -----------------------------------------------------------------------------

template <>
class ElasticRelation<UniaxialMaterial> {

public:
    // --- Concept-required type aliases ----------------------------------------
    using KinematicT = Strain<1>;
    using ConjugateT = Stress<1>;
    using TangentT   = Eigen::Matrix<double, 1, 1>;

    // --- Legacy aliases ------------------------------------------------------
    using MaterialPolicy = UniaxialMaterial;
    using StrainT        = KinematicT;
    using StressT        = ConjugateT;
    using MatrixT        = TangentT;
    using MaterialStateT = MaterialState<ElasticState, StrainT>;
    using StateVariableT = StrainT;

    static constexpr std::size_t dim           = KinematicT::dim;
    static constexpr std::size_t num_strains_  = KinematicT::num_components;
    static constexpr std::size_t num_stresses_ = ConjugateT::num_components;

private:
    double E_{0.0};

public:
    // --- ConstitutiveRelation interface --------------------------------------

    ConjugateT compute_response(const KinematicT& strain) const {
        ConjugateT stress;
        stress.set_components(E_ * strain.components());
        return stress;
    }

    TangentT tangent([[maybe_unused]] const KinematicT& k) const {
        TangentT C;
        C(0, 0) = E_;
        return C;
    }

    // --- ElasticConstitutiveRelation interface --------------------------------

    TangentT tangent() const {
        TangentT C;
        C(0, 0) = E_;
        return C;
    }

    // --- Legacy interface ----------------------------------------------------

    StressT compute_stress(const StrainT& strain) const {
        return compute_response(strain);
    }

    void compute_stress(const StrainT& strain, StressT& stress) const {
        stress.set_components(E_ * strain.components());
    }

    // --- Parameter access ----------------------------------------------------

    constexpr void set_parameter(double value)         { E_ = value; }
    constexpr void update_elasticity(double young_mod) { E_ = young_mod; }

    // --- Constructors --------------------------------------------------------

    constexpr ElasticRelation(double young_modulus) : E_{young_modulus} {}
    constexpr ElasticRelation() = default;
    constexpr ~ElasticRelation() = default;

    // --- Testing -------------------------------------------------------------

    void print_constitutive_parameters() const {
        std::cout << "Proportionality Compliance Parameter (Young): " << E_ << std::endl;
    }
};

// Static verification for the uniaxial specialization
static_assert(
    ElasticConstitutiveRelation<ElasticRelation<UniaxialMaterial>>,
    "ElasticRelation<UniaxialMaterial> must satisfy ElasticConstitutiveRelation");


#endif // FALL_N_CONSTITUTIVE_LINEAL_RELATION