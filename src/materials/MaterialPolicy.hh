
#ifndef FALL_N_MATERIAL_POLICY_HH
#define FALL_N_MATERIAL_POLICY_HH

// =============================================================================
//  Material Policy — Trait bags that bind kinematic / conjugate types
// =============================================================================
//
//  Each policy is a non-instantiable struct that carries type aliases used by
//  constitutive relations (ElasticRelation, etc.) and the state machinery
//  (MaterialState, MaterialPoint) to select the correct measure types.
//
//  Two families:
//
//    SolidMaterial<N>       — Continuum mechanics (Voigt notation)
//    BeamMaterial<N, Dim>   — Structural beams (generalized strains / forces)
//
// =============================================================================

#include "Strain.hh"
#include "Stress.hh"
#include "beam/BeamGeneralizedStrain.hh"
#include "beam/BeamSectionForces.hh"


// =============================================================================
//  SolidMaterial<N> — continuum solid (Voigt-notation measures)
// =============================================================================

template <std::size_t N>
class SolidMaterial {
public:
    using StrainT        = Strain<N>;
    using StressT        = Stress<N>;
    using StateVariableT = StrainT;

    static constexpr std::size_t dim = StrainT::dim;

private:
    constexpr SolidMaterial()  = default;
    constexpr ~SolidMaterial() = default;
};

using UniaxialMaterial          = SolidMaterial<1>;
using PlaneMaterial             = SolidMaterial<3>; // Plane Stress or Plane Strain
using AxisymmetricMaterial      = SolidMaterial<4>;
using ThreeDimensionalMaterial  = SolidMaterial<6>;


// =============================================================================
//  BeamMaterial<N, Dim> — structural beam (generalized strains / forces)
// =============================================================================
//
//  Parameters:
//    N   — number of generalized strain/force components at the section
//    Dim — spatial dimension (2D or 3D)
//
//  Provides the same interface as SolidMaterial so that generic code
//  (ElasticRelation, MaterialState, etc.) works without specialization:
//    StrainT  → BeamGeneralizedStrain<N, Dim>  (satisfies KinematicMeasure)
//    StressT  → BeamSectionForces<N>           (satisfies TensionalConjugate)
//

template <std::size_t N, std::size_t Dim>
    requires (N > 0 && Dim > 0 && Dim <= 3)
class BeamMaterial {
public:
    using StrainT        = BeamGeneralizedStrain<N, Dim>;
    using StressT        = BeamSectionForces<N>;
    using StateVariableT = StrainT;

    static constexpr std::size_t dim = Dim;

private:
    constexpr BeamMaterial()  = default;
    constexpr ~BeamMaterial() = default;
};

// 3D Timoshenko beam: 6 generalized strains {ε, κ_y, κ_z, γ_y, γ_z, θ'}
using TimoshenkoBeam3D       = BeamMaterial<6, 3>;

// 2D Timoshenko beam: 3 generalized strains {ε, κ, γ}
using TimoshenkoBeam2D       = BeamMaterial<3, 2>;

// 2D Euler-Bernoulli beam: 2 generalized strains {ε, κ} (no shear)
using EulerBernoulliBeam2D   = BeamMaterial<2, 2>;


#endif // FALL_N_MATERIAL_POLICY_HH