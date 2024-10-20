
#ifndef FALL_N_MATERIAL_POLICY_HH
#define FALL_N_MATERIAL_POLICY_HH

// This header defines the different....

#include "Strain.hh"
#include "Stress.hh"


template <std::size_t N>
class SolidMaterial
{
  public:

    using StrainType     = Strain<N>;
    using StressType     = Stress<N>;
    using StateVariableT = Strain<N>;

    static constexpr std::size_t dim = StrainType::dim;

  private:

    constexpr SolidMaterial() = default;
    constexpr ~SolidMaterial() = default;
};

typedef SolidMaterial<1> UniaxialMaterial; 
typedef SolidMaterial<3> PlaneMaterial; // Plane Stress or Plane Strain
typedef SolidMaterial<4> AxisymmetricMaterial;
typedef SolidMaterial<6> ThreeDimensionalMaterial;




//class ForceDeformationMaterial // Define InternalForce Classes as Strain and Stress
//{
//  public:
//    static auto StrainID() -> Strain<1> {return Strain<1>();};
//    static auto StressID() -> Stress<1> {return Stress<1>();};
//
//  private:
//    constexpr ForceDeformationMaterial() = default;
//    constexpr ~ForceDeformationMaterial() = default;
//};




/*
More Material Policies can be defined here. For example, a TemperatureMaterial policy can be defined as follows:

template <std::size_t N>
class TermoMaterial: public SolidMaterial<N> {};


*/


#endif // FALL_N_MATERIAL_POLICY_HH