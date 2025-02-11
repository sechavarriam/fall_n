
#ifndef FALL_N_MATERIAL_POLICY_HH
#define FALL_N_MATERIAL_POLICY_HH

// This header defines the different....

#include "Strain.hh"
#include "Stress.hh"


template <std::size_t N>
class SolidMaterial
{
  public:

    using StrainT        = Strain<N>;
    using StressT        = Stress<N>;

    using StateVariableT = StrainT;

    static constexpr std::size_t dim = StrainT::dim;

    //DIFERENTIAL OPERATORS
    

  private:
  
    constexpr SolidMaterial() = default;
    constexpr ~SolidMaterial() = default;
};

typedef SolidMaterial<1> UniaxialMaterial; 
typedef SolidMaterial<3> PlaneMaterial; // Plane StressDeprecated or Plane StrainDeprecated
typedef SolidMaterial<4> AxisymmetricMaterial;
typedef SolidMaterial<6> ThreeDimensionalMaterial;




//class ForceDeformationMaterial // Define InternalForce Classes as StrainDeprecated and StressDeprecated
//{
//  public:
//    static auto StrainID() -> StrainDeprecated<1> {return StrainDeprecated<1>();};
//    static auto StressID() -> StressDeprecated<1> {return Stress<1>();};
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