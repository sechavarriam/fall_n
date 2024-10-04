
#ifndef FALL_N_MATERIAL_POLICY_HH
#define FALL_N_MATERIAL_POLICY_HH


// This header defines the different....

#include "Strain.hh"
#include "Stress.hh"


class UniaxialMaterial
{
  public:
    static constexpr auto StrainID() -> Strain<1> {return Strain<1>();};
    static constexpr auto StressID() -> Stress<1> {return Stress<1>();};

  private:
    constexpr UniaxialMaterial() = default;
    constexpr ~UniaxialMaterial() = default;
};


class PlaneStrainMaterial
{
  public:
    static auto StrainID() -> Strain<3> {return Strain<3>();};
    static auto StressID() -> Stress<3> {return Stress<3>();};

  private:
    constexpr PlaneStrainMaterial() = default;
    constexpr ~PlaneStrainMaterial() = default;
};


class PlaneStressMaterial
{
  public:
    static auto StrainID() -> Strain<3> {return Strain<3>();};
    static auto StressID() -> Stress<3> {return Stress<3>();};

  private:
    constexpr PlaneStressMaterial() = default;
    constexpr ~PlaneStressMaterial() = default;
};


class AxisymmetricMaterial
{
  public:
    static auto StrainID() -> Strain<4> {return Strain<4>();};
    static auto StressID() -> Stress<4> {return Stress<4>();};

  private:
    constexpr AxisymmetricMaterial() = default;
    constexpr ~AxisymmetricMaterial() = default;
};

class ThreeDimensionalMaterial
{
  public:
    static auto StrainID() -> Strain<6> {return Strain<6>();};
    static auto StressID() -> Stress<6> {return Stress<6>();};

  private:
    constexpr ThreeDimensionalMaterial() = default;
    constexpr ~ThreeDimensionalMaterial() = default;
};






#endif // FALL_N_MATERIAL_POLICY_HH