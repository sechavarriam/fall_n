
#ifndef FALL_N_CONSTUTUTIVE_LINEAL_RELATION
#define FALL_N_CONSTUTUTIVE_LINEAL_RELATION


#include <iostream>
#include <cstddef>
#include <type_traits>
#include <concepts>

#include "Stress.hh"
#include "Strain.hh"

#include "../numerics/linear_algebra/Matrix.hh"
#include "../utils/index.hh"


template<Stress StressType, Strain StrainType> //requires (/*operations well defined*/) <---- TODO
class LinealRelation
  {
    public:

    static constexpr auto StrainID()->StrainType {std::unreachable();};
    static constexpr auto StressID()->StressType {std::unreachable();};

    static constexpr std::size_t dim               = StrainType::dim;
    static constexpr std::size_t num_strains_      = StrainType::num_components;
    static constexpr std::size_t num_stresses_     = StressType::num_components;
    static constexpr std::size_t total_parameters_ = num_stresses_*num_strains_;

  private:

    std::array<double, total_parameters_> compliance_parameters_{0.0};

    

  public:

    Matrix compliance_matrix{compliance_parameters_,num_stresses_,num_strains_}; //elasticity tensor or material stiffness matrix 



    void compute_stress(const StrainType& strain, StressType& stress){
        stress.tensor = compliance_matrix*strain.tensor;
    };
    
    constexpr void set_parameter(std::size_t i, std::size_t j, double value){
        compliance_parameters_[utils::md_index_2_list<num_stresses_,num_strains_>(i,j)] = value;
    };

    void print_constitutive_parameters(){
        std::cout << "Elasticity Tensor Components: " << std::endl;
        compliance_matrix.print_content();
    };
  
    constexpr LinealRelation(){};
    constexpr ~LinealRelation() = default;
};



template<> //Specialization for 1D stress (Uniaxial Stress) avoiding array overhead
class LinealRelation<VoigtStress<1>, VoigtStrain<1>>{ 
  using StrainType = VoigtStrain<1>;
  using StressType = VoigtStress<1>;

  public:

    static constexpr auto StrainID()->StrainType {return StrainType();};
    static constexpr auto StressID()->StressType {return StressType();};

    static constexpr std::size_t dim               = StrainType::dim;
    static constexpr std::size_t num_strains_      = StrainType::num_components;
    static constexpr std::size_t num_stresses_     = StressType::num_components;
    static constexpr std::size_t total_parameters_ = num_stresses_*num_strains_;

  private:

    double E_{0.0}; // E

  public:

    void print_constitutive_parameters() const{
        std::cout << "Proportionality Compliance Parameter (Young): " << E_ << std::endl;
    };

    void compute_stress(const VoigtStrain<1>& strain, VoigtStress<1>& stress){
        stress.tensor = E_*strain.tensor;
    };
    
    constexpr inline void set_parameter    (double value)        {E_ = value;};
    constexpr inline void update_elasticity(double young_modulus){E_ = young_modulus;};

    constexpr LinealRelation(double young_modulus) : E_{std::forward<double>(young_modulus)}{};

    constexpr  LinealRelation() = default;
    constexpr ~LinealRelation() = default;
};


//typedef LinealRelation<VoigtStress<1>, VoigtStrain<1>> UniaxialMaterial;
//typedef LinealRelation<VoigtStress<3>, VoigtStrain<3>> ContinuumMaterial2D;
//typedef LinealRelation<VoigtStress<6>, VoigtStrain<6>> ContinuumMaterial3D;

#endif // FALL_N_LINEAL_MATERIAL