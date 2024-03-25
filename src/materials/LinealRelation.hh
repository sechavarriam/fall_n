
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

//Some concept
template<Stress StressPolicy, Strain StrainPolicy> //Continuum, Plane, etc. 
class LinealRelation{

    static constexpr std::size_t num_stresses_     = StressPolicy::num_components;
    static constexpr std::size_t num_strains_      = StrainPolicy::num_components;
    static constexpr std::size_t total_parameters_ = num_stresses_*num_strains_;

    std::array<double, total_parameters_> compliance_parameters_{0.0};

  public:

    void compute_stress(const StrainPolicy& strain, StressPolicy& stress){
        stress.tensor = compliance_matrix*strain.tensor;
    };
    
    constexpr inline void set_parameter(std::size_t i, std::size_t j, double value){
        compliance_parameters_[utils::md_index_2_list<num_stresses_,num_strains_>(i,j)] = value;
    };

    Matrix compliance_matrix{compliance_parameters_,num_stresses_,num_strains_}; //elasticity tensor or material stiffness matrix 


    void print_constitutive_parameters(){
        std::cout << "Elasticity Tensor Components: " << std::endl;
        compliance_matrix.print_content();
    };

    constexpr LinealRelation(){compliance_parameters_.fill(0.0);};
    constexpr ~LinealRelation() = default;
};



template<> //Specialization for 1D stress (Uniaxial Stress) avoiding array overhead
class LinealRelation<VoigtStress<1>, VoigtStrain<1>>{ 

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