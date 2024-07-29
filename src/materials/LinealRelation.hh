
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

// Linear and non linear as policys?
template<StressC StressType, StrainC StrainType> //requires (/*operations well defined*/) <---- TODO
class ElasticRelation
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
        stress.vector = compliance_matrix*strain.vector; // sigma = C*epsilon
    };
    
    constexpr void set_parameter(std::size_t i, std::size_t j, double value){
        compliance_parameters_[utils::md_index_2_list<num_stresses_,num_strains_>(i,j)] = value;
    };

    void print_constitutive_parameters(){
        std::cout << "Elasticity Tensor Components: " << std::endl;
        compliance_matrix.print_content();
    };
  
    constexpr ElasticRelation(){};
    constexpr ~ElasticRelation() = default;
};

template<> //Specialization for 1D stress (Uniaxial Stress) avoiding array overhead
class ElasticRelation<Stress<1>, Strain<1>>{ 
  using StrainType = Strain<1>;
  using StressType = Stress<1>;

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

    void compute_stress(const Strain<1>& strain, Stress<1>& stress){
        stress.vector = E_*strain.vector;
    };
    
    constexpr inline void set_parameter    (double value)        {E_ = value;};
    constexpr inline void update_elasticity(double young_modulus){E_ = young_modulus;};

    constexpr ElasticRelation(double young_modulus) : E_{std::forward<double>(young_modulus)}{};

    constexpr  ElasticRelation() = default;
    constexpr ~ElasticRelation() = default;
};


//typedef ElasticRelation<Stress<1>, Strain<1>> UniaxialMaterial;
//typedef ElasticRelation<Stress<3>, Strain<3>> ContinuumMaterial2D;
//typedef ElasticRelation<Stress<6>, Strain<6>> ContinuumMaterial3D;

#endif // FALL_N_LINEAL_MATERIAL