
#ifndef FALL_N_CONSTUTUTIVE_LINEAL_RELATION
#define FALL_N_CONSTUTUTIVE_LINEAL_RELATION


#include <iostream>
#include <cstddef>
#include <type_traits>
#include <concepts>

#include "../../MaterialPolicy.hh"
#include "../../StateVariable.hh"

#include "../../../numerics/linear_algebra/Matrix.hh"
#include "../../../utils/index.hh"



//template<StressC StressType, StrainC StrainType> //requires (/*operations well defined*/) <---- TODO

template<class MaterialPolicy>
class ElasticRelation
  {
    public:

    // result_of can be used with a pointer to member function as follows
    using StrainType = std::invoke_result_t<decltype(&MaterialPolicy::StrainID)>;
    using StressType = std::invoke_result_t<decltype(&MaterialPolicy::StressID)>;
    
    using StateVariableType = MaterialState<ElasticState,StrainType>;

    static constexpr auto StrainID() -> StrainType {return MaterialPolicy::StrainID();};
    static constexpr auto StressID() -> StressType {return MaterialPolicy::StressID();};

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

    // =========== CONSTRUCTORS ==========================  
    constexpr ElasticRelation(){};
    constexpr ~ElasticRelation() = default;

    // =========== TESTING FUNCTIONS ============================
        void print_constitutive_parameters(){
        std::cout << "Elasticity Tensor Components: " << std::endl;
        compliance_matrix.print_content();
    };
};





template<> //Specialization for 1D stress (Uniaxial Stress) avoiding array overhead
class ElasticRelation<UniaxialMaterial>{ 
  using StrainType = Strain<1>;
  using StressType = Stress<1>;

  public:

    static constexpr auto StrainID()->StrainType {return StrainType();};
    static constexpr auto StressID()->StressType {return StressType();};

    using StateVariableType = MaterialState<ElasticState,StrainType>;


    static constexpr std::size_t dim               = StrainType::dim;
    static constexpr std::size_t num_strains_      = StrainType::num_components;
    static constexpr std::size_t num_stresses_     = StressType::num_components;
    static constexpr std::size_t total_parameters_ = num_stresses_*num_strains_;

  private:

    double E_{0.0}; // E

public:

    void compute_stress(const Strain<1>& strain, Stress<1>& stress){
        stress.vector = E_*strain.vector;
    };
    
    constexpr inline void set_parameter    (double value)        {E_ = value;};
    constexpr inline void update_elasticity(double young_modulus){E_ = young_modulus;};

    constexpr ElasticRelation(double young_modulus) : E_{young_modulus}{};


    // =========== CONSTRUCTORS ==========================
    constexpr  ElasticRelation() = default;
    constexpr ~ElasticRelation() = default;

    // =========== TESTING FUNCTIONS ============================
    void print_constitutive_parameters() const{
        std::cout << "Proportionality Compliance Parameter (Young): " << E_ << std::endl;
    };

};

#endif // FALL_N_LINEAL_MATERIAL