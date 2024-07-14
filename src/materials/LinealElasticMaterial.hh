#ifndef FALL_LINEAL_MATERIAL_ABSTRACTION_HH
#define FALL_LINEAL_MATERIAL_ABSTRACTION_HH

// Your code goes here

#include <memory>

#include "LinealRelation.hh"

#include "IsotropicRelation.hh"
#include "MaterialState.hh"

class LinealElasticMaterial{};

template<class ConstitutiveRelation>
class IsotropicElasticMaterial{
    
    using StrainType = std::invoke_result_t<decltype(&ConstitutiveRelation::StrainID)>;
    using StressType = std::invoke_result_t<decltype(&ConstitutiveRelation::StressID)>;


    using State = ElasticMaterialState<StrainType>;

    std::unique_ptr<State>                state_;
    std::shared_ptr<ConstitutiveRelation> constitutive_law_;

  public:

    static constexpr std::size_t num_strains = StrainType::num_components;


    void update_state(const StrainType& strain){state_->update_state(strain);};
    
    auto compute_stress(const StrainType& strain) const{
        return constitutive_law_->compute_stress(strain);
    };
    
    void print_material_parameters() const{constitutive_law_->print_constitutive_parameters();};
    
    auto get_state() const{return state_->current_state();};
    
    template<typename... Args>  
    auto set_elasticity(Args... args){
        constitutive_law_->update_elasticity(std::move(args)...);
    }

    template<std::floating_point... Args>
    IsotropicElasticMaterial(Args... args) : 
        state_{std::make_unique<State>()},
        constitutive_law_{std::make_shared<ConstitutiveRelation>(std::forward<Args>(args)...)}
        {}
};


//inline void IsotropicElasticMaterial::print_material_parameters() const{
//    std::cout << "Elasticity tensor: " << std::endl;
//    std::cout << constitutive_law_->compliance_matrix << std::endl;
//};


typedef IsotropicElasticMaterial<ContinuumIsotropicRelation> ContinuumIsotropicElasticMaterial;
typedef IsotropicElasticMaterial<UniaxialIsotropicRelation > UniaxialIsotropicElasticMaterial;

#endif // FALL_LINEAL_MATERIAL_ABSTRACTION_HH