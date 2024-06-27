#ifndef FALL_LINEAL_MATERIAL_ABSTRACTION_HH
#define FALL_LINEAL_MATERIAL_ABSTRACTION_HH

// Your code goes here

#include <memory>

#include "LinealRelation.hh"

#include "IsotropicRelation.hh"
#include "MaterialState.hh"

class LinealElasticMaterial{};

template<typename StrainType, typename ConstitutiveLaw>
class IsotropicElasticMaterial{
    using State = ElasticMaterialState<StrainType>;

    std::unique_ptr<State> state_;
    std::shared_ptr<ConstitutiveLaw> constitutive_law_;

  public:
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
        constitutive_law_{std::make_shared<ConstitutiveLaw>(std::forward<Args>(args)...)}
        {}
};


//inline void IsotropicElasticMaterial::print_material_parameters() const{
//    std::cout << "Elasticity tensor: " << std::endl;
//    std::cout << constitutive_law_->compliance_matrix << std::endl;
//};


typedef IsotropicElasticMaterial<VoigtStrain<6>, ContinuumIsotropicRelation> ContinuumIsotropicElasticMaterial;
typedef IsotropicElasticMaterial<VoigtStrain<1>, UniaxialIsotropicRelation>  UniaxialIsotropicElasticMaterial;

#endif // FALL_LINEAL_MATERIAL_ABSTRACTION_HH