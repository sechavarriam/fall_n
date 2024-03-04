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
    using State           = ElasticMaterialState<StrainType>;
    //using ConstitutiveLaw = ContinuumIsotropicRelation      ; // ERROR:

    std::unique_ptr<State> state_;
    std::shared_ptr<ConstitutiveLaw> constitutive_law_;

    public:

    auto get_state() const{return state_->current_state();};
    
    template<typename... Args>  
    auto set_elasticity(Args... args){
        constitutive_law_->update_elasticity(std::move(args)...);
    };


    template<std::floating_point... Args>
    IsotropicElasticMaterial(Args... args) : 
        state_{
            std::make_unique<State>()
            },
        constitutive_law_{
            std::make_shared<ConstitutiveLaw>(
                std::forward<Args>(args)...
            )
            }
        {};

    /*
    IsotropicElasticMaterial(double young_modulus, double poisson_ratio) : 
        state_{
            std::make_unique<State>()
            },
        constitutive_law_{
            std::make_shared<ConstitutiveLaw>(
                std::forward<double>(young_modulus),
                std::forward<double>(poisson_ratio)
                )
            }
        {};
    */

};

typedef IsotropicElasticMaterial<VoigtStrain<6>, ContinuumIsotropicRelation> ContinuumIsotropicElasticMaterial;
typedef IsotropicElasticMaterial<VoigtStrain<1>, UniaxialIsotropicRelation> UniaxialIsotropicElasticMaterial;

#endif // LINEALMATERIAL_HH