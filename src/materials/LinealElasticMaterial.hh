#ifndef FALL_LINEAL_MATERIAL_ABSTRACTION_HH
#define FALL_LINEAL_MATERIAL_ABSTRACTION_HH

// Your code goes here

#include <memory>

#include "LinealRelation.hh"

#include "IsotropicRelation.hh"
#include "MaterialState.hh"

class LinealElasticMaterial{};


class IsotropicElasticMaterial{
    using State = ElasticMaterialState<VoigtStrain<6>>;
    using ConstitutiveLaw = ContinuumIsotropicRelation;

    std::unique_ptr<State> state_;
    std::shared_ptr<ConstitutiveLaw> constitutive_law_;


    public:

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





};


#endif // LINEALMATERIAL_HH