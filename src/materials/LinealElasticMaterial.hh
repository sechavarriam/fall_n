#ifndef FALL_LINEAL_MATERIAL_ABSTRACTION_HH
#define FALL_LINEAL_MATERIAL_ABSTRACTION_HH

// Your code goes here

#include <memory>

#include "constitutive_models/lineal/ElasticRelation.hh"
#include "constitutive_models/lineal/IsotropicRelation.hh"

#include "MaterialState.hh"

class LinealElasticMaterial{};

template<class ConstitutiveRelation>
class IsotropicElasticMaterial{

  public:  

    using StrainType = typename ConstitutiveRelation::StrainType;
    using StressType = typename ConstitutiveRelation::StressType;

    using StateVariableT = ConstitutiveRelation::StateVariableT;
    using MaterialStateT = typename ConstitutiveRelation::MaterialStateT;
    using MaterialPolicy = typename ConstitutiveRelation::MaterialPolicy;

    static constexpr std::size_t dim         = StrainType::dim;
    static constexpr std::size_t num_strains = StrainType::num_components;

  private:
    
    MaterialStateT   state_ ; // Strain
    StressType       stress_; // Default initalized in zeros.

    std::shared_ptr<ConstitutiveRelation> constitutive_law_;

  public:

    inline StateVariableT get_state() const {return state_.current_value();};
    
    inline void compute_stress(const StrainType& strain, StressType& stress) const{
        return constitutive_law_->compute_stress(strain, stress);
    };

    auto C() const {return constitutive_law_->compliance_matrix;}

    template<typename... Args>  
    auto set_elasticity(Args... args){
        constitutive_law_->update_elasticity(std::move(args)...);
    }

    // ========== CONSTRUCTORS =================================

    template<std::floating_point... Args>
    IsotropicElasticMaterial(Args... args) :
        constitutive_law_{std::make_shared<ConstitutiveRelation>(std::forward<Args>(args)...)}
        {}

    ~IsotropicElasticMaterial() = default;

    IsotropicElasticMaterial(const IsotropicElasticMaterial &other) = default;
    IsotropicElasticMaterial(IsotropicElasticMaterial &&other) = default;
    IsotropicElasticMaterial &operator=(const IsotropicElasticMaterial &other) = default;
    IsotropicElasticMaterial &operator=(IsotropicElasticMaterial &&other) = default;  


    // ========== TESTING FUNCTIONS ============================
    void print_material_parameters() const{constitutive_law_->print_constitutive_parameters();};

};

typedef IsotropicElasticMaterial<ContinuumIsotropicRelation> ContinuumIsotropicElasticMaterial;
typedef IsotropicElasticMaterial<UniaxialIsotropicRelation > UniaxialIsotropicElasticMaterial;

#endif // FALL_LINEAL_MATERIAL_ABSTRACTION_HH