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

    using StrainType = std::invoke_result_t<decltype(&ConstitutiveRelation::StrainID)>;
    using StressType = std::invoke_result_t<decltype(&ConstitutiveRelation::StressID)>;   

    using StateVar = ConstitutiveRelation::MaterialState;


  public:  
    static constexpr std::size_t dim         = StrainType::dim;
    static constexpr std::size_t num_strains = StrainType::num_components;

  private:
    
    StateVar   state_ ; // Strain
    StressType stress_; // Default initalized in zeros.

    std::shared_ptr<ConstitutiveRelation> constitutive_law_;

  public:

    inline void update_state(const StateVar& strain){state_->update_state(strain);};
    inline void update_stress(){ compute_stress(state_.current_value(), stress_);};

    inline StateVar   get_state(){state_.current_value();};
    inline StressType get_stress(){return stress_;};

    inline void set_stress(const StressType& stress){stress_ = stress;};

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

    // ========== TESTING FUNCTIONS ============================
    void print_material_parameters() const{constitutive_law_->print_constitutive_parameters();};

};

typedef IsotropicElasticMaterial<ContinuumIsotropicRelation> ContinuumIsotropicElasticMaterial;
typedef IsotropicElasticMaterial<UniaxialIsotropicRelation > UniaxialIsotropicElasticMaterial;

#endif // FALL_LINEAL_MATERIAL_ABSTRACTION_HH