#ifndef FALL_LINEAL_MATERIAL_ABSTRACTION_HH
#define FALL_LINEAL_MATERIAL_ABSTRACTION_HH


#include <memory>

#include "constitutive_models/lineal/ElasticRelation.hh"
#include "constitutive_models/lineal/IsotropicRelation.hh"

#include "MaterialState.hh"


template<class ConstitutiveRelation>
class IsotropicElasticMaterial{

  public:  
    using MaterialPolicy = typename ConstitutiveRelation::MaterialPolicy;
    using StrainT        = typename ConstitutiveRelation::StrainT;
    using StressT        = typename ConstitutiveRelation::StressT;
    using MaterialStateT = typename ConstitutiveRelation::MaterialStateT;
    using StateVariableT = typename ConstitutiveRelation::StateVariableT;
    using MatrixT        = Eigen::Matrix<double, StrainT::num_components, StressT::num_components>;

    static constexpr std::size_t dim         = StrainT::dim;
    static constexpr std::size_t num_strains = StrainT::num_components;

  private:
    
    MaterialStateT   state_ ; // Strain
    StressT          stress_; // Default initalized in zeros.

    std::shared_ptr<ConstitutiveRelation> constitutive_law_;

  public:

    inline constexpr MatrixT C() const {return constitutive_law_->compliance_matrix;}
    
    inline constexpr StateVariableT current_state()   const {return state_.current_value()  ;};
    inline constexpr StateVariableT current_state_p() const {return state_.current_value_p();};

    inline constexpr void update_state(const StrainT& e) {state_.update(e);};

    inline constexpr StressT compute_stress(const StrainT& strain) const{
        return constitutive_law_->compute_stress(strain);
    };

    inline StressT compute_stress(const MaterialStateT& state) const{
        return constitutive_law_->compute_stress(state);
    };

    template<typename... Args>  
    auto set_elasticity(Args... args){
        constitutive_law_->update_elasticity(std::forward<Args>(args)...);
    }

    // ========== CONSTRUCTORS =================================

    template<std::floating_point... Args>
    explicit IsotropicElasticMaterial(Args... args) :
        constitutive_law_{std::make_shared<ConstitutiveRelation>(std::forward<Args>(args)...)}
        {}

    ~IsotropicElasticMaterial() = default;

    // ========== TESTING FUNCTIONS ============================
    void print_material_parameters() const{constitutive_law_->print_constitutive_parameters();};
};

typedef IsotropicElasticMaterial<ContinuumIsotropicRelation> ContinuumIsotropicElasticMaterial;
typedef IsotropicElasticMaterial<UniaxialIsotropicRelation > UniaxialIsotropicElasticMaterial;

#endif // FALL_LINEAL_MATERIAL_ABSTRACTION_HH