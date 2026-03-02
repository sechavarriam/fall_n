#ifndef FALL_LINEAL_MATERIAL_ABSTRACTION_HH
#define FALL_LINEAL_MATERIAL_ABSTRACTION_HH

#include <memory>

#include "constitutive_models/lineal/ElasticRelation.hh"
#include "constitutive_models/lineal/IsotropicRelation.hh"

#include "MaterialState.hh"


template <class ConstitutiveRelation>
class IsotropicElasticMaterial {

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
    MaterialStateT state_{};

    // shared_ptr: multiple material instances can share the same constitutive
    // parameters (e.g. all integration points of the same material zone).
    // Copies of this class intentionally share the relation — this is by design.
    std::shared_ptr<ConstitutiveRelation> constitutive_law_;

public:

    // --- State access ---------------------------------------------------------

    constexpr MatrixT C() const { return constitutive_law_->compliance_matrix; }

    constexpr const StateVariableT& current_state() const { return state_.current_value(); }

    constexpr void update_state(const StateVariableT& e) { state_.update(e); }
    constexpr void update_state(StateVariableT&& e)      { state_.update(std::move(e)); }

    // --- Stress computation ---------------------------------------------------

    constexpr StressT compute_stress(const StrainT& strain) const {
        return constitutive_law_->compute_stress(strain);
    }

    StressT compute_stress(const MaterialStateT& state) const {
        return constitutive_law_->compute_stress(state);
    }

    // --- Elasticity update ----------------------------------------------------

    template <typename... Args>
    void set_elasticity(Args&&... args) {
        constitutive_law_->update_elasticity(std::forward<Args>(args)...);
    }

    // --- Constructors ---------------------------------------------------------

    template <std::floating_point... Args>
    explicit IsotropicElasticMaterial(Args... args)
        : constitutive_law_{std::make_shared<ConstitutiveRelation>(args...)}
    {}

    ~IsotropicElasticMaterial() = default;

    // --- Testing --------------------------------------------------------------

    void print_material_parameters() const {
        constitutive_law_->print_constitutive_parameters();
    }
};

using ContinuumIsotropicElasticMaterial = IsotropicElasticMaterial<ContinuumIsotropicRelation>;
using UniaxialIsotropicElasticMaterial  = IsotropicElasticMaterial<UniaxialIsotropicRelation>;

#endif // FALL_LINEAL_MATERIAL_ABSTRACTION_HH