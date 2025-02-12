#ifndef FALL_N_MATERIAL_POINT_HH
#define FALL_N_MATERIAL_POINT_HH

#include <optional> 
#include <memory>

#include "../materials/Material.hh"
#include "../geometry/IntegrationPoint.hh"

template<class MaterialPolicy> 
class MaterialPoint{// : public IntegrationPoint<dim>{ or Point

        static constexpr std::size_t counter{0};

    private:

        using MaterialT      = Material<MaterialPolicy>;
        using StateVariableT = MaterialPolicy::StateVariableT;
        using StressT        = MaterialPolicy::StressT;

        static constexpr std::size_t dim = MaterialPolicy::dim;

        std::size_t id_; // Unique identifier for the material point.

        IntegrationPoint<dim>* gauss_point_; // The integration point where the material point is located.

        MaterialT material_; // material instance. 
                            // It encapsulates the constitutive relation, the state variable and the update strategy.

    public:

        std::array<double, dim> coord() const noexcept {return gauss_point_->coord();};

        void update_state(const StateVariableT& state) noexcept {material_.update_state(state);};
        void update_state(StateVariableT&& state) noexcept {material_.update_state(std::forward<StateVariableT>(state));};
        //void update_state(StateVariableT state) noexcept {material_.update_state(state);};
        
        auto current_state()const noexcept {return material_.current_state();};

        void bind_integration_point(IntegrationPoint<dim>& p){
            gauss_point_ = &p;
        };


        Eigen::Matrix<double, StateVariableT::num_components, StateVariableT::num_components> C() {return material_.C();}; 
        //C() const {return material_.C();}; 

        MaterialPoint(MaterialT material)   : material_{material} {};
        MaterialPoint(MaterialT&& material) : material_{std::move(material)} {};

        MaterialPoint() = default;
        ~MaterialPoint() = default;
        
};

#endif // FALL_N_MATERIAL_POINT_HH