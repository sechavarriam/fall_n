#ifndef FALL_N_MATERIAL_POINT_HH
#define FALL_N_MATERIAL_POINT_HH

#include <optional> 
#include <memory>

#include "../materials/Material.hh"


template<class MaterialPolicy> 
class MaterialPoint{// : public IntegrationPoint<dim>{
    private:
        
        using Material       = Material<MaterialPolicy>;
        using StateVariableT = MaterialPolicy::StateVariableT;
        using StressT        = MaterialPolicy::StressType;

        Material material_; // material instance. 
                            // It encapsulates the constitutive relation, the state variable and the update strategy.

    public:

        auto& C() const {return material_.C();}; //The Compliance Matrix (Stiffness Matrix)

        MaterialPoint(Material material)   : material_{material} {};
        MaterialPoint(Material&& material) : material_{std::move(material)} {};

        MaterialPoint() = default;
        ~MaterialPoint() = default;
        



};

#endif // FALL_N_MATERIAL_POINT_HH