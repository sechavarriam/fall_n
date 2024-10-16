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

        Material material_; //Instancia del Material.

        

        
    
    public:

        MaterialPoint() = default;
        ~MaterialPoint() = default;
};

#endif // FALL_N_MATERIAL_POINT_HH