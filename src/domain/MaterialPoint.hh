#ifndef FALL_N_MATERIAL_POINT_HH
#define FALL_N_MATERIAL_POINT_HH

// Your code goes here


#include <optional> 

#include "IntegrationPoint.hh"

#include "../materials/Material.hh"

template<std::size_t dim> 
class MaterialPoint //: public IntegrationPoint<dim>
{
    using IntegrationPoints = std::optional<IntegrationPoint<dim>>;
    private:
        
        Material          material_{};
        IntegrationPoints integration_points_{};

    public:

        MaterialPoint() = default;
        ~MaterialPoint() = default;
};



#endif // FALL_N_MATERIAL_POINT_HH