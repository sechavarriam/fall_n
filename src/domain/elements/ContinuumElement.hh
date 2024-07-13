#ifndef FN_CONTINUUM_ELEMENT
#define FN_CONTINUUM_ELEMENT


#include <memory>

#include "ElementGeometry.hh"


#include "../../materials/Material.hh"
#include "../../numerics/linear_algebra/LinalgOperations.hh"


template <typename MaterialType>
class ContinuumElement{

    std::unique_ptr<ElementGeometry> element_;

    //std::array<double, MaterialType::num_dofs> H_storage_;
    std::array<double, 8> B_storage_;


    public:

    Matrix H; // Displacement Interpolation Matriz Evaluation
    Matrix B; // Strain Displacement Matrix






}; // Forward declaration



#endif