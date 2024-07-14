#ifndef FN_CONTINUUM_ELEMENT
#define FN_CONTINUUM_ELEMENT


#include <memory>

#include "ElementGeometry.hh"


#include "../../materials/Material.hh"
#include "../../numerics/linear_algebra/LinalgOperations.hh"


template <typename MaterialType, std::size_t ndof>
class ContinuumElement{

    static constexpr auto dim         = MaterialType::dim;
    static constexpr auto num_strains = MaterialType::num_strains;

    ElementGeometry* geometry_;

    std::array<double, ndof*ndof> K_{0.0}; // Stiffness Matrix data
    
  public:

    constexpr auto num_nodes()   const noexcept {return geometry_->num_nodes();};

    auto H(/*const geometry::Point<dim>& X*/) const noexcept {
      Matrix H{ndof, num_nodes()*dim};
      
      return H;
      };

    auto B(/*const geometry::Point<dim>& X*/) const noexcept {
      Matrix B{num_strains, num_nodes()*dim};
      
      return B;
      };

    void compute_stiffness_matrix(){
        // Compute stiffness matrix
        // K_ = B^T * D * B * det(J)
    };

    // CONSTRUCTOR

    ContinuumElement() = delete;

    ContinuumElement(ElementGeometry* geometry) : geometry_{geometry} 
    {};





}; // Forward declaration



#endif