#ifndef FN_CONTINUUM_ELEMENT
#define FN_CONTINUUM_ELEMENT

#include <memory>
#include <array>

#include "ElementGeometry.hh"

#include "../../materials/Material.hh"
#include "../../numerics/linear_algebra/LinalgOperations.hh"

template <typename MaterialType, std::size_t ndof>
class ContinuumElement
{

  using Array = std::array<double, MaterialType::dim>;

  static constexpr auto dim = MaterialType::dim;
  static constexpr auto num_strains = MaterialType::num_strains;

  ElementGeometry *geometry_;

  // std::array<double, ndof*ndof> K_{0.0}; // Stiffness Matrix data

public:
  constexpr auto num_nodes() const noexcept { return geometry_->num_nodes(); };

  auto H(/*const geometry::Point<dim>& X*/) const noexcept
  {
    Matrix H{ndof, num_nodes() * dim};

    return H;
  };

  auto B(const Array &X)
  {
    Matrix B{num_strains, num_nodes() * dim};
    B.assembly_begin();

    if      constexpr (dim == 1) std::runtime_error("1D material not implemented yet for ContinuumElement.");
    else if constexpr (dim == 2)
    {
      // Ordering defined acoording to Voigt notation [11, 22, 12]
      auto k = 0;
      for (std::size_t i = 0; i < num_nodes(); ++i)
      {
        B.insert_values(0, k    , geometry_->dH_dx(i, 0, X[0]));
        B.insert_values(0, k + 1, 0.0);

        B.insert_values(1, k    , 0.0);
        B.insert_values(1, k + 1, geometry_->dH_dx(i, 1, X[1]));

        B.insert_values(2, k    , 2*geometry_->dH_dx(i, 1, X[1]));
        B.insert_values(2, k + 1, 2*geometry_->dH_dx(i, 0, X[0]));

        k += dim;
      }
    }
    else if constexpr (dim == 3)
    {
      auto k=0;
      for (std::size_t i = 0; i < num_nodes(); ++i)
      {
          // Ordering defined acoording to Voigt notation [11, 22, 33, 32, 31, 12]
          B.insert_values(0, k  , geometry_->dH_dx(i, 0, X[0]));
          B.insert_values(0, k+1, 0.0);
          B.insert_values(0, k+2, 0.0);

          B.insert_values(1, k  , 0.0);
          B.insert_values(1, k+1, geometry_->dH_dx(i, 1, X[1]));
          B.insert_values(1, k+2, 0.0);

          B.insert_values(2, k  , 0.0);
          B.insert_values(2, k+1, 0.0);
          B.insert_values(2, k+2, geometry_->dH_dx(i, 2, X[2]));

          B.insert_values(3, k  , 0.0);
          B.insert_values(3, k+1, 2*geometry_->dH_dx(i, 2, X[1]));
          B.insert_values(3, k+2, 2*geometry_->dH_dx(i, 1, X[2]));

          B.insert_values(4, k  , 2*geometry_->dH_dx(i, 2, X[0]));
          B.insert_values(4, k+1, 0.0);
          B.insert_values(4, k+2, 2*geometry_->dH_dx(i, 0, X[2]));

          B.insert_values(5, k  , 0.0);
          B.insert_values(5, k+1, 2*geometry_->dH_dx(i, 0, X[1]));
          B.insert_values(5, k+2, 2*geometry_->dH_dx(i, 1, X[0]));

          k+=dim;
      }
    }

    B.assembly_end();
    return B;
  };


  void inject_K(/*const Matrix& K, const std::array<std::size_t, ndof>& dofs*/){
      // Inject (BUILD ) K into global stiffness matrix
  };

  void compute_stiffness_matrix() {
    // Compute stiffness matrix
    // K_ = B^T * D * B * det(J)
  };

  // CONSTRUCTOR

  ContinuumElement() = delete;

  ContinuumElement(ElementGeometry *geometry) : geometry_{geometry} {};

}; // Forward declaration

#endif