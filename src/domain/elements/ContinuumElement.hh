#ifndef FN_CONTINUUM_ELEMENT
#define FN_CONTINUUM_ELEMENT

#include <memory>
#include <array>

#include "ElementGeometry.hh"


#include "../../model/MaterialPoint.hh"

#include "../../numerics/linear_algebra/LinalgOperations.hh"

template <typename MaterialType, std::size_t ndof>
class ContinuumElement
{
  using Array = std::array<double, MaterialType::dim>;

  static constexpr auto dim         = MaterialType::dim;
  static constexpr auto num_strains = MaterialType::num_strains;

  ElementGeometry<dim> *geometry_;
  

  constexpr auto num_integration_points() const noexcept { return geometry_->num_integration_points(); };

  

  //std::vector<MaterialPoint<MaterialType>> material_points_;


public:

  constexpr auto num_nodes() const noexcept { return geometry_->num_nodes(); };

  auto H(const Array& X) const noexcept
  {
    Matrix H(ndof, num_nodes() * dim);
    H.assembly_begin();

    if constexpr (dim == 1)
    {
      for (std::size_t i = 0; i < num_nodes(); ++i)
      {
        H.insert_values(0, i, geometry_->H(i, X));
      }
    }
    else if constexpr (dim == 2)
    {
      auto k = 0;
      for (std::size_t i = 0; i < num_nodes(); ++i)
      {
        H.insert_values(0, k, geometry_->H(i, X));
        H.insert_values(0, k + 1, 0.0);

        H.insert_values(1, k, 0.0);
        H.insert_values(1, k + 1, geometry_->H(i, X));

        k += dim;
      }
    }
    else if constexpr (dim == 3)
    {
      auto k = 0;
      for (std::size_t i = 0; i < num_nodes(); ++i)
      {
        H.insert_values(0, k, geometry_->H(i, X));
        H.insert_values(0, k + 1, 0.0);
        H.insert_values(0, k + 2, 0.0);

        H.insert_values(1, k, 0.0);
        H.insert_values(1, k + 1, geometry_->H(i, X));
        H.insert_values(1, k + 2, 0.0);

        H.insert_values(2, k, 0.0);
        H.insert_values(2, k + 1, 0.0);
        H.insert_values(2, k + 2, geometry_->H(i, X));

        k += dim;
      }
    }

    H.assembly_end();
    return H;
  };

  auto B(const Array &X)
  {
    Matrix B(num_strains, num_nodes() * dim);
    B.assembly_begin();

    if      constexpr (dim == 1) std::runtime_error("1D material not implemented yet for ContinuumElement.");
    else if constexpr (dim == 2)
    {
      // Ordering defined acoording to Voigt notation [11, 22, 12]
      auto k = 0;
      for (std::size_t i = 0; i < num_nodes(); ++i)
      {
        B.insert_values(0, k    , geometry_->dH_dx(i, 0, X));
        B.insert_values(0, k + 1, 0.0);

        B.insert_values(1, k    , 0.0);
        B.insert_values(1, k + 1, geometry_->dH_dx(i, 1, X));

        B.insert_values(2, k    , 2*geometry_->dH_dx(i, 1, X));
        B.insert_values(2, k + 1, 2*geometry_->dH_dx(i, 0, X));

        k += dim;
      }
    }
    else if constexpr (dim == 3)
    {
      auto k=0;
      for (std::size_t i = 0; i < num_nodes(); ++i)
      {
          // Ordering defined acoording to Voigt notation [11, 22, 33, 32, 31, 12]
          B.insert_values(0, k  , geometry_->dH_dx(i, 0, X));
          B.insert_values(0, k+1, 0.0);
          B.insert_values(0, k+2, 0.0);

          B.insert_values(1, k  , 0.0);
          B.insert_values(1, k+1, geometry_->dH_dx(i, 1, X));
          B.insert_values(1, k+2, 0.0);

          B.insert_values(2, k  , 0.0);
          B.insert_values(2, k+1, 0.0);
          B.insert_values(2, k+2, geometry_->dH_dx(i, 2, X));

          B.insert_values(3, k  , 0.0);
          B.insert_values(3, k+1, 2*geometry_->dH_dx(i, 2, X));
          B.insert_values(3, k+2, 2*geometry_->dH_dx(i, 1, X));

          B.insert_values(4, k  , 2*geometry_->dH_dx(i, 2, X));
          B.insert_values(4, k+1, 0.0);
          B.insert_values(4, k+2, 2*geometry_->dH_dx(i, 0, X));

          B.insert_values(5, k  , 0.0);
          B.insert_values(5, k+1, 2*geometry_->dH_dx(i, 0, X));
          B.insert_values(5, k+2, 2*geometry_->dH_dx(i, 1, X));

          k+=dim;
      }
    }
    B.assembly_end();
    return B;
  };

  auto BtCB (const MaterialType& M,  const Array &X) {
    Matrix K{ndof, ndof};                          
    K = linalg::mat_mat_PtAP(B(X), M.C() );
    return K; // B^t * C * B
  }; 

  auto K(const MaterialType& mat) {
    Matrix K{ndof, ndof};

    K=geometry_->integrate([this, &mat](const Array &X)
      {
        return BtCB(mat, X);
      }
    );
    
    return K;

  };
  
  void inject_K(/*const Matrix& K, const std::array<std::size_t, ndof>& dofs*/){
      // Inject (BUILD ) K into global stiffness matrix
  };


  // CONSTRUCTOR

  ContinuumElement() = delete;
  ContinuumElement(ElementGeometry<dim> *geometry) : geometry_{geometry} {};

}; // Forward declaration

#endif