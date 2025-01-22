#ifndef FN_CONTINUUM_ELEMENT
#define FN_CONTINUUM_ELEMENT

#include <memory>
#include <array>

#include <print>

#include <petsc.h>

#include "element_geometry/ElementGeometry.hh"

#include "../model/MaterialPoint.hh"
#include "../materials/Material.hh"
#include "../numerics/linear_algebra/LinalgOperations.hh"

template <typename MaterialPolicy, std::size_t ndof>
class ContinuumElement
{
  using PETScMatrix = Mat; // TODO: Use PETSc DeprecatedDenseMatrix

  // using MaterialPolicy = MaterialPolicy;
  using MaterialPoint = MaterialPoint<MaterialPolicy>;
  using Material = Material<MaterialPolicy>;

  using Array = std::array<double, MaterialPolicy::dim>;

  static constexpr auto dim = MaterialPolicy::dim;
  static constexpr auto num_strains = MaterialPolicy::StrainType::num_components;

  ElementGeometry<dim> *geometry_;
  std::vector<MaterialPoint> material_points_{};

  bool is_multimaterial_{true}; // If true, the element has different materials in each integration point.

public:
  constexpr auto get_geometry() const noexcept { return geometry_; };

  constexpr auto sieve_id() const noexcept { return geometry_->sieve_id.value(); };
  constexpr auto node_p(std::size_t i) const noexcept { return geometry_->node_p(i); };

  constexpr void set_num_dof_in_nodes() noexcept
  {
    for (std::size_t i = 0; i < num_nodes(); ++i)
      geometry_->node_p(i).set_num_dof(dim);
  };

  constexpr inline auto num_integration_points() const noexcept { return geometry_->num_integration_points(); };

  constexpr inline auto bind_integration_points() noexcept
  {
    std::size_t count{0};
    for (auto &point : material_points_)
    {
      point.bind_integration_point(geometry_->integration_point_[count++]);
    }
  };

  constexpr auto num_nodes() const noexcept { return geometry_->num_nodes(); };

  constexpr auto get_dofs_index() const noexcept
  {
    std::size_t i;

    auto N = num_nodes();
    std::vector<std::span<std::size_t>> dofs_index; // This thing avoids the copy of the vector.
    dofs_index.reserve(N);

    for (i = 0; i < N; ++i)
      dofs_index.emplace_back(geometry_->node(i).dof_index());

    return dofs_index;
  };

  DeprecatedDenseMatrix H(const Array &X); // Declaration. Definition at the end of the file // Could be injected from material policy.
  DeprecatedDenseMatrix B(const Array &X); // Declaration. Definition at the end of the file // Could be injected from material policy.

  DeprecatedDenseMatrix BtCB([[maybe_unused]] const Array &X)
  { // TODO: Optimize this for each dimension.

    auto N = static_cast<PetscInt>(ndof * num_nodes());

    auto get_C = [this]()
    {
      static std::size_t call{0}; // Esto tiene que ser una muy mala practica.
      static std::size_t N = num_integration_points();
      if (is_multimaterial_)
      { // TODO: Cheks...
        // std::cout` << "Call: " << call%N << std::endl; //TODO:: Reset call when N is reached.
        return material_points_[(call++) % N].C();
      }
      else
      {
        return material_points_[0].C();
      }
    };

    DeprecatedDenseMatrix BtCB_{N, N};

    //MatView(get_C().mat_, PETSC_VIEWER_STDOUT_WORLD); 
    BtCB_ = linalg::mat_mat_PtAP(B(X), get_C());//, geometry_->detJ(X)); 
    // Considerar C(X) como funcion para integracion multimaterial.

    //std::println("********************************************************************************");
    //std::println("BtCB({0:> 2.5f}, {1:> 2.5f}, {2:> 2.5f})", X[0], X[1], X[2] );
    //MatView(BtCB_.mat_, PETSC_VIEWER_STDOUT_WORLD); 
    //std::println("********************************************************************************");
    return BtCB_; // B^t * C * B
  };

  DeprecatedDenseMatrix K()
  {
    //bool tests_on = true
    auto N = static_cast<PetscInt>(ndof * num_nodes());
    DeprecatedDenseMatrix K{N, N};
    K = geometry_->integrate([this](const Array &X){return BtCB(X);});

    //if (tests_on){
    //  PetscBool IsSymmetric;
    //  MatIsSymmetric(K.mat_, 1.0e-6, &IsSymmetric);
    //  if (IsSymmetric == PETSC_FALSE) std::println(" WARNING: ContinuumElement::K is not symmetric. ");
    //  MatView(K.mat_, PETSC_VIEWER_STDOUT_WORLD); // Spy view (draw) of the matrix
    //}

    // MatView(K.mat_, PETSC_VIEWER_DRAW_WORLD); // Spy view (draw) of the matrix
    return K;
  };

  // ==============================================================================================
  // ==============================================================================================

  // template<typename M>
  // void inject_K(M model){ // TODO: Constrain with concept
  // };

  // Inject (BUILD ) K into global stiffness matrix
  void inject_K([[maybe_unused]] PETScMatrix &model_K)
  { // No se pasa K sino el modelo_? TER EL PLEX.

    std::vector<PetscInt> idxs;

    for (std::size_t i = 0; i < num_nodes(); ++i)
    {
      for (const auto idx : geometry_->node_p(i).dof_index())
      {
        idxs.push_back(idx);
      }
    }

    // Print DOFs index in order

    std::println("DOFs index in order: ");
    for (const auto idx : idxs)
      std::cout << idx << " ";
    std::println(" ");

    MatSetValuesLocal(model_K, idxs.size(), idxs.data(), idxs.size(), idxs.data(), this->K().data(), ADD_VALUES);
  };

  ContinuumElement() = delete;

  ContinuumElement(ElementGeometry<dim> *geometry) : geometry_{geometry} {
                                                       // M[etodo para setear materiales debe ser llamado despues de la creacion de los elementos.
                                                     };

  ContinuumElement(ElementGeometry<dim> *geometry, Material material) : geometry_{geometry}
  {
    // set_num_dof_in_nodes();
    for (std::size_t i = 0; i < geometry_->num_integration_points(); ++i)
    {
      material_points_.reserve(geometry_->num_integration_points());
      material_points_.emplace_back(MaterialPoint{material});
    }
    bind_integration_points(); // its not nedded here. Move and allocate when needed (TODO).
  };

  ~ContinuumElement() = default;

}; // ContinuumElement

//==================================================================================================
//======================== Methods Definitions =====================================================
//==================================================================================================

// YA ACA SE ESTA ASUMIENDO EL NUMERO DE GRADOS DE LIBERTAD! TAL VEZ SE DEBA MOVER AL MATERIAL O AL MODEL POLICY.
template <typename MaterialPolicy, std::size_t ndof>
inline DeprecatedDenseMatrix ContinuumElement<MaterialPolicy, ndof>::H(const Array &X)
{
  DeprecatedDenseMatrix H(ndof, num_nodes() * dim);
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

template <typename MaterialPolicy, std::size_t ndof>
inline DeprecatedDenseMatrix ContinuumElement<MaterialPolicy, ndof>::B(const Array &X)
{
  DeprecatedDenseMatrix B(num_strains, num_nodes() * dim);
  B.assembly_begin();

  if constexpr (dim == 1)
    std::runtime_error("1D material not implemented yet for ContinuumElement.");
  else if constexpr (dim == 2)
  {
    // Ordering defined acoording to Voigt notation [11, 22, 12]
    auto k = 0;
    for (std::size_t i = 0; i < num_nodes(); ++i)
    {
      B.insert_values(0, k, geometry_->dH_dx(i, 0, X));
      B.insert_values(0, k + 1, 0.0);

      B.insert_values(1, k, 0.0);
      B.insert_values(1, k + 1, geometry_->dH_dx(i, 1, X));

      B.insert_values(2, k, 1 * geometry_->dH_dx(i, 1, X));
      B.insert_values(2, k + 1, 1 * geometry_->dH_dx(i, 0, X));

      k += dim;
    }
  }
  else if constexpr (dim == 3)
  {
    
    bool test = true;

    auto k = 0;

    //if (test){
    //  std::println( " ===============================================================================================  ");
    //  std::println( " Point X = {0:> 2.5f} {1:> 2.5f} {2:> 2.5f} ", X[0], X[1], X[2]); 
    //}

    for (std::size_t i = 0; i < num_nodes(); ++i)
    {
      // Ordering defined acoording to Voigt notation [00, 11, 22, 21, 20, 10]
      B.insert_values(0, k    , geometry_->dH_dx(i, 0, X));
      B.insert_values(0, k + 1, 0.0);
      B.insert_values(0, k + 2  , 0.0);

      B.insert_values(1, k    , 0.0);
      B.insert_values(1, k + 1, geometry_->dH_dx(i, 1, X));
      B.insert_values(1, k + 2, 0.0);

      B.insert_values(2, k, 0.0);
      B.insert_values(2, k + 1, 0.0);
      B.insert_values(2, k + 2, geometry_->dH_dx(i, 2, X));

      B.insert_values(3, k    , 0.0);
      B.insert_values(3, k + 1, geometry_->dH_dx(i, 2, X));
      B.insert_values(3, k + 2, geometry_->dH_dx(i, 1, X));

      B.insert_values(4, k    , geometry_->dH_dx(i, 2, X));
      B.insert_values(4, k + 1, 0.0);
      B.insert_values(4, k + 2, geometry_->dH_dx(i, 0, X));

      B.insert_values(5, k     ,geometry_->dH_dx(i, 1, X));
      B.insert_values(5, k + 1, geometry_->dH_dx(i, 0, X));
      B.insert_values(5, k + 2, 0.0);

      k += dim;

      if (!test){
        
        std::println( "dH{0:>1}_dx: {1:> 2.5f} dH{0:>1}_dy: {2:> 2.5f} dH{0:>1}_dz: {3:> 2.5f} ",
           i, 
           geometry_->dH_dx(i, 0, X),
           geometry_->dH_dx(i, 1, X),
           geometry_->dH_dx(i, 2, X));
    
        //std::array<PetscInt, 24> idx = {0,1,2, 3,4,5, 6,7,8, 9,10,11, 12,13,14, 15,16,17, 18,19,20, 21,22,23};
        //auto test_data = std::array<double,24>{0,0,0, 0,0,0, 0,0,0, 0,0,0, 1,0,0, 1,0,0, 1,0,0, 1,0,0 }; 
        //Vec U, e;
        //VecCreateSeq(PETSC_COMM_WORLD, 24, &U);
        //VecCreateSeq(PETSC_COMM_WORLD, 6, &e);
        //VecSet(e, 0.0);
        //VecSetValues(U, 24, idx.data(), test_data.data(), INSERT_VALUES);
        //VecAssemblyBegin(U);
        //VecAssemblyEnd(U);
        //MatMult(B.mat_, U, e);
        //PetscScalar *array;
        //VecGetArray(e, &array);
        //std::println( "e{0:>1}    : \n  {1:> 2.5f}\n  {2:> 2.5f}\n  {3:> 2.5f}\n  {4:> 2.5f}\n  {5:> 2.5f}\n  {6:> 2.5f} ",
        //   i, 
        //   array[0], array[1], array[2], array[3], array[4], array[5]);
        //}
        //std::println( " --------------------------------------------------------------  ");
      }
    }

    B.assembly_end();
    //MatView(B.mat_, PETSC_VIEWER_STDOUT_WORLD);
    return B;
  }
};

#endif