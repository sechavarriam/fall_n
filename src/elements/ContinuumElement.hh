#ifndef FN_CONTINUUM_ELEMENT
#define FN_CONTINUUM_ELEMENT

#include <memory>
#include <array>

#ifdef __clang__ 
  #include <format>
  #include <print>
#endif

#include <Eigen/Dense>
#include <petsc.h>

#include "element_geometry/ElementGeometry.hh"

#include "../model/MaterialPoint.hh"
#include "../materials/Material.hh"

template <typename MaterialPolicy, std::size_t ndof>
class ContinuumElement
{
  using PETScMatrix = Mat; // TODO: Use PETSc DeprecatedDenseMatrix

  // using MaterialPolicy = MaterialPolicy;
  using MaterialPointT = MaterialPoint<MaterialPolicy>;
  using MaterialT      = Material     <MaterialPolicy>;
  using Array          = std::array<double, MaterialPolicy::dim>;

  static constexpr auto dim         = MaterialPolicy::dim;
  static constexpr auto num_strains = MaterialPolicy::StrainType::num_components;

  using StrainMatrixT = Eigen::Matrix<double, num_strains, Eigen::Dynamic>;
  using InterpMatrixT = Eigen::Matrix<double, dim        , Eigen::Dynamic>;


  ElementGeometry<dim>*       geometry_         ;
  std::vector<MaterialPointT> material_points_{};

  bool is_multimaterial_{true}; // If true, the element has different materials in each integration point.

public:
  constexpr auto get_geometry() const noexcept { return geometry_; };

  constexpr auto sieve_id() const noexcept { return geometry_->sieve_id.value(); };
  constexpr auto node_p(std::size_t i) const noexcept { return geometry_->node_p(i); };

  constexpr void set_num_dof_in_nodes() noexcept{
    for (std::size_t i = 0; i < num_nodes(); ++i)
      geometry_->node_p(i).set_num_dof(dim);
  };

  constexpr inline auto num_integration_points() const noexcept { return geometry_->num_integration_points(); };

  constexpr inline auto bind_integration_points() noexcept{
    std::size_t count{0};
    for (auto &point : material_points_){
      point.bind_integration_point(geometry_->integration_point_[count++]);
    }
  };

  constexpr auto num_nodes() const noexcept { return geometry_->num_nodes(); };

  constexpr auto get_dofs_index() const noexcept{
    std::size_t i;

    auto N = num_nodes();
    std::vector<std::span<std::size_t>> dofs_index; // This thing avoids the copy of the vector.
    dofs_index.reserve(N);

    for (i = 0; i < N; ++i)
      dofs_index.emplace_back(geometry_->node(i).dof_index());

    return dofs_index;
  };
  
  inline InterpMatrixT H(const Array &X){
    std::size_t i, k = 0;
    InterpMatrixT H = InterpMatrixT::Zero(dim, ndof * num_nodes());

    if constexpr (dim == 1){
      for (i = 0; i < num_nodes(); ++i)
        H(0, i) = geometry_->H(i, X);
    }
    else if constexpr (dim == 2){
      for (i = 0; i < num_nodes(); ++i){
        H(0, k) = geometry_->H(i, X);
        H(1, k + 1) = geometry_->H(i, X);
        k += dim;
      }
    }
    else if constexpr (dim == 3){
      for (i = 0; i < num_nodes(); ++i){
        H(0, k) = geometry_->H(i, X);
        H(1, k + 1) = geometry_->H(i, X);
        H(2, k + 2) = geometry_->H(i, X);
        k += dim;
      }
    }
    return H;
  };

  // This coul be injected in terms of the material policy StrainDifferentialOperator or something like that.
  inline StrainMatrixT B(const Array &X){
    StrainMatrixT B = StrainMatrixT::Zero(num_strains, ndof * num_nodes());

    std::size_t i, k = 0;

    if constexpr (dim == 1){
      std::runtime_error("B not implemented for dim = 1 yet ");
    }
    else if constexpr (dim == 2){
      for (i = 0; i < num_nodes(); ++i){
        B(0, k    ) = geometry_->dH_dx(i, 0, X);
        B(1, k + 1) = geometry_->dH_dx(i, 1, X);
        B(2, k    ) = geometry_->dH_dx(i, 1, X);
        B(2, k + 1) = geometry_->dH_dx(i, 0, X);
        k += dim;
      }
    }
    else if constexpr (dim == 3){
      for (i = 0; i < num_nodes(); ++i){
        B(0, k    ) = geometry_->dH_dx(i, 0, X);
        B(1, k + 1) = geometry_->dH_dx(i, 1, X);
        B(2, k + 2) = geometry_->dH_dx(i, 2, X);
        B(3, k + 1) = geometry_->dH_dx(i, 2, X);
        B(3, k + 2) = geometry_->dH_dx(i, 1, X);
        B(4, k    ) = geometry_->dH_dx(i, 2, X);
        B(4, k + 2) = geometry_->dH_dx(i, 0, X);        
        B(5, k    ) = geometry_->dH_dx(i, 1, X);
        B(5, k + 1) = geometry_->dH_dx(i, 0, X);
        k += dim;
      }
    }
    return B;
  }

  inline Eigen::MatrixXd BtCB(const Array &X){
    auto get_C = [this](){
      static std::size_t call{0}; // Esto tiene que ser una muy mala practica.
      static std::size_t N = num_integration_points(); 
      if (is_multimaterial_){ // TODO: Cheks...
        return material_points_[(call++) % N].C();
      }
      else{
        return material_points_[0].C();
      }
    };
    return B(X).transpose() * get_C() * B(X);
  };

  Eigen::MatrixXd K(){ return geometry_->integrate([this](const Array &X)->Eigen::MatrixXd{return BtCB(X);});};

  // template<typename M>
  // void inject_K(M model){ // TODO: Constrain with concept };

  void inject_K([[maybe_unused]] PETScMatrix &model_K){ // No se pasa K sino el modelo_? TER EL PLEX.
    std::vector<PetscInt> idxs;
    for (std::size_t i = 0; i < num_nodes(); ++i){
      for (const auto idx : geometry_->node_p(i).dof_index()){
        idxs.push_back(idx);
      }
    }

    #ifdef __clang__ 
      std::println("DOFs index in order: ");
      for (const auto idx : idxs)
        std::cout << idx << " ";
      std::println(" ");
    #endif

    MatSetValuesLocal(model_K, idxs.size(), idxs.data(), idxs.size(), idxs.data(), this->K().data(), ADD_VALUES);
  };

  ContinuumElement() = delete;
  ContinuumElement(ElementGeometry<dim> *geometry) : geometry_{geometry} {
                                                       // M[etodo para setear materiales debe ser llamado despues de la creacion de los elementos.
  }

  ContinuumElement(ElementGeometry<dim> *geometry, MaterialT material) : geometry_{geometry}
  {
    // set_num_dof_in_nodes();
    for (std::size_t i = 0; i < geometry_->num_integration_points(); ++i){
      material_points_.reserve(geometry_->num_integration_points());
      material_points_.emplace_back(MaterialPointT{material});
    }
    bind_integration_points(); // its not nedded here. Move and allocate when needed (TODO).
  };

  ~ContinuumElement() = default;

}; // ContinuumElement


#endif