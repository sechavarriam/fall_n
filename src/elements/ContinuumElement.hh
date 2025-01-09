#ifndef FN_CONTINUUM_ELEMENT
#define FN_CONTINUUM_ELEMENT

#include <memory>
#include <array>

#include "element_geometry/ElementGeometry.hh"

#include "../model/MaterialPoint.hh"
#include "../materials/Material.hh"
#include "../numerics/linear_algebra/LinalgOperations.hh"

template <typename MaterialPolicy, std::size_t ndof>
class ContinuumElement
{
  using PETScMatrix = Mat; //TODO: Use PETSc Matrix

  //using MaterialPolicy = MaterialPolicy;
  using MaterialPoint = MaterialPoint<MaterialPolicy>;
  using Material      = Material     <MaterialPolicy>;

  using Array = std::array<double, MaterialPolicy::dim>;

  static constexpr auto dim         = MaterialPolicy::dim;
  static constexpr auto num_strains = MaterialPolicy::StrainType::num_components;
 
  ElementGeometry<dim>       *geometry_;
  std::vector<MaterialPoint> material_points_{};

  bool is_multimaterial_{true}; // If true, the element has different materials in each integration point.

public:

  constexpr auto get_geometry() const noexcept { return geometry_; };

  constexpr auto sieve_id() const noexcept { return geometry_->sieve_id.value(); };
  constexpr auto node_p (std::size_t i) const noexcept { return geometry_->node_p(i); };

  //constexpr PetscInt dofs_at_node([[maybe_unused]]std::size_t i) const noexcept { return dim; };

  constexpr void set_num_dof_in_nodes() noexcept {
    for (std::size_t i = 0; i < num_nodes(); ++i){
      geometry_->node_p(i).set_num_dof(dim);
    }
  };


  constexpr inline auto num_integration_points() const noexcept { return geometry_->num_integration_points(); };

  constexpr inline auto bind_integration_points() noexcept {
    std::size_t count{0};
    for (auto& point : material_points_){
      point.bind_integration_point(geometry_->integration_point_[count++]);
    }
  };

  //constexpr auto set_material_point_coordinates(){
  //  for(auto& gauss_point : material_points_){
  //    gauss_point.set_coord(geometry_->map_local_point(gauss_point.coord()));
  //  } 
  //};



  constexpr auto num_nodes() const noexcept {
     //std::cout << "Num nodes: " << geometry_->num_nodes() << std::endl;
     return geometry_->num_nodes(); 
     };

  constexpr auto get_dofs_index() const noexcept {
    auto N = num_nodes();
    std::vector<std::span<std::size_t>> dofs_index; //This thing avoids the copy of the vector.
    dofs_index.reserve(N);

    for (std::size_t i = 0; i < N; ++i){
      dofs_index.emplace_back(geometry_->node(i).dof_index());
    }
    return dofs_index;
    };

  Matrix H(const Array &X); // Declaration. Definition at the end of the file // Could be injected from material policy.
  Matrix B(const Array &X); // Declaration. Definition at the end of the file // Could be injected from material policy.
  
  Matrix BtCB (const Array &X) {  // TODO: Optimize this for each dimension.

    auto get_C = [this](){
      static std::size_t call{0}; //Esto tiene que ser una muy mala practica.
      static std::size_t N = num_integration_points();
        if (is_multimaterial_){ // TODO: Cheks...
          //std::cout` << "Call: " << call%N << std::endl; //TODO:: Reset call when N is reached.
          return material_points_[(call++)%N].C();
        }else{
          return material_points_[0].C();
        }
    };

    Matrix BtCB_{ndof, ndof};                          
    BtCB_= linalg::mat_mat_PtAP(B(X), get_C()); // Considerar C(X) como funcion para integracion multimaterial.
    return  BtCB_; // B^t * C * B
  }; 

  Matrix K() {
    Matrix K{ndof, ndof};
    
    K=geometry_->integrate([this](const Array &X){
        return BtCB(X);
      }
    );
    
    //MatView(K.mat_, PETSC_VIEWER_DRAW_WORLD); // Spy view (draw) of the matrix
    return K;
  };

  // ==============================================================================================
  // ==============================================================================================

  //template<typename M>
  //void inject_K(M model){ // TODO: Constrain with concept
  //};

  void inject_K([[maybe_unused]]PETScMatrix& model_K){ // No se pasa K sino el modelo_? TER EL PLEX.
      // Inject (BUILD ) K into global stiffness matrix
      std::vector<PetscInt> idxs;
      for (std::size_t i = 0; i < num_nodes(); ++i){
        for (const auto idx : geometry_->node_p(i).dof_index()){
          idxs.push_back(idx);
        }
      }
      MatSetValuesLocal(model_K, idxs.size(), idxs.data(), idxs.size(), idxs.data(), this->K().data(), ADD_VALUES);
  };

  ContinuumElement() = default;

  ContinuumElement(ElementGeometry<dim> *geometry) : geometry_{geometry} {};

  ContinuumElement(ElementGeometry<dim> *geometry, Material material) : geometry_{geometry}
  {
    //set_num_dof_in_nodes();
    for (std::size_t i = 0; i < geometry_->num_integration_points(); ++i){
      material_points_.reserve(geometry_->num_integration_points());
      material_points_.emplace_back(MaterialPoint{material});
    }
    bind_integration_points();// its not nedded here. Move and allocate when needed (TODO).
  };

  ~ContinuumElement() = default;

}; // ContinuumElement

//==================================================================================================
//======================== Methods Definitions ===================================================
//==================================================================================================


// YA ACA SE ESTA ASUMIENDO EL NUMERO DE GRADOS DE LIBERTAD! TAL VEZ SE DEBA MOVER AL MATERIAL O AL MODEL POLICY.
template <typename MaterialPolicy, std::size_t ndof>
inline Matrix ContinuumElement<MaterialPolicy, ndof>::H(const Array &X)
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


template <typename MaterialPolicy, std::size_t ndof>
inline Matrix ContinuumElement<MaterialPolicy, ndof>::B(const Array &X)
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


#endif