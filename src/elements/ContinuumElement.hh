#ifndef FN_CONTINUUM_ELEMENT
#define FN_CONTINUUM_ELEMENT

#include <memory>
#include <array>
#include <span>

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
  // ========================= Types and Static Constant Definitions =================================

  using StateVariableT = typename MaterialPolicy::StateVariableT;

  using MaterialPointT = MaterialPoint<MaterialPolicy>;
  using MaterialT      = Material     <MaterialPolicy>;
  using Array          = std::array<double, MaterialPolicy::dim>;


  static constexpr auto dim         = MaterialPolicy::dim;
  static constexpr auto topological_dim = dim; // continuum elements are full-dimensional, so the topological dimension is the same as the spatial dimension.

  static constexpr auto num_strains = MaterialPolicy::StrainT::num_components;

  using StrainMatrixT = Eigen::Matrix<double, num_strains, Eigen::Dynamic>;
  using InterpMatrixT = Eigen::Matrix<double, dim        , Eigen::Dynamic>;

  // =================================================================================================

  ElementGeometry<dim>*       geometry_         ;
  std::vector<MaterialPointT> material_points_{};

  // ── DOF index cache (flat array, lazily populated) ────────────────────
  std::vector<PetscInt> dof_indices_;
  bool                  dofs_cached_{false};

  void ensure_dof_cache() noexcept {
      if (dofs_cached_) return;
      collect_dof_indices();
  }

  void collect_dof_indices() noexcept {
      const auto total = ndof * num_nodes();
      dof_indices_.clear();
      dof_indices_.reserve(total);
      for (std::size_t i = 0; i < num_nodes(); ++i)
          for (const auto idx : geometry_->node_p(i).dof_index())
              dof_indices_.push_back(idx);
      dofs_cached_ = true;
  }

  // Invalidate cache (call after DOF renumbering, e.g. Cuthill-McKee)
  void invalidate_dof_cache() noexcept { dofs_cached_ = false; }

  // Helper: extract element DOFs from a local PETSc vector
  Eigen::VectorXd extract_element_dofs(Vec u_local) {
      ensure_dof_cache();
      const auto n = static_cast<PetscInt>(dof_indices_.size());
      Eigen::VectorXd u_e(n);
      VecGetValues(u_local, n, dof_indices_.data(), u_e.data());
      return u_e;
  }

private:

public:

  constexpr auto material_points() const noexcept { return material_points_; };
  constexpr auto get_material_point(std::size_t i) noexcept { return material_points_[i]; };

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

  
  inline InterpMatrixT H(const Array &X){
    std::size_t i, k = 0;
    InterpMatrixT H = InterpMatrixT::Zero(dim, ndof * num_nodes());

    if constexpr (dim == 1){
      for (i = 0; i < num_nodes(); ++i)
        H(0, i) = geometry_->H(i, X);
    }
    else if constexpr (dim == 2){
      for (i = 0; i < num_nodes(); ++i){
        H(0, k    ) = geometry_->H(i, X);
        H(1, k + 1) = geometry_->H(i, X);
        k += dim;
      }
    }
    else if constexpr (dim == 3){
      for (i = 0; i < num_nodes(); ++i){
        H(0, k    ) = geometry_->H(i, X);
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

  inline Eigen::MatrixXd BtCB(std::size_t gp, const Array &X){
    auto B_x = B(X);
    return B_x.transpose() * material_points_[gp].C() * B_x;
  };

  Eigen::MatrixXd K(){
    const auto n = ndof * num_nodes();
    Eigen::MatrixXd K_e = Eigen::MatrixXd::Zero(n, n);

    for (std::size_t gp = 0; gp < num_integration_points(); ++gp) {
        auto   ref_pt = geometry_->reference_integration_point(gp);
        double w      = geometry_->weight(gp);
        double Jdet   = geometry_->detJ(ref_pt);

        Array Xi{};
        for (std::size_t k = 0; k < dim; ++k) Xi[k] = ref_pt[k];

        K_e += w * Jdet * BtCB(gp, Xi);
    }
    return K_e;
  };

  void inject_K(Mat &model_K){
    ensure_dof_cache();
    auto K_e = K();
    const auto n = static_cast<PetscInt>(dof_indices_.size());
    MatSetValuesLocal(model_K, n, dof_indices_.data(), n, dof_indices_.data(), K_e.data(), ADD_VALUES);
  };

  // ========================== Nonlinear element operations ================================

  // Compute internal force vector:  f_int_e = Σ_gp  w·|J|·Bᵀ·σ(ε(u))
  // The stress σ is computed through the material's Strategy:
  //   material_point.compute_response(ε) → Strategy.compute_response(model, ε)
  void compute_internal_forces(Vec u_local, Vec f_local) {
      Eigen::VectorXd u_e = extract_element_dofs(u_local);
      Eigen::VectorXd f_e = Eigen::VectorXd::Zero(dim * num_nodes());

      for (std::size_t gp = 0; gp < num_integration_points(); ++gp) {
          auto   ref_pt = geometry_->reference_integration_point(gp);
          double w      = geometry_->weight(gp);
          double Jdet   = geometry_->detJ(ref_pt);

          Array Xi{};
          for (std::size_t k = 0; k < dim; ++k) Xi[k] = ref_pt[k];

          auto B_x = B(Xi);
          StateVariableT strain;
          strain.set_strain(B_x * u_e);

          auto sigma = material_points_[gp].compute_response(strain);
          f_e += w * Jdet * (B_x.transpose() * sigma.components());
      }

      ensure_dof_cache();
      VecSetValues(f_local, static_cast<PetscInt>(dof_indices_.size()),
                   dof_indices_.data(), f_e.data(), ADD_VALUES);
  }

  // Assemble tangent stiffness:  K_e = Σ_gp  w·|J|·Bᵀ·C_t(ε(u))·B
  // The tangent C_t is computed through the material's Strategy:
  //   material_point.tangent(ε) → Strategy.tangent(model, ε)
  void inject_tangent_stiffness(Vec u_local, Mat J_mat) {
      Eigen::VectorXd u_e = extract_element_dofs(u_local);
      Eigen::MatrixXd K_e = Eigen::MatrixXd::Zero(dim * num_nodes(), dim * num_nodes());

      for (std::size_t gp = 0; gp < num_integration_points(); ++gp) {
          auto   ref_pt = geometry_->reference_integration_point(gp);
          double w      = geometry_->weight(gp);
          double Jdet   = geometry_->detJ(ref_pt);

          Array Xi{};
          for (std::size_t k = 0; k < dim; ++k) Xi[k] = ref_pt[k];

          auto B_x = B(Xi);
          StateVariableT strain;
          strain.set_strain(B_x * u_e);

          auto C_t = material_points_[gp].tangent(strain);
          K_e += w * Jdet * (B_x.transpose() * C_t * B_x);
      }

      ensure_dof_cache();
      const auto n = static_cast<PetscInt>(dof_indices_.size());
      MatSetValuesLocal(J_mat, n, dof_indices_.data(), n, dof_indices_.data(),
                        K_e.data(), ADD_VALUES);
  }

  // Commit material state at all Gauss points after global convergence.
  // Calls Strategy.commit(model, ε) to evolve internal variables.
  void commit_material_state(Vec u_local) {
      Eigen::VectorXd u_e = extract_element_dofs(u_local);

      for (std::size_t gp = 0; gp < num_integration_points(); ++gp) {
          auto ref_pt = geometry_->reference_integration_point(gp);
          Array Xi{};
          for (std::size_t k = 0; k < dim; ++k) Xi[k] = ref_pt[k];

          StateVariableT strain;
          strain.set_strain(B(Xi) * u_e);

          material_points_[gp].commit(strain);
          material_points_[gp].update_state(strain);
      }
  }

// =================================== Solution manipulation =====================================


  auto get_current_state(const auto &model) noexcept{ // CONSTRAIN WITH MODEL CONCEPT
    ensure_dof_cache();
    std::vector<PetscScalar> u(dof_indices_.size());
    VecGetValues(model.current_state, static_cast<PetscInt>(dof_indices_.size()),
                 dof_indices_.data(), u.data());
    return u;
  };

  // Templatize this method with an AnalisisT concept
  auto compute_strain(const Array &X, const auto &model) noexcept{
    typename MaterialPolicy::StateVariableT e_h;
    using EigenMap = Eigen::Map<Eigen::Vector<double,Eigen::Dynamic>>;

    std::ranges::contiguous_range auto u = get_current_state(model);
    EigenMap u_h(u.data(), static_cast<Eigen::Index>(u.size()));

    e_h.set_strain(B(X)*u_h);

    return e_h; 
  };

  void set_material_point_state(const auto &model) noexcept{ // CONSTRAIN WITH MODEL CONCEPT
    for (auto &point : material_points_) point.update_state(compute_strain(point.coord(), model));
  };

  //void get_nodal_strains(){}; //Esto no se puede tan facil si el elemento es multimaterial.
  
  // ================================= Constructors and Destructor =================================
  ContinuumElement() = delete;
  ContinuumElement(ElementGeometry<dim> *geometry) : geometry_{geometry} {}; // Metodo para setear materiales debe ser llamado despues de la creacion de los elementos.

  ContinuumElement(ElementGeometry<dim> *geometry, MaterialT material) : geometry_{geometry}{
    const auto n_gp = geometry_->num_integration_points();
    material_points_.reserve(n_gp);
    for (std::size_t i = 0; i < n_gp; ++i){
      material_points_.emplace_back(MaterialPointT{material});
    }
    bind_integration_points();
  };

  ~ContinuumElement() = default;
}; // ContinuumElement


#endif // FN_CONTINUUM_ELEMENT