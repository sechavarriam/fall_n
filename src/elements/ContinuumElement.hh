#ifndef FN_CONTINUUM_ELEMENT
#define FN_CONTINUUM_ELEMENT

#include <memory>
#include <array>
#include <span>
#include <vector>

#ifdef __clang__ 
  #include <format>
  #include <print>
#endif

#include <Eigen/Dense>
#include <petsc.h>

#include "element_geometry/ElementGeometry.hh"

#include "../model/MaterialPoint.hh"
#include "../materials/Material.hh"
#include "../continuum/ConstitutiveKinematics.hh"
#include "../continuum/KinematicPolicy.hh"

template <typename MaterialPolicy, std::size_t ndof,
          typename KinematicPolicy = continuum::SmallStrain>
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

  double density_{0.0};  // Mass density ρ [kg/m³] — needed for dynamic analysis

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

public:
  // Helper: extract element DOFs from a local PETSc vector.
  // Thread-safe for concurrent reads on a shared local Vec.
  Eigen::VectorXd extract_element_dofs(Vec u_local) {
      ensure_dof_cache();
      const auto n = static_cast<PetscInt>(dof_indices_.size());
      Eigen::VectorXd u_e(n);
      VecGetValues(u_local, n, dof_indices_.data(), u_e.data());
      return u_e;
  }

private:

public:

  constexpr const auto& material_points() const noexcept { return material_points_; };
  constexpr       auto& material_points()       noexcept { return material_points_; };

  constexpr const auto& get_material_point(std::size_t i) const noexcept { return material_points_[i]; };
  constexpr       auto& get_material_point(std::size_t i)       noexcept { return material_points_[i]; };

  constexpr auto get_geometry() const noexcept { return geometry_; };

  constexpr auto sieve_id() const noexcept { return geometry_->sieve_id(); };
  constexpr auto node_p(std::size_t i) const noexcept { return geometry_->node_p(i); };

  const std::string& physical_group() const noexcept { return geometry_->physical_group(); }
  bool has_physical_group() const noexcept { return geometry_->has_physical_group(); }

  // ── Post-processing: type-erased Gauss-point field export ──────
  //
  //  Returns stress, strain, and internal-state snapshots for each
  //  integration point.  Used by FEM_Element::collect_gauss_fields()
  //  to enable VTK material-field export for MultiElementPolicy models.

  std::vector<GaussFieldRecord> collect_gauss_fields(Vec /*u_local*/) const {
      std::vector<GaussFieldRecord> records;
      records.reserve(material_points_.size());
      for (const auto& mp : material_points_) {
          GaussFieldRecord rec;

          const auto& state = mp.current_state();
          rec.strain.assign(state.data(), state.data() + num_strains);

          auto sigma = mp.compute_response(state);
          rec.stress.assign(sigma.data(), sigma.data() + num_strains);

          auto snap = mp.internal_field_snapshot();

          // Deep-copy plastic_strain so the record is self-contained
          if (snap.has_plastic_strain()) {
              auto sp = snap.plastic_strain.value();
              rec.pstrain_storage.assign(sp.begin(), sp.end());
              snap.plastic_strain = std::span<const double>{
                  rec.pstrain_storage.data(), rec.pstrain_storage.size()};
          }

          rec.snapshot = std::move(snap);
          records.push_back(std::move(rec));
      }
      return records;
  }

  constexpr void set_num_dof_in_nodes() noexcept{
    for (std::size_t i = 0; i < num_nodes(); ++i)
      geometry_->node_p(i).set_num_dof(dim);
  };

  constexpr inline auto num_integration_points() const noexcept { return geometry_->num_integration_points(); };

  constexpr inline auto bind_integration_points() noexcept{
    std::size_t count{0};
    for (auto &point : material_points_){
      point.bind_integration_point(geometry_->integration_points()[count++]);
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

  // ── Linear B matrix — delegated to KinematicPolicy ──────────────────────
  //
  // For SmallStrain:       Standard ∇ˢ operator (identical to old hardcoded B).
  // For TotalLagrangian:   SmallStrain B is reused as initial-stiffness B.
  //
  // Note: for nonlinear formulations, the displacement-dependent B_NL is
  // obtained through KinematicPolicy::evaluate(), not through this method.
  //
  inline StrainMatrixT B(const Array &X){
    return KinematicPolicy::template compute_B<dim>(geometry_, num_nodes(), ndof, X);
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
        double Jdet   = geometry_->differential_measure(ref_pt);

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

  // ── Pure-compute methods (no PETSc writes — thread-safe) ──────────────────
  //
  //  These take pre-extracted element DOFs (Eigen::VectorXd) and return
  //  element-level contributions as Eigen objects. They perform NO writes
  //  to PETSc Mat/Vec, enabling parallel computation across elements
  //  while the injection phase remains sequential.

  /// Compute element internal force vector:  f_e = Σ_gp  w·|J|·Bᵀ·σ(ε(u))
  ///
  /// Thread-safe: reads only from element-local geometry and material state.
  Eigen::VectorXd compute_internal_force_vector(const Eigen::VectorXd& u_e) {
      Eigen::VectorXd f_e = Eigen::VectorXd::Zero(dim * num_nodes());

      for (std::size_t gp = 0; gp < num_integration_points(); ++gp) {
          auto   ref_pt = geometry_->reference_integration_point(gp);
          double w      = geometry_->weight(gp);
          double Jdet   = geometry_->differential_measure(ref_pt);

          Array Xi{};
          for (std::size_t k = 0; k < dim; ++k) Xi[k] = ref_pt[k];

          auto kin = KinematicPolicy::template evaluate<dim>(
              geometry_, num_nodes(), ndof, Xi, u_e);
          auto constitutive_kin =
              continuum::make_constitutive_kinematics<KinematicPolicy>(kin);
          auto sigma = material_points_[gp].compute_response(constitutive_kin);
          const double volume_factor =
              KinematicPolicy::needs_current_volume_factor ? kin.detF : 1.0;
          f_e += w * Jdet * volume_factor * (kin.B.transpose() * sigma.components());
      }
      return f_e;
  }

  /// Compute element tangent stiffness:  K_e = Σ_gp  w·|J|·(Bᵀ·C_t·B + K_σ)
  ///
  /// Thread-safe: reads only from element-local geometry and material state.
  Eigen::MatrixXd compute_tangent_stiffness_matrix(const Eigen::VectorXd& u_e) {
      Eigen::MatrixXd K_e = Eigen::MatrixXd::Zero(dim * num_nodes(), dim * num_nodes());

      for (std::size_t gp = 0; gp < num_integration_points(); ++gp) {
          auto   ref_pt = geometry_->reference_integration_point(gp);
          double w      = geometry_->weight(gp);
          double Jdet   = geometry_->differential_measure(ref_pt);

          Array Xi{};
          for (std::size_t k = 0; k < dim; ++k) Xi[k] = ref_pt[k];

          auto kin = KinematicPolicy::template evaluate<dim>(
              geometry_, num_nodes(), ndof, Xi, u_e);
          auto constitutive_kin =
              continuum::make_constitutive_kinematics<KinematicPolicy>(kin);
          auto C_t = material_points_[gp].tangent(constitutive_kin);
          const double volume_factor =
              KinematicPolicy::needs_current_volume_factor ? kin.detF : 1.0;

          // Material stiffness: K_mat = ∫ Bᵀ C B dV
          K_e += w * Jdet * volume_factor * (kin.B.transpose() * C_t * kin.B);

          // Geometric stiffness K_σ (only for nonlinear formulations)
          if constexpr (KinematicPolicy::needs_geometric_stiffness) {
              auto sigma = material_points_[gp].compute_response(constitutive_kin);

              constexpr auto NV = continuum::voigt_size<dim>();
              Eigen::Vector<double, static_cast<int>(NV)> S_voigt;
              for (std::size_t i = 0; i < NV; ++i)
                  S_voigt(static_cast<Eigen::Index>(i)) = sigma[i];

              Eigen::MatrixXd K_sigma;
              if constexpr (std::same_as<KinematicPolicy, continuum::UpdatedLagrangian>) {
                  auto sigma_mat =
                      continuum::TotalLagrangian::stress_voigt_to_matrix<dim>(S_voigt);
                  auto grad_X =
                      continuum::detail::physical_gradients<dim>(geometry_, num_nodes(), Xi);
                  auto grad_x =
                      continuum::UpdatedLagrangian::compute_spatial_gradients<dim>(grad_X, kin.F);
                  K_sigma =
                      continuum::UpdatedLagrangian::compute_spatial_geometric_stiffness<dim>(
                          grad_x, ndof, sigma_mat);
              }
              else {
                  auto S_mat =
                      continuum::TotalLagrangian::stress_voigt_to_matrix<dim>(S_voigt);
                  K_sigma = KinematicPolicy::template compute_geometric_stiffness<dim>(
                      geometry_, num_nodes(), ndof, Xi, S_mat);
              }

              K_e += w * Jdet * volume_factor * K_sigma;
          }
      }
      return K_e;
  }

  /// Expose DOF index cache for external injection (parallel assembly).
  const std::vector<PetscInt>& get_dof_indices() {
      ensure_dof_cache();
      return dof_indices_;
  }

  // ========================== Mass matrix (for dynamics) ================================

  /// Set mass density [kg/m³].
  void set_density(double rho) noexcept { density_ = rho; }

  /// Get mass density.
  [[nodiscard]] double density() const noexcept { return density_; }

  /// Consistent mass matrix:  M_e = ρ · Σ_gp (w · |J| · Nᵀ · N)
  ///
  /// where N = H(ξ) is the interpolation matrix (dim × ndof·nnodes).
  /// Thread-safe: reads only from element-local geometry.
  Eigen::MatrixXd compute_consistent_mass_matrix() {
      const auto n = ndof * num_nodes();
      Eigen::MatrixXd M_e = Eigen::MatrixXd::Zero(n, n);

      if (density_ <= 0.0) return M_e;  // no mass if density not set

      for (std::size_t gp = 0; gp < num_integration_points(); ++gp) {
          auto   ref_pt = geometry_->reference_integration_point(gp);
          double w      = geometry_->weight(gp);
          double Jdet   = geometry_->differential_measure(ref_pt);

          Array Xi{};
          for (std::size_t k = 0; k < dim; ++k) Xi[k] = ref_pt[k];

          auto N = H(Xi);  // dim × (ndof * num_nodes)
          M_e += density_ * w * Jdet * (N.transpose() * N);
      }
      return M_e;
  }

  /// Lumped mass vector (row-sum of consistent mass matrix).
  ///
  /// Returns a vector of size ndof*nnodes where each entry is the
  /// diagonal mass assigned to that DOF.  This is the simplest
  /// lumping scheme; for better accuracy, use HRZ lumping.
  Eigen::VectorXd compute_lumped_mass_vector() {
      auto M_c = compute_consistent_mass_matrix();
      return M_c.rowwise().sum();
  }

  /// Inject consistent mass matrix into a PETSc Mat.
  void inject_mass(Mat M) {
      ensure_dof_cache();
      auto M_e = compute_consistent_mass_matrix();
      if (M_e.isZero()) return;  // skip if no density
      const auto n = static_cast<PetscInt>(dof_indices_.size());
      MatSetValuesLocal(M, n, dof_indices_.data(), n, dof_indices_.data(),
                        M_e.data(), ADD_VALUES);
  }

  // ── Legacy PETSc-injecting methods (delegate to pure-compute) ─────────────

  // Compute internal force vector:  f_int_e = Σ_gp  w·|J|·Bᵀ·σ(ε(u))
  //
  // For SmallStrain:     B is the standard linear operator, ε = B·u.
  // For TotalLagrangian: B_NL(F) is the nonlinear operator, E = B_NL·u.
  //
  // The stress σ (or S for TL) is computed through the material's Strategy:
  //   material_point.compute_response(ε) → Strategy.compute_response(model, ε)
  void compute_internal_forces(Vec u_local, Vec f_local) {
      Eigen::VectorXd u_e = extract_element_dofs(u_local);
      Eigen::VectorXd f_e = compute_internal_force_vector(u_e);

      ensure_dof_cache();
      VecSetValues(f_local, static_cast<PetscInt>(dof_indices_.size()),
                   dof_indices_.data(), f_e.data(), ADD_VALUES);
  }

  // Assemble tangent stiffness:  K_e = Σ_gp  w·|J|·Bᵀ·C_t(ε(u))·B
  //
  // For SmallStrain:     K = ∫ Bᵀ C B dV          (no geometric stiffness)
  // For TotalLagrangian: K = ∫ B_NLᵀ C B_NL dV₀ + K_σ
  //                      where K_σ = ∫ Σ_{IJ} (g_Iᵀ·S·g_J)·I_dim dV₀
  //
  void inject_tangent_stiffness(Vec u_local, Mat J_mat) {
      Eigen::VectorXd u_e = extract_element_dofs(u_local);
      Eigen::MatrixXd K_e = compute_tangent_stiffness_matrix(u_e);

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

          // Evaluate kinematics through the policy
          auto kin = KinematicPolicy::template evaluate<dim>(
              geometry_, num_nodes(), ndof, Xi, u_e);
          auto constitutive_kin =
              continuum::make_constitutive_kinematics<KinematicPolicy>(kin);
          material_points_[gp].commit(constitutive_kin);
          auto strain =
              continuum::make_kinematic_measure<StateVariableT>(constitutive_kin);
          material_points_[gp].update_state(std::move(strain));
      }
  }

  // Revert material state at all Gauss points to the last committed state.
  // Called on solver divergence / sub-step bisection.
  void revert_material_state() {
      for (auto& mp : material_points_) {
          mp.revert();
      }
  }

  /// Inject type-erased internal state at a specific Gauss point.
  /// @param gp   Gauss point index (0-based)
  /// @param state  StateRef referencing the correct InternalVariablesT
  void inject_material_state(std::size_t gp, impl::StateRef state) {
      material_points_[gp].inject_internal_state(state);
  }

  /// Inject the same state into ALL Gauss points.
  void inject_material_state(const std::vector<impl::StateRef>& states) {
      for (std::size_t gp = 0; gp < material_points_.size(); ++gp) {
          material_points_[gp].inject_internal_state(states[gp]);
      }
  }

  [[nodiscard]] bool supports_state_injection() const noexcept {
      if (material_points_.empty()) return false;
      return material_points_.front().supports_state_injection();
  }

// =================================== Solution manipulation =====================================


  auto get_current_state(const auto &model) noexcept{ // CONSTRAIN WITH MODEL CONCEPT
    ensure_dof_cache();
    std::vector<PetscScalar> u(dof_indices_.size());
    VecGetValues(model.state_vector(), static_cast<PetscInt>(dof_indices_.size()),
                 dof_indices_.data(), u.data());
    return u;
  };

  // Templatize this method with an AnalisisT concept
  auto compute_strain(const Array &X, const auto &model) noexcept{
    typename MaterialPolicy::StateVariableT e_h;
    using EigenMap = Eigen::Map<Eigen::Vector<double,Eigen::Dynamic>>;

    std::ranges::contiguous_range auto u = get_current_state(model);
    EigenMap u_h(u.data(), static_cast<Eigen::Index>(u.size()));

    // Delegate to KinematicPolicy — handles both linear and nonlinear strain
    auto kin = KinematicPolicy::template evaluate<dim>(
        geometry_, num_nodes(), ndof, X, u_h);
    auto constitutive_kin =
        continuum::make_constitutive_kinematics<KinematicPolicy>(kin);
    e_h = continuum::make_kinematic_measure<StateVariableT>(constitutive_kin);

    return e_h; 
  };

  void set_material_point_state(const auto &model) noexcept{ // CONSTRAIN WITH MODEL CONCEPT
    // Extract element DOFs once — identical to commit_material_state path
    Eigen::VectorXd u_e = extract_element_dofs(model.state_vector());

    for (std::size_t gp = 0; gp < num_integration_points(); ++gp) {
        auto ref_pt = geometry_->reference_integration_point(gp);
        Array Xi{};
        for (std::size_t k = 0; k < dim; ++k) Xi[k] = ref_pt[k];

        auto kin = KinematicPolicy::template evaluate<dim>(
            geometry_, num_nodes(), ndof, Xi, u_e);
        auto constitutive_kin =
            continuum::make_constitutive_kinematics<KinematicPolicy>(kin);
        auto strain =
            continuum::make_kinematic_measure<StateVariableT>(constitutive_kin);
        material_points_[gp].update_state(std::move(strain));
    }
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
