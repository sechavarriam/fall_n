#ifndef FN_MATERIAL
#define FN_MATERIAL

#include <array>
#include <concepts>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <typeinfo>
#include <utility>

// ============================================================================
//  Material<ConstitutiveSpace> — owning erased constitutive handle
// ============================================================================
//
//  Semantically, this wrapper is not a physical material.  It is an owning
//  runtime handle that binds:
//    1. a concrete stateful constitutive site (MaterialInstance / ConstitutiveSite),
//    2. a constitutive integrator,
//    3. a uniform erased interface for heterogeneous containers.
//
//  The borrowed companions MaterialConstRef / MaterialRef provide the
//  corresponding non-owning erased views.
//
// ============================================================================

#include "../numerics/Tensor.hh"
#include "../continuum/ConstitutiveKinematics.hh"
#include "ConstitutiveIntegrator.hh"
#include "InternalFieldSnapshot.hh"
#include "MaterialPolicy.hh"
#include "SectionConstitutiveSnapshot.hh"

template<class MaterialPolicy> class Material;
template<class MaterialPolicy> class MaterialConstRef;
template<class MaterialPolicy> class MaterialRef;

namespace impl {

// ─── StateRef: lightweight non-owning type-erased reference ──────────────
//
//  Replaces std::any in the state-injection API.  Zero allocation, zero copy:
//  just a const void* plus a type hash for safety.  The referred-to object
//  must outlive the StateRef.
//
//  Construct:  StateRef::from(some_state)
//  Extract:    ref.as<MenegottoPintoState>()    — throws on type mismatch.

struct StateRef {
   const void*           data_  = nullptr;
   const std::type_info* type_  = nullptr;

   constexpr StateRef() noexcept = default;
   constexpr StateRef(const void* d, const std::type_info* t) noexcept
      : data_{d}, type_{t} {}

   template <typename T>
   static StateRef from(const T& value) noexcept {
      return StateRef{&value, &typeid(T)};
   }

   template <typename T>
   [[nodiscard]] const T& as() const {
      if (!type_ || *type_ != typeid(T))
         throw std::runtime_error("StateRef::as(): type mismatch in state injection");
      return *static_cast<const T*>(data_);
   }

   [[nodiscard]] constexpr explicit operator bool() const noexcept { return data_ != nullptr; }
};

template<class MaterialPolicy>
class MaterialConcept;

template<class MaterialPolicy>
class MaterialConstConcept {
   using StateVariableT = typename MaterialPolicy::StateVariableT;
   using StressT = typename MaterialPolicy::StressT;
   using MatrixT = Eigen::Matrix<double, StateVariableT::num_components, StateVariableT::num_components>;

public:
   virtual ~MaterialConstConcept() = default;

   virtual std::unique_ptr<MaterialConcept<MaterialPolicy>> clone_owned() const = 0;
   virtual void clone_const_ref(MaterialConstConcept* memory) const = 0;

   virtual MatrixT C() const = 0;
   virtual const StateVariableT& current_state() const = 0;
   virtual StressT compute_response(const StateVariableT& k) const = 0;
   virtual StressT compute_response(
      const continuum::ConstitutiveKinematics<MaterialPolicy::dim>& kin) const = 0;
   virtual MatrixT tangent(const StateVariableT& k) const = 0;
    virtual MatrixT tangent(
      const continuum::ConstitutiveKinematics<MaterialPolicy::dim>& kin) const = 0;
   virtual InternalFieldSnapshot internal_field_snapshot() const { return {}; }
   virtual SectionConstitutiveSnapshot section_snapshot() const { return {}; }
};

// Helper: detect set_internal_state() capability at compile time.
template <typename M>
concept HasSetInternalState = requires(M& m, const typename M::InternalVariablesT& a) {
   m.set_internal_state(a);
};

// Helper: detect constitutive_relation() accessor wrapping an injectable relation.
template <typename M>
concept WrapsInjectableRelation = requires(M& m) {
   { m.constitutive_relation() };
} && HasSetInternalState<
   std::remove_cvref_t<decltype(std::declval<M&>().constitutive_relation())>>;

template<class MaterialPolicy>
class MaterialConcept : public MaterialConstConcept<MaterialPolicy> {
   using StateVariableT = typename MaterialPolicy::StateVariableT;

public:
   virtual void clone_ref(MaterialConcept* memory) const = 0;
   virtual void update_state(const StateVariableT& state) = 0;
   virtual void update_state(StateVariableT&& state) = 0;
   virtual void commit(const StateVariableT& k) = 0;
   virtual void commit(
      const continuum::ConstitutiveKinematics<MaterialPolicy::dim>& kin) = 0;
   virtual void revert() = 0;

   /// Inject a type-erased internal state (for FE² state transfer).
   /// Default: throws if the concrete model does not support injection.
   virtual void inject_internal_state(StateRef /*state*/) {
      throw std::runtime_error(
         "inject_internal_state: material does not support state injection");
   }

   /// Check at runtime whether the concrete model supports state injection.
   virtual bool supports_state_injection() const noexcept { return false; }
};

template <typename MaterialType>
InternalFieldSnapshot make_internal_field_snapshot(const MaterialType& material) {
   InternalFieldSnapshot snap;

   if constexpr (requires { material.internal_state().eps_p(); }) {
      const auto& alpha = material.internal_state();
      snap.plastic_strain = std::span<const double>(
         alpha.eps_p().data(), alpha.eps_p().size());
   }

   if constexpr (requires { material.internal_state().eps_bar_p(); }) {
      snap.equivalent_plastic_strain = material.internal_state().eps_bar_p();
   }

   // Continuum damage models may store the degradation variable either as a
   // direct public member (`damage`) or through a semantic accessor (`d()`).
   // The snapshot stays layout-agnostic and only records the scalar when the
   // constitutive state exposes one of those compile-time signatures.
   if constexpr (requires {
      { material.internal_state().damage } -> std::convertible_to<double>;
   }) {
      snap.damage = material.internal_state().damage;
   } else if constexpr (requires {
      { material.internal_state().d() } -> std::convertible_to<double>;
   }) {
      snap.damage = material.internal_state().d();
   }

   if constexpr (requires {
      material.internal_state().eps_min;
      material.internal_state().sig_at_eps_min;
      material.internal_state().eps_pl;
      material.internal_state().eps_t_max;
      material.internal_state().sig_at_eps_t_max;
      material.internal_state().state;
      material.internal_state().eps_committed;
      material.internal_state().sig_committed;
      material.internal_state().cracked;
   }) {
      const auto& alpha = material.internal_state();
      snap.history_state_code = alpha.state;
      snap.history_min_strain = alpha.eps_min;
      snap.history_min_stress = alpha.sig_at_eps_min;
      snap.history_closure_strain = alpha.eps_pl;
      snap.history_max_tensile_strain = alpha.eps_t_max;
      snap.history_max_tensile_stress = alpha.sig_at_eps_t_max;
      snap.history_committed_strain = alpha.eps_committed;
      snap.history_committed_stress = alpha.sig_committed;
      snap.history_cracked = alpha.cracked;
   }

   // ── Smeared cracking state (Ko-Bathe concrete and similar models) ─────
   //  Detects `.num_cracks` and `.crack_normals` on the internal state.
   //  Crack normals are promoted to 3D vectors for ParaView Glyph export.
   if constexpr (requires {
      material.internal_state().num_cracks;
      material.internal_state().crack_normals;
   }) {
      const auto& alpha = material.internal_state();
      snap.num_cracks = alpha.num_cracks;

      // Detect whether crack normals are 3D (e.g. KoBatheConcrete3D)
      // or 2D (KoBatheConcrete).  Promote 2D normals to 3D with z=0.
      using NormalT = std::remove_cvref_t<decltype(alpha.crack_normals[0])>;

      if (alpha.num_cracks >= 1) {
         if constexpr (NormalT::SizeAtCompileTime >= 3) {
            snap.crack_normal_1 = {alpha.crack_normals[0][0],
                                   alpha.crack_normals[0][1],
                                   alpha.crack_normals[0][2]};
         } else {
            snap.crack_normal_1 = {alpha.crack_normals[0][0],
                                   alpha.crack_normals[0][1], 0.0};
         }
         snap.crack_strain_1 = alpha.crack_strain[0];
         snap.crack_closed_1 = alpha.crack_closed[0] ? 1.0 : 0.0;
      }
      if (alpha.num_cracks >= 2) {
         if constexpr (NormalT::SizeAtCompileTime >= 3) {
            snap.crack_normal_2 = {alpha.crack_normals[1][0],
                                   alpha.crack_normals[1][1],
                                   alpha.crack_normals[1][2]};
         } else {
            snap.crack_normal_2 = {alpha.crack_normals[1][0],
                                   alpha.crack_normals[1][1], 0.0};
         }
         snap.crack_strain_2 = alpha.crack_strain[1];
         snap.crack_closed_2 = alpha.crack_closed[1] ? 1.0 : 0.0;
      }
      if (alpha.num_cracks >= 3) {
         if constexpr (NormalT::SizeAtCompileTime >= 3) {
            snap.crack_normal_3 = {alpha.crack_normals[2][0],
                                   alpha.crack_normals[2][1],
                                   alpha.crack_normals[2][2]};
         } else {
            snap.crack_normal_3 = {alpha.crack_normals[2][0],
                                   alpha.crack_normals[2][1], 0.0};
         }
         snap.crack_strain_3 = alpha.crack_strain[2];
         snap.crack_closed_3 = alpha.crack_closed[2] ? 1.0 : 0.0;
      }
   }

   // ── Fracturing history invariants (concrete models) ───────────────────
   if constexpr (requires {
      material.internal_state().sigma_o_max;
      material.internal_state().tau_o_max;
   }) {
      const auto& alpha = material.internal_state();
      snap.sigma_o_max = alpha.sigma_o_max;
      snap.tau_o_max   = alpha.tau_o_max;
   }

   if constexpr (requires {
      material.internal_state().last_solution_mode;
      material.internal_state().last_trial_sigma_o;
      material.internal_state().last_trial_tau_o;
      material.internal_state().last_no_flow_coupling_update_norm;
      material.internal_state().last_no_flow_recovery_residual;
      material.internal_state().last_no_flow_stabilization_iterations;
      material.internal_state().last_no_flow_crack_state_switches;
      material.internal_state().last_no_flow_stabilized;
   }) {
      const auto& alpha = material.internal_state();
      snap.solution_mode =
         static_cast<int>(alpha.last_solution_mode);
      snap.trial_sigma_o = alpha.last_trial_sigma_o;
      snap.trial_tau_o = alpha.last_trial_tau_o;
      snap.no_flow_coupling_update_norm =
         alpha.last_no_flow_coupling_update_norm;
      snap.no_flow_recovery_residual =
         alpha.last_no_flow_recovery_residual;
      snap.no_flow_stabilization_iterations =
         alpha.last_no_flow_stabilization_iterations;
      snap.no_flow_crack_state_switches =
         alpha.last_no_flow_crack_state_switches;
      snap.no_flow_stabilized = alpha.last_no_flow_stabilized;
   }

   if constexpr (requires { material.last_evaluation_diagnostics(); }) {
      const auto& diag = material.last_evaluation_diagnostics();
      snap.solution_mode = static_cast<int>(diag.solution_mode);
      snap.trial_sigma_o = diag.trial_sigma_o;
      snap.trial_tau_o = diag.trial_tau_o;
      snap.no_flow_coupling_update_norm = diag.no_flow_coupling_update_norm;
      snap.no_flow_recovery_residual = diag.no_flow_recovery_residual;
      snap.no_flow_stabilization_iterations =
         diag.no_flow_stabilization_iterations;
      snap.no_flow_crack_state_switches =
         diag.no_flow_crack_state_switches;
      snap.no_flow_stabilized = diag.no_flow_stabilized;
   }

   return snap;
}

template <typename MaterialType>
SectionConstitutiveSnapshot make_section_snapshot(const MaterialType& material) {
   SectionConstitutiveSnapshot snap;

   const auto& relation = material.constitutive_relation();

   if constexpr (requires {
      relation.young_modulus();
      relation.shear_modulus();
      relation.area();
   }) {
      BeamSectionConstitutiveSnapshot beam;
      beam.young_modulus = relation.young_modulus();
      beam.shear_modulus = relation.shear_modulus();
      beam.area = relation.area();

      if constexpr (requires { relation.moment_of_inertia_y(); }) {
         beam.moment_y = relation.moment_of_inertia_y();
      } else if constexpr (requires { relation.moment_of_inertia(); }) {
         beam.moment_y = relation.moment_of_inertia();
      }

      if constexpr (requires { relation.moment_of_inertia_z(); }) {
         beam.moment_z = relation.moment_of_inertia_z();
      }

      if constexpr (requires { relation.torsional_constant(); }) {
         beam.torsion_J = relation.torsional_constant();
      }

      if constexpr (requires { relation.shear_correction_y(); }) {
         beam.shear_factor_y = relation.shear_correction_y();
      } else if constexpr (requires { relation.shear_correction(); }) {
         beam.shear_factor_y = relation.shear_correction();
      }

      if constexpr (requires { relation.shear_correction_z(); }) {
         beam.shear_factor_z = relation.shear_correction_z();
      } else if constexpr (requires { relation.shear_correction(); }) {
         beam.shear_factor_z = relation.shear_correction();
      }

      snap.beam = beam;
   }

   if constexpr (requires {
      relation.young_modulus();
      relation.poisson_ratio();
      relation.thickness();
   }) {
      ShellSectionConstitutiveSnapshot shell;
      shell.young_modulus = relation.young_modulus();
      shell.poisson_ratio = relation.poisson_ratio();
      if constexpr (requires { relation.shear_modulus(); }) {
         shell.shear_modulus = relation.shear_modulus();
      }
      shell.thickness = relation.thickness();
      if constexpr (requires { relation.shear_correction(); }) {
         shell.shear_correction = relation.shear_correction();
      }
      snap.shell = shell;
   }

   if constexpr (requires { relation.fiber_field_snapshot(material.current_state()); }) {
      snap.fibers = relation.fiber_field_snapshot(material.current_state());
   }

   return snap;
}

template <typename MaterialType, typename ConstitutiveIntegratorT>
class NonOwningMaterialConstModel;

template <typename MaterialType, typename ConstitutiveIntegratorT>
class NonOwningMaterialModel;

// ─────────────────────────────────────────────────────────────────────────────
//  Kinematic dispatch helpers
// ─────────────────────────────────────────────────────────────────────────────
//  Consolidate the recurring if-constexpr pattern used by the three
//  type-erased model classes.  Prefer the full ConstitutiveKinematics
//  overload when the integrator provides one; otherwise collapse to the
//  reduced kinematic measure so that structural (non-continuum) integrators
//  remain backward-compatible.
// ─────────────────────────────────────────────────────────────────────────────

template <typename StateVariableT, typename IntegratorT, typename MaterialT, std::size_t Dim>
auto dispatch_response(
    const IntegratorT& integrator,
    const MaterialT& material,
    const continuum::ConstitutiveKinematics<Dim>& kin)
{
    if constexpr (requires { integrator.compute_response(material, kin); }) {
        return integrator.compute_response(material, kin);
    } else {
        return integrator.compute_response(
            material,
            continuum::make_kinematic_measure<StateVariableT>(kin));
    }
}

template <typename StateVariableT, typename IntegratorT, typename MaterialT, std::size_t Dim>
auto dispatch_tangent(
    const IntegratorT& integrator,
    const MaterialT& material,
    const continuum::ConstitutiveKinematics<Dim>& kin)
{
    if constexpr (requires { integrator.tangent(material, kin); }) {
        return integrator.tangent(material, kin);
    } else {
        return integrator.tangent(
            material,
            continuum::make_kinematic_measure<StateVariableT>(kin));
    }
}

template <typename StateVariableT, typename IntegratorT, typename MaterialT, std::size_t Dim>
void dispatch_commit(
    const IntegratorT& integrator,
    MaterialT& material,
    const continuum::ConstitutiveKinematics<Dim>& kin)
{
    if constexpr (requires { integrator.commit(material, kin); }) {
        integrator.commit(material, kin);
    } else {
        integrator.commit(
            material,
            continuum::make_kinematic_measure<StateVariableT>(kin));
    }
}

template <typename MaterialType, typename ConstitutiveIntegratorT>
class OwningMaterialModel : public MaterialConcept<typename MaterialType::MaterialPolicy> {
   using MaterialPolicyT = typename MaterialType::MaterialPolicy;
   using StateVariableT = typename MaterialPolicyT::StateVariableT;
   using StressT = typename MaterialPolicyT::StressT;
   using MatrixT = Eigen::Matrix<double, StateVariableT::num_components, StateVariableT::num_components>;

   MaterialType material_;
   ConstitutiveIntegratorT integrator_;

public:
   explicit OwningMaterialModel(MaterialType material, ConstitutiveIntegratorT integrator)
      : material_{std::move(material)}
      , integrator_{std::move(integrator)}
   {}

   MatrixT C() const override {
      StateVariableT zero_state{};
      return integrator_.tangent(material_, zero_state);
   }

   const StateVariableT& current_state() const override {
      return material_.current_state();
   }

   void update_state(const StateVariableT& state) override {
      material_.update_state(state);
   }

   void update_state(StateVariableT&& state) override {
      material_.update_state(std::move(state));
   }

   StressT compute_response(const StateVariableT& k) const override {
      return integrator_.compute_response(material_, k);
   }

   StressT compute_response(
      const continuum::ConstitutiveKinematics<MaterialPolicyT::dim>& kin) const override {
      return dispatch_response<StateVariableT>(integrator_, material_, kin);
   }

   MatrixT tangent(const StateVariableT& k) const override {
      return integrator_.tangent(material_, k);
   }

   MatrixT tangent(
      const continuum::ConstitutiveKinematics<MaterialPolicyT::dim>& kin) const override {
      return dispatch_tangent<StateVariableT>(integrator_, material_, kin);
   }

   void commit(const StateVariableT& k) override {
      integrator_.commit(material_, k);
   }

   void commit(
      const continuum::ConstitutiveKinematics<MaterialPolicyT::dim>& kin) override {
      dispatch_commit<StateVariableT>(integrator_, material_, kin);
   }

   void revert() override {
      integrator_.revert(material_);
   }

   void inject_internal_state(StateRef state) override {
      if constexpr (HasSetInternalState<MaterialType>) {
         material_.set_internal_state(
            state.as<typename MaterialType::InternalVariablesT>());
      } else if constexpr (WrapsInjectableRelation<MaterialType>) {
         using RelT = std::remove_cvref_t<decltype(material_.constitutive_relation())>;
         const auto& s = state.as<typename RelT::InternalVariablesT>();
         material_.constitutive_relation().set_internal_state(s);
         // Also sync the ConstitutiveSite's algorithmic state
         // (used by Level 3 compute_response dispatch).
         if constexpr (requires { material_.algorithmic_state() = s; }) {
            material_.algorithmic_state() = s;
         }
      } else {
         throw std::runtime_error(
            "inject_internal_state: concrete material lacks set_internal_state()");
      }
   }

   bool supports_state_injection() const noexcept override {
      return HasSetInternalState<MaterialType> || WrapsInjectableRelation<MaterialType>;
   }

   InternalFieldSnapshot internal_field_snapshot() const override {
      return make_internal_field_snapshot(material_);
   }

   SectionConstitutiveSnapshot section_snapshot() const override {
      return make_section_snapshot(material_);
   }

   std::unique_ptr<MaterialConcept<MaterialPolicyT>> clone_owned() const override {
      return std::make_unique<OwningMaterialModel>(*this);
   }

   void clone_const_ref(MaterialConstConcept<MaterialPolicyT>* memory) const override {
      using Model = NonOwningMaterialConstModel<const MaterialType, const ConstitutiveIntegratorT>;
      std::construct_at(static_cast<Model*>(memory), material_, integrator_);
   }

   void clone_ref(MaterialConcept<MaterialPolicyT>* memory) const override {
      using Model = NonOwningMaterialModel<MaterialType, ConstitutiveIntegratorT>;
      std::construct_at(
         static_cast<Model*>(memory),
         const_cast<MaterialType&>(material_),
         const_cast<ConstitutiveIntegratorT&>(integrator_));
   }
};

template <typename MaterialType, typename ConstitutiveIntegratorT>
class NonOwningMaterialConstModel
   : public MaterialConstConcept<typename std::remove_cvref_t<MaterialType>::MaterialPolicy> {
   using RawMaterialT = std::remove_cvref_t<MaterialType>;
   using MaterialPolicyT = typename RawMaterialT::MaterialPolicy;
   using StateVariableT = typename MaterialPolicyT::StateVariableT;
   using StressT = typename MaterialPolicyT::StressT;
   using MatrixT = Eigen::Matrix<double, StateVariableT::num_components, StateVariableT::num_components>;
   using OwningModel = OwningMaterialModel<std::remove_cv_t<MaterialType>, std::remove_cv_t<ConstitutiveIntegratorT>>;

   MaterialType* material_{nullptr};
   ConstitutiveIntegratorT* integrator_{nullptr};

public:
   NonOwningMaterialConstModel(MaterialType& material, ConstitutiveIntegratorT& integrator)
      : material_{std::addressof(material)}
      , integrator_{std::addressof(integrator)}
   {}

   MatrixT C() const override {
      StateVariableT zero_state{};
      return integrator_->tangent(*material_, zero_state);
   }

   const StateVariableT& current_state() const override {
      return material_->current_state();
   }

   StressT compute_response(const StateVariableT& k) const override {
      return integrator_->compute_response(*material_, k);
   }

   StressT compute_response(
      const continuum::ConstitutiveKinematics<MaterialPolicyT::dim>& kin) const override {
      return dispatch_response<StateVariableT>(*integrator_, *material_, kin);
   }

   MatrixT tangent(const StateVariableT& k) const override {
      return integrator_->tangent(*material_, k);
   }

   MatrixT tangent(
      const continuum::ConstitutiveKinematics<MaterialPolicyT::dim>& kin) const override {
      return dispatch_tangent<StateVariableT>(*integrator_, *material_, kin);
   }

   InternalFieldSnapshot internal_field_snapshot() const override {
      return make_internal_field_snapshot(*material_);
   }

   SectionConstitutiveSnapshot section_snapshot() const override {
      return make_section_snapshot(*material_);
   }

   std::unique_ptr<MaterialConcept<MaterialPolicyT>> clone_owned() const override {
      return std::make_unique<OwningModel>(*material_, *integrator_);
   }

   void clone_const_ref(MaterialConstConcept<MaterialPolicyT>* memory) const override {
      std::construct_at(static_cast<NonOwningMaterialConstModel*>(memory), *this);
   }
};

template <typename MaterialType, typename ConstitutiveIntegratorT>
class NonOwningMaterialModel
   : public MaterialConcept<typename std::remove_cvref_t<MaterialType>::MaterialPolicy> {
   using RawMaterialT = std::remove_cvref_t<MaterialType>;
   using MaterialPolicyT = typename RawMaterialT::MaterialPolicy;
   using StateVariableT = typename MaterialPolicyT::StateVariableT;
   using StressT = typename MaterialPolicyT::StressT;
   using MatrixT = Eigen::Matrix<double, StateVariableT::num_components, StateVariableT::num_components>;
   using OwningModel = OwningMaterialModel<std::remove_cv_t<MaterialType>, std::remove_cv_t<ConstitutiveIntegratorT>>;

   MaterialType* material_{nullptr};
   ConstitutiveIntegratorT* integrator_{nullptr};

public:
   NonOwningMaterialModel(MaterialType& material, ConstitutiveIntegratorT& integrator)
      : material_{std::addressof(material)}
      , integrator_{std::addressof(integrator)}
   {}

   MatrixT C() const override {
      StateVariableT zero_state{};
      return integrator_->tangent(*material_, zero_state);
   }

   const StateVariableT& current_state() const override {
      return material_->current_state();
   }

   void update_state(const StateVariableT& state) override {
      material_->update_state(state);
   }

   void update_state(StateVariableT&& state) override {
      material_->update_state(std::move(state));
   }

   StressT compute_response(const StateVariableT& k) const override {
      return integrator_->compute_response(*material_, k);
   }

   StressT compute_response(
      const continuum::ConstitutiveKinematics<MaterialPolicyT::dim>& kin) const override {
      return dispatch_response<StateVariableT>(*integrator_, *material_, kin);
   }

   MatrixT tangent(const StateVariableT& k) const override {
      return integrator_->tangent(*material_, k);
   }

   MatrixT tangent(
      const continuum::ConstitutiveKinematics<MaterialPolicyT::dim>& kin) const override {
      return dispatch_tangent<StateVariableT>(*integrator_, *material_, kin);
   }

   void commit(const StateVariableT& k) override {
      integrator_->commit(*material_, k);
   }

   void commit(
      const continuum::ConstitutiveKinematics<MaterialPolicyT::dim>& kin) override {
      dispatch_commit<StateVariableT>(*integrator_, *material_, kin);
   }

   void revert() override {
      integrator_->revert(*material_);
   }

   void inject_internal_state(StateRef state) override {
      using MT = std::remove_cvref_t<MaterialType>;
      if constexpr (HasSetInternalState<MT>) {
         material_->set_internal_state(
            state.as<typename MT::InternalVariablesT>());
      } else if constexpr (WrapsInjectableRelation<MT>) {
         using RelT = std::remove_cvref_t<decltype(material_->constitutive_relation())>;
         const auto& s = state.as<typename RelT::InternalVariablesT>();
         material_->constitutive_relation().set_internal_state(s);
         if constexpr (requires { material_->algorithmic_state() = s; }) {
            material_->algorithmic_state() = s;
         }
      } else {
         throw std::runtime_error(
            "inject_internal_state: concrete material lacks set_internal_state()");
      }
   }

   bool supports_state_injection() const noexcept override {
      using MT = std::remove_cvref_t<MaterialType>;
      return HasSetInternalState<MT> || WrapsInjectableRelation<MT>;
   }

   InternalFieldSnapshot internal_field_snapshot() const override {
      return make_internal_field_snapshot(*material_);
   }

   SectionConstitutiveSnapshot section_snapshot() const override {
      return make_section_snapshot(*material_);
   }

   std::unique_ptr<MaterialConcept<MaterialPolicyT>> clone_owned() const override {
      return std::make_unique<OwningModel>(*material_, *integrator_);
   }

   void clone_const_ref(MaterialConstConcept<MaterialPolicyT>* memory) const override {
      using ConstModel = NonOwningMaterialConstModel<const MaterialType, const ConstitutiveIntegratorT>;
      std::construct_at(static_cast<ConstModel*>(memory), *material_, *integrator_);
   }

   void clone_ref(MaterialConcept<MaterialPolicyT>* memory) const override {
      std::construct_at(static_cast<NonOwningMaterialModel*>(memory), *this);
   }
};

} // namespace impl

template<class MaterialPolicy>
class MaterialConstRef {
   using StateVariableT = typename MaterialPolicy::StateVariableT;
   using StressT = typename MaterialPolicy::StressT;
   using MatrixT = Eigen::Matrix<double, StateVariableT::num_components, StateVariableT::num_components>;

   static constexpr std::size_t MODEL_SIZE = 3U * sizeof(void*);
   alignas(void*) std::array<std::byte, MODEL_SIZE> raw_{};

   impl::MaterialConstConcept<MaterialPolicy>* pimpl() {
      return reinterpret_cast<impl::MaterialConstConcept<MaterialPolicy>*>(raw_.data());
   }

   const impl::MaterialConstConcept<MaterialPolicy>* pimpl() const {
      return reinterpret_cast<const impl::MaterialConstConcept<MaterialPolicy>*>(raw_.data());
   }

public:
   using ConstitutiveSpace = MaterialPolicy;

   template <typename MaterialType, typename ConstitutiveIntegratorT>
   MaterialConstRef(MaterialType& material, ConstitutiveIntegratorT& integrator) {
      using Model = impl::NonOwningMaterialConstModel<const MaterialType, const ConstitutiveIntegratorT>;
      static_assert(sizeof(Model) == MODEL_SIZE, "Invalid non-owning material const-ref size");
      static_assert(alignof(Model) == alignof(void*), "Invalid non-owning material const-ref alignment");
      std::construct_at(static_cast<Model*>(pimpl()), material, integrator);
   }

   MaterialConstRef(Material<MaterialPolicy>& other);
   MaterialConstRef(const Material<MaterialPolicy>& other);
   MaterialConstRef(MaterialRef<MaterialPolicy>& other);

   MaterialConstRef(const MaterialConstRef& other) {
      other.pimpl()->clone_const_ref(pimpl());
   }

   MaterialConstRef& operator=(const MaterialConstRef& other) {
      MaterialConstRef copy(other);
      raw_.swap(copy.raw_);
      return *this;
   }

   ~MaterialConstRef() {
      std::destroy_at(pimpl());
   }

   [[nodiscard]] MatrixT C() const { return pimpl()->C(); }
   [[nodiscard]] const StateVariableT& current_state() const { return pimpl()->current_state(); }
   [[nodiscard]] StressT compute_response(const StateVariableT& k) const { return pimpl()->compute_response(k); }
   [[nodiscard]] StressT compute_response(
      const continuum::ConstitutiveKinematics<MaterialPolicy::dim>& kin) const {
      return pimpl()->compute_response(kin);
   }
   [[nodiscard]] MatrixT tangent(const StateVariableT& k) const { return pimpl()->tangent(k); }
   [[nodiscard]] MatrixT tangent(
      const continuum::ConstitutiveKinematics<MaterialPolicy::dim>& kin) const {
      return pimpl()->tangent(kin);
   }
   [[nodiscard]] InternalFieldSnapshot internal_field_snapshot() const { return pimpl()->internal_field_snapshot(); }
   [[nodiscard]] SectionConstitutiveSnapshot section_snapshot() const { return pimpl()->section_snapshot(); }

   friend class Material<MaterialPolicy>;
   friend class MaterialRef<MaterialPolicy>;
};

template<class MaterialPolicy>
class MaterialRef {
   using StateVariableT = typename MaterialPolicy::StateVariableT;
   using StressT = typename MaterialPolicy::StressT;
   using MatrixT = Eigen::Matrix<double, StateVariableT::num_components, StateVariableT::num_components>;

   static constexpr std::size_t MODEL_SIZE = 3U * sizeof(void*);
   alignas(void*) std::array<std::byte, MODEL_SIZE> raw_{};

   impl::MaterialConcept<MaterialPolicy>* pimpl() {
      return reinterpret_cast<impl::MaterialConcept<MaterialPolicy>*>(raw_.data());
   }

   const impl::MaterialConcept<MaterialPolicy>* pimpl() const {
      return reinterpret_cast<const impl::MaterialConcept<MaterialPolicy>*>(raw_.data());
   }

public:
   using ConstitutiveSpace = MaterialPolicy;

   template <typename MaterialType, typename ConstitutiveIntegratorT>
   MaterialRef(MaterialType& material, ConstitutiveIntegratorT& integrator) {
      using Model = impl::NonOwningMaterialModel<MaterialType, ConstitutiveIntegratorT>;
      static_assert(sizeof(Model) == MODEL_SIZE, "Invalid non-owning material ref size");
      static_assert(alignof(Model) == alignof(void*), "Invalid non-owning material ref alignment");
      std::construct_at(static_cast<Model*>(pimpl()), material, integrator);
   }

   MaterialRef(Material<MaterialPolicy>& other);

   MaterialRef(const MaterialRef& other) {
      other.pimpl()->clone_ref(pimpl());
   }

   MaterialRef& operator=(const MaterialRef& other) {
      MaterialRef copy(other);
      raw_.swap(copy.raw_);
      return *this;
   }

   ~MaterialRef() {
      std::destroy_at(pimpl());
   }

   [[nodiscard]] MatrixT C() const { return pimpl()->C(); }
   [[nodiscard]] const StateVariableT& current_state() const { return pimpl()->current_state(); }
   void update_state(const StateVariableT& state) { pimpl()->update_state(state); }
   void update_state(StateVariableT&& state) { pimpl()->update_state(std::move(state)); }
   [[nodiscard]] StressT compute_response(const StateVariableT& k) const { return pimpl()->compute_response(k); }
   [[nodiscard]] StressT compute_response(
      const continuum::ConstitutiveKinematics<MaterialPolicy::dim>& kin) const {
      return pimpl()->compute_response(kin);
   }
   [[nodiscard]] MatrixT tangent(const StateVariableT& k) const { return pimpl()->tangent(k); }
   [[nodiscard]] MatrixT tangent(
      const continuum::ConstitutiveKinematics<MaterialPolicy::dim>& kin) const {
      return pimpl()->tangent(kin);
   }
   void commit(const StateVariableT& k) { pimpl()->commit(k); }
   void commit(const continuum::ConstitutiveKinematics<MaterialPolicy::dim>& kin) {
      pimpl()->commit(kin);
   }
   void revert() { pimpl()->revert(); }
   void inject_internal_state(impl::StateRef state) { pimpl()->inject_internal_state(state); }
   [[nodiscard]] bool supports_state_injection() const noexcept { return pimpl()->supports_state_injection(); }
   [[nodiscard]] InternalFieldSnapshot internal_field_snapshot() const { return pimpl()->internal_field_snapshot(); }
   [[nodiscard]] SectionConstitutiveSnapshot section_snapshot() const { return pimpl()->section_snapshot(); }
   [[nodiscard]] MaterialConstRef<MaterialPolicy> cref() const { return MaterialConstRef<MaterialPolicy>(*this); }

   friend class Material<MaterialPolicy>;
   friend class MaterialConstRef<MaterialPolicy>;
};

template<class MaterialPolicy>
class Material {
   using StateVariableT = typename MaterialPolicy::StateVariableT;
   using StressT = typename MaterialPolicy::StressT;
   using MatrixT = Eigen::Matrix<double, StateVariableT::num_components, StateVariableT::num_components>;

   std::unique_ptr<impl::MaterialConcept<MaterialPolicy>> pimpl_;

public:
   using ConstitutiveSpace = MaterialPolicy;

   template <typename MaterialType, typename ConstitutiveIntegratorT>
   Material(MaterialType material, ConstitutiveIntegratorT integrator) {
      using Model = impl::OwningMaterialModel<MaterialType, ConstitutiveIntegratorT>;
      pimpl_ = std::make_unique<Model>(std::move(material), std::move(integrator));
   }

   Material(const Material& other)
      : pimpl_(other.pimpl_->clone_owned())
   {}

   Material(const MaterialConstRef<MaterialPolicy>& other)
      : pimpl_(other.pimpl()->clone_owned())
   {}

   Material(const MaterialRef<MaterialPolicy>& other)
      : pimpl_(other.pimpl()->clone_owned())
   {}

   Material& operator=(const Material& other) {
      Material copy(other);
      pimpl_.swap(copy.pimpl_);
      return *this;
   }

   [[nodiscard]] MatrixT C() const { return pimpl_->C(); }
   [[nodiscard]] const StateVariableT& current_state() const { return pimpl_->current_state(); }
   void update_state(const StateVariableT& state) { pimpl_->update_state(state); }
   void update_state(StateVariableT&& state) { pimpl_->update_state(std::move(state)); }
   [[nodiscard]] StressT compute_response(const StateVariableT& k) const { return pimpl_->compute_response(k); }
   [[nodiscard]] StressT compute_response(
      const continuum::ConstitutiveKinematics<MaterialPolicy::dim>& kin) const {
      return pimpl_->compute_response(kin);
   }
   [[nodiscard]] MatrixT tangent(const StateVariableT& k) const { return pimpl_->tangent(k); }
   [[nodiscard]] MatrixT tangent(
      const continuum::ConstitutiveKinematics<MaterialPolicy::dim>& kin) const {
      return pimpl_->tangent(kin);
   }
   void commit(const StateVariableT& k) { pimpl_->commit(k); }
   void commit(const continuum::ConstitutiveKinematics<MaterialPolicy::dim>& kin) {
      pimpl_->commit(kin);
   }
   void revert() { pimpl_->revert(); }
   void inject_internal_state(impl::StateRef state) { pimpl_->inject_internal_state(state); }
   [[nodiscard]] bool supports_state_injection() const noexcept { return pimpl_->supports_state_injection(); }
   [[nodiscard]] InternalFieldSnapshot internal_field_snapshot() const { return pimpl_->internal_field_snapshot(); }
   [[nodiscard]] SectionConstitutiveSnapshot section_snapshot() const { return pimpl_->section_snapshot(); }
   [[nodiscard]] MaterialConstRef<MaterialPolicy> cref() const { return MaterialConstRef<MaterialPolicy>(*this); }
   [[nodiscard]] MaterialRef<MaterialPolicy> ref() { return MaterialRef<MaterialPolicy>(*this); }

   ~Material() = default;
   Material(Material&&) = default;
   Material& operator=(Material&&) = default;

   friend class MaterialConstRef<MaterialPolicy>;
   friend class MaterialRef<MaterialPolicy>;
};

template<class MaterialPolicy>
MaterialConstRef<MaterialPolicy>::MaterialConstRef(Material<MaterialPolicy>& other) {
   other.pimpl_->clone_const_ref(pimpl());
}

template<class MaterialPolicy>
MaterialConstRef<MaterialPolicy>::MaterialConstRef(const Material<MaterialPolicy>& other) {
   other.pimpl_->clone_const_ref(pimpl());
}

template<class MaterialPolicy>
MaterialConstRef<MaterialPolicy>::MaterialConstRef(MaterialRef<MaterialPolicy>& other) {
   other.pimpl()->clone_const_ref(pimpl());
}

template<class MaterialPolicy>
MaterialRef<MaterialPolicy>::MaterialRef(Material<MaterialPolicy>& other) {
   other.pimpl_->clone_ref(pimpl());
}

template<class ConstitutiveSpace>
using ConstitutiveHandle = Material<ConstitutiveSpace>;

template<class ConstitutiveSpace>
using ConstitutiveConstHandleRef = MaterialConstRef<ConstitutiveSpace>;

template<class ConstitutiveSpace>
using ConstitutiveHandleRef = MaterialRef<ConstitutiveSpace>;

#endif
