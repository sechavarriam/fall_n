#ifndef FN_MATERIAL
#define FN_MATERIAL

#include <array>
#include <concepts>
#include <cstddef>
#include <memory>
#include <type_traits>
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
      if constexpr (requires {
         integrator_.compute_response(material_, kin);
      }) {
         return integrator_.compute_response(material_, kin);
      } else {
         return integrator_.compute_response(
            material_,
            continuum::make_kinematic_measure<StateVariableT>(kin));
      }
   }

   MatrixT tangent(const StateVariableT& k) const override {
      return integrator_.tangent(material_, k);
   }

   MatrixT tangent(
      const continuum::ConstitutiveKinematics<MaterialPolicyT::dim>& kin) const override {
      if constexpr (requires {
         integrator_.tangent(material_, kin);
      }) {
         return integrator_.tangent(material_, kin);
      } else {
         return integrator_.tangent(
            material_,
            continuum::make_kinematic_measure<StateVariableT>(kin));
      }
   }

   void commit(const StateVariableT& k) override {
      integrator_.commit(material_, k);
   }

   void commit(
      const continuum::ConstitutiveKinematics<MaterialPolicyT::dim>& kin) override {
      if constexpr (requires {
         integrator_.commit(material_, kin);
      }) {
         integrator_.commit(material_, kin);
      } else {
         integrator_.commit(
            material_,
            continuum::make_kinematic_measure<StateVariableT>(kin));
      }
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
      if constexpr (requires {
         integrator_->compute_response(*material_, kin);
      }) {
         return integrator_->compute_response(*material_, kin);
      } else {
         return integrator_->compute_response(
            *material_,
            continuum::make_kinematic_measure<StateVariableT>(kin));
      }
   }

   MatrixT tangent(const StateVariableT& k) const override {
      return integrator_->tangent(*material_, k);
   }

   MatrixT tangent(
      const continuum::ConstitutiveKinematics<MaterialPolicyT::dim>& kin) const override {
      if constexpr (requires {
         integrator_->tangent(*material_, kin);
      }) {
         return integrator_->tangent(*material_, kin);
      } else {
         return integrator_->tangent(
            *material_,
            continuum::make_kinematic_measure<StateVariableT>(kin));
      }
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
      if constexpr (requires {
         integrator_->compute_response(*material_, kin);
      }) {
         return integrator_->compute_response(*material_, kin);
      } else {
         return integrator_->compute_response(
            *material_,
            continuum::make_kinematic_measure<StateVariableT>(kin));
      }
   }

   MatrixT tangent(const StateVariableT& k) const override {
      return integrator_->tangent(*material_, k);
   }

   MatrixT tangent(
      const continuum::ConstitutiveKinematics<MaterialPolicyT::dim>& kin) const override {
      if constexpr (requires {
         integrator_->tangent(*material_, kin);
      }) {
         return integrator_->tangent(*material_, kin);
      } else {
         return integrator_->tangent(
            *material_,
            continuum::make_kinematic_measure<StateVariableT>(kin));
      }
   }

   void commit(const StateVariableT& k) override {
      integrator_->commit(*material_, k);
   }

   void commit(
      const continuum::ConstitutiveKinematics<MaterialPolicyT::dim>& kin) override {
      if constexpr (requires {
         integrator_->commit(*material_, kin);
      }) {
         integrator_->commit(*material_, kin);
      } else {
         integrator_->commit(
            *material_,
            continuum::make_kinematic_measure<StateVariableT>(kin));
      }
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
