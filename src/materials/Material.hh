#ifndef FN_MATERIAL
#define FN_MATERIAL

#include <cstddef>
#include <memory>

#include <vector>
#include <array>

#include <functional>
#include <concepts>
#include <utility>


#include "../numerics/Tensor.hh"
#include "MaterialState.hh"
#include "MaterialPolicy.hh"
#include "InternalFieldSnapshot.hh"

#include "update_strategy/IntegrationStrategy.hh"


// =============================================================================
//  Material<Policy> — type-erased material with Strategy-mediated interface
// =============================================================================
//
//  The virtual interface exposes BOTH the legacy API (C, current_state,
//  update_state) and the Strategy-mediated API (compute_response, tangent,
//  commit).  The Strategy routes calls through the underlying concrete
//  material (MaterialInstance<Relation, StatePolicy>).
//
// =============================================================================

namespace impl{
   template<class MaterialPolicy>
   class MaterialConcept{ 

      using StateVariableT = typename MaterialPolicy::StateVariableT;
      using StressT        = typename MaterialPolicy::StressT;
      using MatrixT        = Eigen::Matrix<double, StateVariableT::num_components, StateVariableT::num_components>;

   public:
      virtual constexpr ~MaterialConcept() = default;
      virtual constexpr std::unique_ptr<MaterialConcept> clone() const = 0; // Prototype

   public:
      // Legacy interface
      virtual constexpr MatrixT               C()             const = 0;
      virtual constexpr const StateVariableT&  current_state() const = 0;

      virtual constexpr void update_state(const StateVariableT& state) = 0;
      virtual constexpr void update_state(StateVariableT&& state)      = 0;

      // Strategy-mediated interface
      virtual StressT compute_response(const StateVariableT& k) const = 0;
      virtual MatrixT tangent(const StateVariableT& k)          const = 0;
      virtual void    commit(const StateVariableT& k)                 = 0;

      // ── Internal state export (post-processing) ───────────────────────
      //  Returns a stack-allocated snapshot of internal variables.
      //  Default: empty (all nullopt) — elastic materials need no override.
      //  Inelastic materials fill the snapshot via if constexpr in OwningMaterialModel.
      virtual InternalFieldSnapshot internal_field_snapshot() const { return {}; }
   };

   template <typename MaterialType, typename UpdateStrategy>
   class OwningMaterialModel : public MaterialConcept<typename MaterialType::MaterialPolicy>{

      using MaterialPolicy = typename MaterialType::MaterialPolicy;
      using StateVariableT = typename MaterialPolicy::StateVariableT;
      using StressT        = typename MaterialPolicy::StressT;
      using MatrixT        = Eigen::Matrix<double, StateVariableT::num_components, StateVariableT::num_components>;

      MaterialType   material_        ;
      UpdateStrategy strategy_;

   public:
      // Legacy interface
      MatrixT C() const override {
         // Route through strategy with zero strain to get the elastic tangent.
         StateVariableT zero_strain{};
         return strategy_.tangent(material_, zero_strain);
      }

      const StateVariableT& current_state() const override {
         return material_.current_state();
      }

      void update_state(const StateVariableT& state) override {
         material_.update_state(state);
      }
      void update_state(StateVariableT&& state) override {
         material_.update_state(std::forward<StateVariableT>(state));
      }

      // Strategy-mediated interface
      StressT compute_response(const StateVariableT& k) const override {
         return strategy_.compute_response(material_, k);
      }

      MatrixT tangent(const StateVariableT& k) const override {
         return strategy_.tangent(material_, k);
      }

      void commit(const StateVariableT& k) override {
         strategy_.commit(material_, k);
      }

      // ── Internal state snapshot (post-processing export) ──────────────
      //  Uses if constexpr to detect whether the concrete material has
      //  internal_state() (satisfies InelasticConstitutiveRelation).
      //  This compiles away to a no-op for elastic materials.
      InternalFieldSnapshot internal_field_snapshot() const override {
         InternalFieldSnapshot snap;
         if constexpr (requires { material_.internal_state().eps_p(); }) {
             const auto& alpha = material_.internal_state();
             snap.plastic_strain = std::span<const double>(
                 alpha.eps_p().data(), alpha.eps_p().size());
         }
         if constexpr (requires { material_.internal_state().eps_bar_p(); }) {
             snap.equivalent_plastic_strain = material_.internal_state().eps_bar_p();
         }
         return snap;
      }

      explicit OwningMaterialModel(MaterialType material, UpdateStrategy strategy): 
          material_ {std::move(material)}, 
          strategy_ {std::move(strategy)}
      {}

      std::unique_ptr<MaterialConcept<MaterialPolicy>> clone() const override{
         return std::make_unique<OwningMaterialModel>(*this);
      }

   }; // OwningMaterialModel

} // namespace impl

template<class MaterialPolicy>
class Material{
   using StateVariableT = typename MaterialPolicy::StateVariableT;
   using StressT        = typename MaterialPolicy::StressT;
   using MatrixT        = Eigen::Matrix<double, StateVariableT::num_components, StateVariableT::num_components>;

   std::unique_ptr<impl::MaterialConcept<MaterialPolicy>> pimpl_;

public:

   // Legacy interface
   MatrixT C() const { return pimpl_->C(); }

   const StateVariableT& current_state() const { return pimpl_->current_state(); }
   
   void update_state(const StateVariableT& state) { pimpl_->update_state(state);                          }
   void update_state(StateVariableT&& state)      { pimpl_->update_state(std::forward<StateVariableT>(state)); }

   // Strategy-mediated interface
   StressT compute_response(const StateVariableT& k) const { return pimpl_->compute_response(k); }
   MatrixT tangent(const StateVariableT& k)          const { return pimpl_->tangent(k);          }
   void    commit(const StateVariableT& k)                 { pimpl_->commit(k);                  }

   // ── Internal state export (post-processing) ──────────────────────────
   [[nodiscard]] InternalFieldSnapshot internal_field_snapshot() const {
       return pimpl_->internal_field_snapshot();
   }

public:
   
   template <typename MaterialType, typename UpdateStrategy>
   Material(MaterialType material, UpdateStrategy strategy){
      using Model = impl::OwningMaterialModel<MaterialType, UpdateStrategy>;
      pimpl_ = std::make_unique<Model>(
         std::move(material), 
         std::move(strategy)
         );
   }

   Material(Material<MaterialPolicy> const &other) : pimpl_(other.pimpl_->clone()){}

   Material &operator=(Material const &other)
   {
      Material copy(other);
      pimpl_.swap(copy.pimpl_);
      return *this;
   }

   ~Material() = default;
   Material(Material &&) = default;
   Material &operator=(Material &&) = default;

};


#endif