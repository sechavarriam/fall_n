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

#include "update_strategy/lineal/ElasticUpdate.hh"


// Simple Type Erausre

namespace impl{
   template<class MaterialPolicy>
   class MaterialConcept{ 

      using StateVariableT = MaterialPolicy::StateVariableT;
      using StressT        = MaterialPolicy::StressT;
      using MatrixT        = Eigen::Matrix<double, StateVariableT::num_components, StateVariableT::num_components>;

   public:
      virtual constexpr ~MaterialConcept() = default;
      virtual constexpr std::unique_ptr<MaterialConcept> clone() const = 0; // The Prototype design pattern

   public:
      virtual constexpr MatrixT        C()         const = 0; //The Compliance DeprecatedDenseMatrix 
      virtual constexpr StateVariableT current_state() const = 0; //The current Value of the State Variable (or the head?)

      virtual constexpr void update_state(const StateVariableT& state) = 0;
      virtual constexpr void update_state(StateVariableT&& state) = 0;


      //virtual constexpr StressT compute_stress(const StateVariableT& state) const = 0; //The current Value of the State Variable (or the head?)

   };

   template <typename MaterialType, typename UpdateStrategy>
   class OwningMaterialModel : public MaterialConcept<typename MaterialType::MaterialPolicy>{

      using MaterialPolicy = typename MaterialType::MaterialPolicy;
      using StateVariableT = MaterialPolicy::StateVariableT;

      using MatrixT        = Eigen::Matrix<double, StateVariableT::num_components, StateVariableT::num_components>;

      MaterialType   material_        ;
      UpdateStrategy update_algorithm_; //or material_integrator

   public:
 
      MatrixT        C()             const override {return material_.C()        ;}; //The Compliance DeprecatedDenseMatrix
      StateVariableT current_state() const override {return material_.current_state();}; //CurrentValue

      void update_state(const StateVariableT& state) override {material_.update_state(state);};
      void update_state(StateVariableT&& state) override {material_.update_state(std::forward<StateVariableT>(state));};


      explicit OwningMaterialModel(MaterialType material, UpdateStrategy material_integrator): 
          material_        {std::move(material)}, 
          update_algorithm_{std::move(material_integrator)}
      {}

      std::unique_ptr<MaterialConcept<MaterialPolicy>> clone() const override{ // The Prototype design pattern
         return std::make_unique<OwningMaterialModel>(*this);
      }

   }; // OwningMaterialModel

} // namespace impl

template<class MaterialPolicy>
class Material{
   using StateVariableT = MaterialPolicy::StateVariableT;

   std::unique_ptr<impl::MaterialConcept<MaterialPolicy>> pimpl_; // The Bridge design pattern

public:

   Eigen::Matrix<double, StateVariableT::num_components, StateVariableT::num_components> C() const {return pimpl_->C();}; //The Compliance DeprecatedDenseMatrix

   StateVariableT current_state() const {return pimpl_->current_state();};
   
   void update_state(const StateVariableT& state) {pimpl_->update_state(state);};
   void update_state(StateVariableT&& state) {pimpl_->update_state(std::forward<StateVariableT>(state));};

public:
   
   template <typename MaterialType, typename UpdateStrategy>
   Material(MaterialType material, UpdateStrategy material_integrator){
      using Model = impl::OwningMaterialModel<MaterialType, UpdateStrategy>;
      pimpl_ = std::make_unique<Model>(
         std::move(material), 
         std::move(material_integrator)
         );
   }

   Material(Material<MaterialPolicy> const &other) : pimpl_(other.pimpl_->clone()){}

   Material &operator=(Material const &other)
   {
      // Copy-and-Swap Idiom
      Material copy(other);
      pimpl_.swap(copy.pimpl_);
      return *this;
   }

   ~Material() = default;
   Material(Material &&) = default;
   Material &operator=(Material &&) = default;

private: //Hidden Friends (Free Functions)


};


#endif