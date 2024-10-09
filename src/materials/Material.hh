#ifndef FN_MATERIAL
#define FN_MATERIAL

#include <memory>
#include <random>
#include <vector>
#include <array>
#include <cstddef>
#include <memory>
#include <utility>


#include "../numerics/Tensor.hh"
#include "MaterialState.hh"
#include "MaterialPolicy.hh"

#include "update_strategy/lineal/ElasticUpdate.hh"

namespace impl //Implementation Details
{
   template<class MaterialPolicy>
   class MaterialConcept // The External Polymorphism design pattern
   {
      using StateVariableT = MaterialPolicy::StateVariable;
      using StressT = MaterialPolicy::StressT;

   public:
      virtual ~MaterialConcept() = default;
      virtual std::unique_ptr<MaterialConcept> clone() const = 0; // The Prototype design pattern
      virtual void clone(MaterialConcept *memory) const = 0;      // The Prototype design pattern
   
   public:

      //virtual StateVariableT const& get_state() const = 0;
      virtual StateVariableT get_state() const = 0;
      
      //virtual void update_state() const = 0;

   };

   template <typename MaterialType, typename UpdateStrategy>
   class NonOwningMaterialModel;//<std::invoke_result_t<decltype(&MaterialType::PolicyID)>>;

   template <typename MaterialType, typename UpdateStrategy>
   class OwningMaterialModel : public MaterialConcept<std::invoke_result_t<decltype(&MaterialType::PolicyID)>>
   {
   private:

      //using MaterialPolicy = std::invoke_result_t<decltype(&MaterialType::PolicyID)>;
      using MaterialPolicy = MaterialType::MaterialPolicy;

      MaterialType   material_        ;
      UpdateStrategy update_algorithm_; //or material_integrator

   public:
      explicit OwningMaterialModel(MaterialType material, UpdateStrategy mat_integrator)
          : material_{std::move(material)}, update_algorithm_{std::move(mat_integrator)}
      {}

      void update_state() const override { update_algorithm_(material_); }

      std::unique_ptr<MaterialConcept<MaterialPolicy>> clone() const override{ // The Prototype design pattern
         return std::make_unique<OwningMaterialModel>(*this);
      }

      void clone(MaterialConcept<MaterialPolicy> *memory) const{
         using Model = NonOwningMaterialModel<MaterialType const, UpdateStrategy const>;

         std::construct_at(static_cast<Model *>(memory), material_, update_algorithm_);
      }
   };

   template <typename MaterialType, typename UpdateStrategy>
   class NonOwningMaterialModel : public MaterialConcept<std::invoke_result_t<decltype(&MaterialType::PolicyID)>>{
   
   private:
      //using MaterialPolicy = std::invoke_result_t<decltype(&MaterialType::PolicyID)>;
      using MaterialPolicy = MaterialType::MaterialPolicy;

      MaterialType *material_{nullptr};
      UpdateStrategy *update_algorithm_{nullptr};
   
   public:
      NonOwningMaterialModel(MaterialType &material, UpdateStrategy &mat_integrator)
          : material_{std::addressof(material)}, update_algorithm_{std::addressof(mat_integrator)}
      {}

      void update_state() const override { (*update_algorithm_)(*material_); }

      std::unique_ptr<MaterialConcept<MaterialPolicy>> clone() const override{
         using Model = OwningMaterialModel<MaterialType, UpdateStrategy>;
         return std::make_unique<Model>(*material_, *update_algorithm_);
      }

      void clone(MaterialConcept<MaterialPolicy> *memory) const override{
         std::construct_at(static_cast<NonOwningMaterialModel *>(memory), *this);
      }


   };

} // namespace impl

template<class MaterialPolicy>
class Material; // Forward declaration

template<class MaterialPolicy>
class MaterialConstRef{
   friend class Material<MaterialPolicy>;

public:
   // Type 'MaterialType' and 'UpdateStrategy' are possibly cv qualified;
   // lvalue references prevent references to rvalues
   template <typename MaterialType, typename UpdateStrategy>
   MaterialConstRef(MaterialType &material, UpdateStrategy &mat_integrator){
      using Model =
          impl::NonOwningMaterialModel<MaterialType const, UpdateStrategy const>;
      static_assert(sizeof(Model) == MODEL_SIZE, "Invalid size detected");
      static_assert(alignof(Model) == alignof(void *), "Misaligned detected");

      std::construct_at(static_cast<Model *>(pimpl()), material, mat_integrator);
   }

   MaterialConstRef(Material<MaterialPolicy> &other);
   MaterialConstRef(Material<MaterialPolicy> const &other);

   MaterialConstRef(MaterialConstRef const &other){
      other.pimpl()->clone(pimpl());
   }

   MaterialConstRef &operator=(MaterialConstRef const &other){
      // Copy-and-swap idiom
      MaterialConstRef copy(other);
      raw_.swap(copy.raw_);
      return *this;
   }

   ~MaterialConstRef(){std::destroy_at(pimpl());} // or: pimpl()->~MaterialConcept();
   // Move operations explicitly not declared

private:
   friend void update_state(MaterialConstRef const &material){
      material.pimpl()->update_state();
   }

   impl::MaterialConcept<MaterialPolicy> *pimpl(){ // The Bridge design pattern
      return reinterpret_cast<impl::MaterialConcept<MaterialPolicy> *>(raw_.data());
   }

   impl::MaterialConcept<MaterialPolicy> const *pimpl() const{
      return reinterpret_cast<impl::MaterialConcept<MaterialPolicy> const *>(raw_.data());
   }

   // Expected size of a model instantiation:
   //     sizeof(MaterialType*) + sizeof(UpdateStrategy*) + sizeof(vptr)
   static constexpr std::size_t MODEL_SIZE = 3U * sizeof(void *);

   alignas(void *) std::array<std::byte, MODEL_SIZE> raw_;
};

template<class MaterialPolicy>
class Material
{
   friend class MaterialConstRef<MaterialPolicy>;

   std::unique_ptr<impl::MaterialConcept<MaterialPolicy>> pimpl_; // The Bridge design pattern

public:
   
   template <typename MaterialType, typename UpdateStrategy>
   Material(MaterialType material, UpdateStrategy mat_integrator){
      using Model = impl::OwningMaterialModel<MaterialType, UpdateStrategy>;
      pimpl_ = std::make_unique<Model>(std::move(material), std::move(mat_integrator));
   }

   Material(Material        <MaterialPolicy> const &other) : pimpl_(other.pimpl_->clone()){}
   Material(MaterialConstRef<MaterialPolicy> const &other) : pimpl_{other.pimpl()->clone()}{}

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

private:
   friend void update_state(Material const &material){
      material.pimpl_->update_state();
   }
};

template<class MaterialPolicy>
inline MaterialConstRef<MaterialPolicy>::MaterialConstRef(Material<MaterialPolicy> &other){
   other.pimpl_->clone(pimpl());
}

template<class MaterialPolicy>
inline MaterialConstRef<MaterialPolicy>::MaterialConstRef(Material<MaterialPolicy> const &other){
   other.pimpl_->clone(pimpl());
}





#endif