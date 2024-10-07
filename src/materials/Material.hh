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

namespace detail
{
   class MaterialConcept // The External Polymorphism design pattern
   {
   
   public:
      virtual ~MaterialConcept() = default;
      virtual std::unique_ptr<MaterialConcept> clone() const = 0; // The Prototype design pattern
      virtual void clone(MaterialConcept *memory) const = 0;      // The Prototype design pattern
   
   public:
      virtual void update_state() const = 0;

   };

   template <typename MaterialType, typename UpdateStrategy>
   class NonOwningMaterialModel; // Forward declaration

   template <typename MaterialType, typename UpdateStrategy>
   class OwningMaterialModel : public MaterialConcept
   {
   private:
      MaterialType   material_        ;
      UpdateStrategy update_algorithm_; //or material_integrator

   public:
      explicit OwningMaterialModel(MaterialType material, UpdateStrategy mat_integrator)
          : material_{std::move(material)}, update_algorithm_{std::move(mat_integrator)}
      {}

      void update_state() const override { update_algorithm_(material_); }

      std::unique_ptr<MaterialConcept> clone() const override{ // The Prototype design pattern
         return std::make_unique<OwningMaterialModel>(*this);
      }

      void clone(MaterialConcept *memory) const{
         using Model = NonOwningMaterialModel<MaterialType const, UpdateStrategy const>;

         std::construct_at(static_cast<Model *>(memory), material_, update_algorithm_);
      }
   };

   template <typename MaterialType, typename UpdateStrategy>
   class NonOwningMaterialModel : public MaterialConcept{
   public:
      NonOwningMaterialModel(MaterialType &material, UpdateStrategy &mat_integrator)
          : material_{std::addressof(material)}, update_algorithm_{std::addressof(mat_integrator)}
      {}

      void update_state() const override { (*update_algorithm_)(*material_); }

      std::unique_ptr<MaterialConcept> clone() const override{
         using Model = OwningMaterialModel<MaterialType, UpdateStrategy>;
         return std::make_unique<Model>(*material_, *update_algorithm_);
      }

      void clone(MaterialConcept *memory) const override{
         std::construct_at(static_cast<NonOwningMaterialModel *>(memory), *this);
      }

   private:
      MaterialType *material_{nullptr};
      UpdateStrategy *update_algorithm_{nullptr};
   };

} // namespace detail

class Material; // Forward declaration

class MaterialConstRef{
   friend class Material;

public:
   // Type 'MaterialType' and 'UpdateStrategy' are possibly cv qualified;
   // lvalue references prevent references to rvalues
   template <typename MaterialType, typename UpdateStrategy>
   MaterialConstRef(MaterialType &material, UpdateStrategy &mat_integrator){
      using Model =
          detail::NonOwningMaterialModel<MaterialType const, UpdateStrategy const>;
      static_assert(sizeof(Model) == MODEL_SIZE, "Invalid size detected");
      static_assert(alignof(Model) == alignof(void *), "Misaligned detected");

      std::construct_at(static_cast<Model *>(pimpl()), material, mat_integrator);
   }

   MaterialConstRef(Material &other);
   MaterialConstRef(Material const &other);

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

   detail::MaterialConcept *pimpl(){ // The Bridge design pattern
      return reinterpret_cast<detail::MaterialConcept *>(raw_.data());
   }

   detail::MaterialConcept const *pimpl() const{
      return reinterpret_cast<detail::MaterialConcept const *>(raw_.data());
   }

   // Expected size of a model instantiation:
   //     sizeof(MaterialType*) + sizeof(UpdateStrategy*) + sizeof(vptr)
   static constexpr std::size_t MODEL_SIZE = 3U * sizeof(void *);

   alignas(void *) std::array<std::byte, MODEL_SIZE> raw_;
};

class Material
{
   friend class MaterialConstRef;

   std::unique_ptr<detail::MaterialConcept> pimpl_; // The Bridge design pattern

public:
   
   template <typename MaterialType, typename UpdateStrategy>
   Material(MaterialType material, UpdateStrategy mat_integrator){
      using Model = detail::OwningMaterialModel<MaterialType, UpdateStrategy>;
      pimpl_ = std::make_unique<Model>(std::move(material), std::move(mat_integrator));
   }

   Material(Material         const &other) : pimpl_(other.pimpl_->clone()){}
   Material(MaterialConstRef const &other) : pimpl_{other.pimpl()->clone()}{}

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

MaterialConstRef::MaterialConstRef(Material &other){
   other.pimpl_->clone(pimpl());
}

MaterialConstRef::MaterialConstRef(Material const &other){
   other.pimpl_->clone(pimpl());
}





#endif