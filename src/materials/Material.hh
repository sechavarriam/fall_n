#ifndef FN_MATERIAL
#define FN_MATERIAL

#include <memory>
#include <random>
#include <vector>

#include "../numerics/Tensor.hh"
#include "Strain.hh"

//---- <Material.h> ----------------------------------------------------------------------------------

#include <array>
#include <cstddef>
#include <memory>
#include <utility>

namespace detail
{

   class MaterialConcept // The External Polymorphism design pattern
   {
   public:
      virtual ~MaterialConcept() = default;
      virtual void draw() const = 0;
      virtual std::unique_ptr<MaterialConcept> clone() const = 0; // The Prototype design pattern
      virtual void clone(MaterialConcept *memory) const = 0;      // The Prototype design pattern
   };

   template <typename MaterialType, typename DrawStrategy>
   class NonOwningMaterialModel; // Forward declaration

   template <typename MaterialType, typename DrawStrategy>
   class OwningMaterialModel : public MaterialConcept
   {
   private:
      MaterialType material_;
      DrawStrategy drawer_;

   public:
      explicit OwningMaterialModel(MaterialType material, DrawStrategy drawer)
          : material_{std::move(material)}, drawer_{std::move(drawer)}
      {
      }

      void draw() const override { drawer_(material_); }

      std::unique_ptr<MaterialConcept> clone() const override // The Prototype design pattern
      {
         return std::make_unique<OwningMaterialModel>(*this);
      }

      void clone(MaterialConcept *memory) const
      {
         using Model = NonOwningMaterialModel<MaterialType const, DrawStrategy const>;

         std::construct_at(static_cast<Model *>(memory), material_, drawer_);
      }
   };

   template <typename MaterialType, typename DrawStrategy>
   class NonOwningMaterialModel : public MaterialConcept
   {
   public:
      NonOwningMaterialModel(MaterialType &material, DrawStrategy &drawer)
          : material_{std::addressof(material)}, drawer_{std::addressof(drawer)}
      {
      }

      void draw() const override { (*drawer_)(*material_); }

      std::unique_ptr<MaterialConcept> clone() const override
      {
         using Model = OwningMaterialModel<MaterialType, DrawStrategy>;
         return std::make_unique<Model>(*material_, *drawer_);
      }

      void clone(MaterialConcept *memory) const override
      {
         std::construct_at(static_cast<NonOwningMaterialModel *>(memory), *this);
      }

   private:
      MaterialType *material_{nullptr};
      DrawStrategy *drawer_{nullptr};
   };

} // namespace detail

class Material; // Forward declaration

class MaterialConstRef
{
   friend class Material;

public:
   // Type 'MaterialType' and 'DrawStrategy' are possibly cv qualified;
   // lvalue references prevent references to rvalues
   template <typename MaterialType, typename DrawStrategy>
   MaterialConstRef(MaterialType &material, DrawStrategy &drawer)
   {
      using Model =
          detail::NonOwningMaterialModel<MaterialType const, DrawStrategy const>;
      static_assert(sizeof(Model) == MODEL_SIZE, "Invalid size detected");
      static_assert(alignof(Model) == alignof(void *), "Misaligned detected");

      std::construct_at(static_cast<Model *>(pimpl()), material, drawer);
   }

   MaterialConstRef(Material &other);
   MaterialConstRef(Material const &other);

   MaterialConstRef(MaterialConstRef const &other)
   {
      other.pimpl()->clone(pimpl());
   }

   MaterialConstRef &operator=(MaterialConstRef const &other)
   {
      // Copy-and-swap idiom
      MaterialConstRef copy(other);
      raw_.swap(copy.raw_);
      return *this;
   }

   ~MaterialConstRef(){
      std::destroy_at(pimpl()); // or: pimpl()->~MaterialConcept();
   }

   // Move operations explicitly not declared

private:
   friend void draw(MaterialConstRef const &material){
      material.pimpl()->draw();
   }

   detail::MaterialConcept *pimpl(){ // The Bridge design pattern
      return reinterpret_cast<detail::MaterialConcept *>(raw_.data());
   }

   detail::MaterialConcept const *pimpl() const{
      return reinterpret_cast<detail::MaterialConcept const *>(raw_.data());
   }

   // Expected size of a model instantiation:
   //     sizeof(MaterialType*) + sizeof(DrawStrategy*) + sizeof(vptr)
   static constexpr std::size_t MODEL_SIZE = 3U * sizeof(void *);

   alignas(void *) std::array<std::byte, MODEL_SIZE> raw_;
};

class Material
{
   friend class MaterialConstRef;

   std::unique_ptr<detail::MaterialConcept> pimpl_; // The Bridge design pattern

public:
   
   template <typename MaterialType, typename DrawStrategy>
   Material(MaterialType material, DrawStrategy drawer){
      using Model = detail::OwningMaterialModel<MaterialType, DrawStrategy>;
      pimpl_ = std::make_unique<Model>(std::move(material), std::move(drawer));
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
   friend void draw(Material const &material){
      material.pimpl_->draw();
   }
};

MaterialConstRef::MaterialConstRef(Material &other){
   other.pimpl_->clone(pimpl());
}

MaterialConstRef::MaterialConstRef(Material const &other){
   other.pimpl_->clone(pimpl());
}





#endif