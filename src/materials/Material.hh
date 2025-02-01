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
      using StressT        = MaterialPolicy::StressType;

   public:
      virtual constexpr ~MaterialConcept() = default;
      virtual constexpr std::unique_ptr<MaterialConcept> clone() const = 0; // The Prototype design pattern

   
   public:

      
      virtual constexpr Eigen::Matrix<double, StateVariableT::num_components, StateVariableT::num_components> C() = 0; //The Compliance DeprecatedDenseMatrix
       
      virtual constexpr StateVariableT get_state() const = 0; //The current Value of the State Variable (or the head?)

      virtual void update_state(const StateVariableT& state) = 0;  

   };

   template <typename MaterialType, typename UpdateStrategy>
   class OwningMaterialModel : public MaterialConcept<typename MaterialType::MaterialPolicy>
   {
   private:

      using MaterialPolicy = typename MaterialType::MaterialPolicy;
      using StateVariableT = MaterialPolicy::StateVariableT;

      MaterialType   material_        ;
      UpdateStrategy update_algorithm_; //or material_integrator

   public:

      //Eigen::Ref<const Eigen::Matrix<double, StateVariableT::num_components, StateVariableT::num_components>> C() const override {return material_.C();}; //The Compliance DeprecatedDenseMatrix      
      Eigen::Matrix<double, StateVariableT::num_components, StateVariableT::num_components> C() override {return material_.C();}; //The Compliance DeprecatedDenseMatrix

      StateVariableT get_state() const override {return material_.get_state();}; //CurrentValue

      void update_state(const StateVariableT& state) override {material_.update_state(state);};

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
class Material
{
   using StateVariableT = MaterialPolicy::StateVariableT;

   std::unique_ptr<impl::MaterialConcept<MaterialPolicy>> pimpl_; // The Bridge design pattern

public:

   Eigen::Matrix<double, StateVariableT::num_components, StateVariableT::num_components> C() const {return pimpl_->C();}; //The Compliance DeprecatedDenseMatrix

   StateVariableT get_state() const {return pimpl_->get_state();};
   void update_state(const StateVariableT& state) {pimpl_->update_state(state);};

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


/*


// Non-owning Type Erasure V2
namespace impl //Implementation Details
{
   template<class MaterialPolicy>
   class MaterialConcept{ 
      using StateVariableT = MaterialPolicy::StateVariableT;
      using StressT        = MaterialPolicy::StressType;

   public:
      virtual constexpr ~MaterialConcept() = default;
      virtual constexpr std::unique_ptr<MaterialConcept> clone() = 0; // The Prototype design pattern

      virtual constexpr void clone(MaterialConcept *address) = 0;      
   
   public:
      virtual constexpr StateVariableT get_state() const = 0; //The current Value of the State Variable (or the head?)
      
      virtual void update_state(StateVariableT& state) = 0;  

   };

   template <typename MaterialType, typename UpdateStrategy> 
   class NonOwningMaterialModel; //Forward Declaration

   template <typename MaterialType, typename UpdateStrategy>
   class OwningMaterialModel : public MaterialConcept<typename MaterialType::MaterialPolicy>
   {
   private:

      using MaterialPolicy = typename MaterialType::MaterialPolicy;
      using StateVariableT = MaterialPolicy::StateVariableT;

      MaterialType   material_        ;
      UpdateStrategy update_algorithm_; //or material_integrator

   public:
      
      StateVariableT get_state() const override {return material_.get_state();}; //CurrentValue

      void update_state(StateVariableT& state) override {material_.update_state(state);};

      explicit OwningMaterialModel(MaterialType material, UpdateStrategy material_integrator): 
          material_        {std::move(material)}, 
          update_algorithm_{std::move(material_integrator)}
      {}

      std::unique_ptr<MaterialConcept<MaterialPolicy>> clone() override{ // The Prototype design pattern
         return std::make_unique<OwningMaterialModel>(*this);
      }

      void clone(MaterialConcept<MaterialPolicy>* address) override{
         using Model = NonOwningMaterialModel<MaterialType const, UpdateStrategy const>;   
         std::construct_at(static_cast<Model*>(address), material_, update_algorithm_);

      }
   }; // OwningMaterialModel

   template <typename MaterialType, typename UpdateStrategy>
   class NonOwningMaterialModel : public MaterialConcept<typename MaterialType::MaterialPolicy>{
   
   private:
      using MaterialPolicy = typename MaterialType::MaterialPolicy;
      using StateVariableT = MaterialPolicy::StateVariableT;

      MaterialType   *material_        {nullptr};
      UpdateStrategy *update_algorithm_{nullptr};
   
   public:

      StateVariableT get_state() const override {return material_->get_state();}; //CurrentValue

      void update_state(StateVariableT& state) override {material_->update_state(state);};

      NonOwningMaterialModel(MaterialType &material, UpdateStrategy &material_integrator):
           material_{std::addressof(material)},
           update_algorithm_{std::addressof(material_integrator)}
      {}

      std::unique_ptr<MaterialConcept<MaterialPolicy>> clone() override{
         using Model = OwningMaterialModel<MaterialType, UpdateStrategy>;
         return std::make_unique<Model>(*material_, *update_algorithm_);
      }

      void clone(MaterialConcept<MaterialPolicy> *address) override{
         std::construct_at(static_cast<NonOwningMaterialModel *>(address), *this);
      }
   };

} // namespace impl

template<class MaterialPolicy>
class Material; // Forward declaration

template<class MaterialPolicy>
class MaterialConstRef{
   friend class Material<MaterialPolicy>;
   using StateVariableT = MaterialPolicy::StateVariableT;

public:

   StateVariableT get_state() const {return pimpl()->get_state();};

   void update_state(StateVariableT& state) {pimpl()->update_state(state);};

private:   

   // Expected size of a model instantiation: sizeof(MaterialType*) + sizeof(UpdateStrategy*) + sizeof(vptr)
   static constexpr std::size_t MODEL_SIZE = 3U * sizeof(void *);
   alignas(void *) std::array<std::byte, MODEL_SIZE> raw_;

public:
   // Type 'MaterialType' and 'UpdateStrategy' are possibly cv qualified;
   // lvalue references prevent references to rvalues
   template <typename MaterialType, typename UpdateStrategy>
   MaterialConstRef(MaterialType &material, UpdateStrategy &material_integrator){
      using Model = impl::NonOwningMaterialModel<MaterialType const, UpdateStrategy const>;
      static_assert(sizeof(Model) == MODEL_SIZE, "Invalid size detected");
      static_assert(alignof(Model) == alignof(void *), "Misaligned detected");

      std::construct_at(static_cast<Model *>(pimpl()), material, material_integrator);
   }

   MaterialConstRef(Material<MaterialPolicy> &other);
   MaterialConstRef(Material<MaterialPolicy> const &other);

   MaterialConstRef(MaterialConstRef const &other){
      other.pimpl()->clone(pimpl());
   }

   MaterialConstRef &operator=(MaterialConstRef const &other){
      MaterialConstRef copy(other); // Copy-and-swap idiom
      raw_.swap(copy.raw_);
      return *this;
   }

   ~MaterialConstRef(){std::destroy_at(pimpl());} 
   // Move operations explicitly not declared

private:

   impl::MaterialConcept<MaterialPolicy> *pimpl(){ // The Bridge design pattern
      return reinterpret_cast<impl::MaterialConcept<MaterialPolicy>*>(raw_.data());
   }

   impl::MaterialConcept<MaterialPolicy> const *pimpl() const{
      return reinterpret_cast<impl::MaterialConcept<MaterialPolicy> const*>(raw_.data());
   }

}; // MaterialConstRef

template<class MaterialPolicy>
class Material
{
   friend class MaterialConstRef<MaterialPolicy>;

   using StateVariableT = MaterialPolicy::StateVariableT;

   std::unique_ptr<impl::MaterialConcept<MaterialPolicy>> pimpl_; // The Bridge design pattern

public:

   StateVariableT get_state() const {return pimpl_->get_state();};
   void update_state(StateVariableT& state) {pimpl_->update_state(state);};

public:
   
   template <typename MaterialType, typename UpdateStrategy>
   Material(MaterialType material, UpdateStrategy material_integrator){
      using Model = impl::OwningMaterialModel<MaterialType, UpdateStrategy>;
      pimpl_ = std::make_unique<Model>(
         std::move(material), 
         std::move(material_integrator)
         );
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

private: //Hidden Friends (Free Functions)

}; // Material

template<class MaterialPolicy>
inline MaterialConstRef<MaterialPolicy>::MaterialConstRef(Material<MaterialPolicy> &other){
   other.pimpl_->clone(pimpl());
}

template<class MaterialPolicy>
inline MaterialConstRef<MaterialPolicy>::MaterialConstRef(Material<MaterialPolicy> const &other){
   other.pimpl_->clone(pimpl());
}

*/



#endif