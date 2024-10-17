#ifndef FALL_N_FEM_ELEMENT_HH
#define FALL_N_FEM_ELEMENT_HH


//---- <Shape.h> ----------------------------------------------------------------------------------

#include <array>
#include <cstdlib>
#include <memory>
#include <utility>

// Simple Type Erausre

namespace impl{
   template<class MaterialPolicy>
   class FEM_ElementConcept{ 
      
      using StateVariableT = MaterialPolicy::StateVariableT;
      using StressT        = MaterialPolicy::StressType;     //Effect

   public:
      virtual constexpr ~FEM_ElementConcept() = default;
      virtual constexpr std::unique_ptr<FEM_ElementConcept> clone() const = 0; // The Prototype design pattern

   
   public:

      //virtual constexpr Matrix& C() const = 0; //The Compliance Matrix
      //virtual constexpr StateVariableT get_state() const = 0; //The current Value of the State Variable (or the head?)
      //virtual void update_state(const StateVariableT& state) = 0;  

   };

   template <typename MaterialType, typename UpdateStrategy>
   class OwningFEM_ElementModel : public FEM_ElementConcept<typename MaterialType::MaterialPolicy>
   {
   private:

      using MaterialPolicy = typename MaterialType::MaterialPolicy;
      using StateVariableT = MaterialPolicy::StateVariableT;

      MaterialType   material_        ;
      UpdateStrategy update_algorithm_; //or material_integrator

   public:
      
      //Matrix& C() const override {return material_.C();}; //The Compliance Matrix
      //StateVariableT get_state() const override {return material_.get_state();}; //CurrentValue
      //void update_state(const StateVariableT& state) override {material_.update_state(state);};

      explicit OwningFEM_ElementModel(MaterialType material, UpdateStrategy material_integrator): 
          material_        {std::move(material)}, 
          update_algorithm_{std::move(material_integrator)}
      {}

      std::unique_ptr<FEM_ElementConcept<MaterialPolicy>> clone() const override{ // The Prototype design pattern
         return std::make_unique<OwningFEM_ElementModel>(*this);
      }

   }; // OwningFEM_ElementModel

} // namespace impl

template<class MaterialPolicy>
class FEM_Element
{
   using StateVariableT = MaterialPolicy::StateVariableT;

   std::unique_ptr<impl::FEM_ElementConcept<MaterialPolicy>> pimpl_; // The Bridge design pattern

public:

   //Matrix& C() const {return pimpl_->C();}; //The Compliance Matrix
   //StateVariableT get_state() const {return pimpl_->get_state();};
   //void update_state(const StateVariableT& state) {pimpl_->update_state(state);};

public:
   
   template <typename MaterialType, typename UpdateStrategy>
   FEM_Element(MaterialType material, UpdateStrategy material_integrator){
      using Model = impl::OwningFEM_ElementModel<MaterialType, UpdateStrategy>;
      pimpl_ = std::make_unique<Model>(
         std::move(material), 
         std::move(material_integrator)
         );
   }

   FEM_Element(FEM_Element<MaterialPolicy> const &other) : pimpl_(other.pimpl_->clone()){}

   FEM_Element &operator=(FEM_Element const &other)
   {
      // Copy-and-Swap Idiom
      FEM_Element copy(other);
      pimpl_.swap(copy.pimpl_);
      return *this;
   }

   ~FEM_Element() = default;
   FEM_Element(FEM_Element &&) = default;
   FEM_Element &operator=(FEM_Element &&) = default;

private: //Hidden Friends (Free Functions)


};




#endif // FALL_N_FEM_ELEMENT_HH