#ifndef FN_MATERIAL
#define FN_MATERIAL


#include <Eigen/Dense>
#include <asm-generic/errno.h>
#include <memory>
#include <random>
#include <vector>

#include "../numerics/Tensor.hh"
#include "Strain.hh"


// TypeErasure with Manual Virtual dispatch.

//template<typename MaterialPolicy> // MaterialPolicy is a concept that defines the material behavior (uniaxial, continuum, plane....)
class Material
{
 public:
   template< typename MaterialType
           , typename UpdateStrategy >
   Material( MaterialType material, UpdateStrategy updater )
      : pimpl_(
            new OwningModel<MaterialType,UpdateStrategy>( std::move(material)
                                                        , std::move(updater) )
          , []( void* materialBytes ){
               using Model = OwningModel<MaterialType,UpdateStrategy>;
               auto* const model = static_cast<Model*>(materialBytes);
               delete model;
            } )
      , clone_(
            []( void* materialBytes ) -> void* {
               using Model = OwningModel<MaterialType,UpdateStrategy>;
               auto* const model = static_cast<Model*>(materialBytes);
               return new Model( *model );
            } )
      , getStress_(
      []( void* materialBytes ){
         using Model = OwningModel<MaterialType,UpdateStrategy>;
         auto* const model = static_cast<Model*>(materialBytes);
         (model->updater_)( model->shape_ );
      } )
   {}

   Material( Material const& other )
      : pimpl_( other.clone_( other.pimpl_.get() ), other.pimpl_.get_deleter() )
      , clone_( other.clone_ )
      , getStress_ ( other.getStress_ )
   {}

   Material& operator=( Material const& other )
   {
      // Copy-and-Swap Idiom
      using std::swap;
      Material copy( other );
      swap( pimpl_, copy.pimpl_ );
      swap( clone_, copy.clone_ );

      swap( getStress_, copy.getStress_ );
      
      return *this;
   }

   ~Material() = default;
   Material( Material&& ) = default;
   Material& operator=( Material&& ) = default;

 private:
   friend void getStress( Material const& material )
   {
      material.getStress_( material.pimpl_.get() );
   }

   template< typename MaterialType
           , typename UpdateStrategy >
   struct OwningModel
   {
      OwningModel( MaterialType value, UpdateStrategy updater )
         : shape_( std::move(value) )
         , updater_( std::move(updater) )
      {}

      MaterialType shape_;
      UpdateStrategy updater_;
   };

   using DestroyOperation = void(void*);
   using CloneOperation   = void*(void*);

   using UpdaterOperation    = void(void*);
   

   std::unique_ptr<void,DestroyOperation*> pimpl_;
   CloneOperation* clone_{ nullptr };

   UpdaterOperation*  getStress_ { nullptr };
   
};




#endif