#ifndef FN_MATERIAL
#define FN_MATERIAL


#include <Eigen/Dense>
#include <asm-generic/errno.h>
#include <memory>
#include <random>
#include <vector>

#include "../numerics/Tensor.hh"
#include "Strain.hh"



namespace impl { //implementation details

class MaterialConcept 
{
    public:
    virtual ~MaterialConcept() = default;

    virtual void get_stress() const = 0;    // get cause 
    //virtual void update_strain() const = 0; // update effect
    
     // virtual void commit_state() const = 0; // From OpenSees Material  

    virtual std::unique_ptr<MaterialConcept> clone() const = 0; 
};

template<typename MaterialType, typename MaterialPolicy>
class OwningMaterialModel : public MaterialConcept{

    MaterialType   material_;
    MaterialPolicy policy_  ; // Uniaxial, Continuum, Hysteretic, Force-Displacement...

  public:
    std::unique_ptr<MaterialConcept> clone() const override{
        return std::make_unique<OwningMaterialModel<MaterialType,MaterialPolicy>>(*this);}; 

    explicit OwningMaterialModel(MaterialType mat , MaterialPolicy pol) :
        material_(std::move(mat)),
        policy_  (std::move(pol)){};

  public: //Implementations

  void get_stress(){material_.get_stress();};      
};

}
//template<ushort Dim, ushort Order> // Dim?
class Material{
    std::unique_ptr<impl::MaterialConcept> pimpl_; //Pointer to implementation details 

  public:
    template<typename MaterialType, typename MaterialPolicy>
    Material (MaterialType mat_, MaterialPolicy pol_)
    {
        using Model = impl::OwningMaterialModel<MaterialType,MaterialPolicy>;

        pimpl_ = std::make_unique<Model>(
            std::move(mat_),
            std::move(pol_));
    }

    ~Material() = default;
    // nVars: Number of state variables (strains)
    //private:        
        //Tensor<Dim,Order> strain;
        //std::vector<Tensor<Dim,Order>> strain_t_; //Store history of strains in simulation for memory materials
    //public:
    //    void preallocate_strain_t_(){};

};


#endif