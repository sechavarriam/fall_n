#ifndef FALL_LINEAL_MATERIAL_ABSTRACTION_HH
#define FALL_LINEAL_MATERIAL_ABSTRACTION_HH

// Your code goes here

#include <memory>

#include "LinealRelation.hh"
#include "IsotropicRelation.hh"



template<typename LinealContitutiveRelation, typename StateStoregePolicy>
class LinealMaterial{
    
    LinealContitutiveRelation constitutive_relation_;
    StateStoregePolicy        state_; //e.g. Strain, displacement... 
    
    public:
    LinealMaterial(){};
    ~LinealMaterial(){};
};

#endif // LINEALMATERIAL_HH