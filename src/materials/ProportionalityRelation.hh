#ifndef FN_PROPORTIONALITY_RELATION
#define FN_PROPORTIONALITY_RELATION

#include <Eigen/Dense>


#include "Strain.hh"
#include "Stress.hh" // REVISAR. PENSAR EN LA CLASE TENSOR.

#include "ConstitutiveRelation.hh"

// F cause
// U efect


template<typename F,typename R,typename U=F> 
class ProportionalityRelation : public ConstitutiveRelation<F,R>{

    
    //F=K*U
    //U=C*K
};



 
template<> 
class ProportionalityRelation<Stress<3>,Strain<3>>{
};
    
    
    //F=K*U
    //U=C*K




#endif