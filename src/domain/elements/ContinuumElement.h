#ifndef FN_CONTINUUM_ELEMENT
#define FN_CONTINUUM_ELEMENT

#include "Element.h"
#include "../../materials/Material.h"


template<unsigned int Dim, unsigned int nDoF> 
class ContinuumElement : public Element<Dim,nDoF> {
    
 private:
  Material Mat;
  
 public:
  ContinuumElement(){};

};

#endif