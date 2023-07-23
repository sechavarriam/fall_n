#ifndef CONTINUUM_ELEMENT_H
#define CONTINUUM_ELEMENT_H

#include "Element.h"
#include "../../materials/Material.h"


template<int nDoF>
class ContinuumElement : public Element<nDoF>{
    
 private:
  Material Mat;
  
 public:
  ContinuumElement(){};

};

#endif