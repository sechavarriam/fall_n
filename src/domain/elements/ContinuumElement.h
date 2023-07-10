#ifndef CONTINUUM_ELEMENT_H
#define CONTINUUM_ELEMENT_H

#include "Element.h"
#include "../../materials/Material.h"

class ContinuumElement:Element{
    
 private:
  Material Mat;
  
 public:
  ContinuumElement(){};

};

#endif