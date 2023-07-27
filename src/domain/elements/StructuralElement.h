#ifndef FN_STRUCTURAL_ELEMENT
#define FN_STRUCTURAL_ELEMENT

#include "Element.h"

template<unsigned int Dim, unsigned int nDoF> 
class StructuralElement: virtual public Element<Dim,nDoF>{

 private:

   bool HasCurvature;

 protected:

   virtual void enable_curvature(){this->HasCurvature = true;} ; // For large displacements?
   virtual void disable_curvature(){this->HasCurvature = false; }; 

   StructuralElement(){};
   StructuralElement(int tag, Node **nodes): Element<Dim,nDoF>(tag,nodes){}

 public:
    virtual ~StructuralElement(){};

};

#endif