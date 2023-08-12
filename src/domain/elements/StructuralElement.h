#ifndef FN_STRUCTURAL_ELEMENT
#define FN_STRUCTURAL_ELEMENT

#include "../Node.h"
#include "Element.h"


template<u_short Dim, u_short nNodes, u_short nDoF>
class StructuralElement: virtual public Element<Dim,nNodes,nDoF>{

 private:

   bool HasCurvature;

 protected:

   virtual void enable_curvature(){this->HasCurvature = true;} ; // For large displacements?
   virtual void disable_curvature(){this->HasCurvature = false; }; 

   StructuralElement(){};
   StructuralElement(int tag, Node<Dim> **nodes): Element<Dim,nNodes,nDoF>(tag,nodes){}

 public:
    virtual ~StructuralElement(){};

};

#endif