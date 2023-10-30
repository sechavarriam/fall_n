#ifndef FN_STRUCTURAL_ELEMENT
#define FN_STRUCTURAL_ELEMENT

#include "../Node.h"
#include "ElementBase.h"

typedef unsigned short ushort;
typedef unsigned int   uint  ;

//template<ushort Dim,
//         ushort nNodes, 
//         ushort nDoF>
//class StructuralElement: virtual public ElementBase<Dim,nNodes,nDoF>{
//
// private:
//
//   bool HasCurvature;
//
// protected:
//
//   virtual void enable_curvature(){this->HasCurvature = true;} ; // For large displacements?
//   virtual void disable_curvature(){this->HasCurvature = false; }; 
//
//   StructuralElement(){};
//   
//   //StructuralElement(int tag, Node<Dim>** nodes): ElementBase<Dim,nNodes,nDoF>(tag,nodes){} // Should not be used yet
//   StructuralElement(int tag, std::array<uint,nNodes> NodeTAGS): ElementBase<Dim,nNodes,nDoF>(tag,NodeTAGS){}
//
// public:
//
//    virtual ~StructuralElement(){};
//
//};

#endif