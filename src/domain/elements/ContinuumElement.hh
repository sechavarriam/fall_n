#ifndef FN_CONTINUUM_ELEMENT
#define FN_CONTINUUM_ELEMENT

#include "../Node.hh"
#include "ElementBase.hh"

typedef unsigned short ushort;
typedef unsigned int   uint  ;
//
//template<ushort Dim,
//         ushort nNodes,
//         ushort nDoF=Dim*nNodes> 
//class ContinuumElement: public ElementBase<Dim,nNodes,nDoF>{
//
//  public:
//
//    ContinuumElement() =  delete;
//
//    //ContinuumElement(uint tag, std::array<ushort,nNodes> NodeTAGS)
//    //  : ElementBase<Dim,nNodes,nDoF>(tag,NodeTAGS){}
//
//    ContinuumElement(uint& tag, std::array<ushort,nNodes>& NodeTAGS)
//      : ElementBase<Dim,nNodes,nDoF>(tag,NodeTAGS){};
//
//    ContinuumElement(uint&& tag, std::array<ushort,nNodes>&& nodes)
//      : ElementBase<Dim,nNodes,nDoF>{
//          std::forward<uint>(tag),
//          std::forward<std::array<ushort,nNodes>>(nodes)}{};
//
//   ~ContinuumElement(){};
//};

#endif