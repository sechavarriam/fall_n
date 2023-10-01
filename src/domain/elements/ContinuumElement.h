#ifndef FN_CONTINUUM_ELEMENT
#define FN_CONTINUUM_ELEMENT

#include "../Node.h"
#include "ElementBase.h"

typedef unsigned short ushort;
typedef unsigned int   uint  ;

template<ushort Dim, ushort nNodes, ushort nDoF=Dim*nNodes, ushort nGauss=0> 
requires Topology::EmbeddableInSpace<Dim> 
class ContinuumElement: public ElementBase<Dim,nNodes,nDoF,nGauss>{

  public:

    ContinuumElement(){};

    template<typename ListType>
    ContinuumElement(int tag, ListType&& nodes): 
    ElementBase<Dim,nNodes,nDoF,nGauss>(tag,std::forward<ListType>(nodes))
    {
      std::cout << "Continuum Element Perfect Forwarding Constructor" << std::endl;
    }

    virtual ~ContinuumElement(){};
};

#endif