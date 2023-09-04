#ifndef FN_CONTINUUM_ELEMENT
#define FN_CONTINUUM_ELEMENT

#include "../Node.h"
#include "ElementBase.h"

template<u_short Dim, u_short nNodes, u_short nDoF, u_short nGauss>
class ContinuumElement: virtual public ElementBase<Dim,nNodes,nDoF,nGauss>{
//    
// private:
//  Material Mat;
//  
  public:

    ContinuumElement(){};
    ContinuumElement(int tag, std::array<u_int,nNodes> NodeTAGS): ElementBase<Dim,nNodes,nDoF,nGauss>(tag,NodeTAGS){}

    virtual ~ContinuumElement(){};
};

#endif