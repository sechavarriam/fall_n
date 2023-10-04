#ifndef FN_ELEMENTBASE_H
#define FN_ELEMENTBASE_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <utility>
#include <vector>
#include <variant>
#include <concepts>
#include <initializer_list>

#include <iostream>

#include "../../numerics/Matrix.h"

#include "../Topology.h"

#include "Element.h"
#include "../Node.h"
#include "../IntegrationPoint.h"

typedef unsigned short ushort;
typedef unsigned int   uint  ;

/*
Every element type must have defined:
     i) His spacial Dimension ------------> (Dim)   
    ii) The number of nodes --------------> (nNodes)
   iii) The number of DoF ----------------> (nDoF)  = Dim*nNodes (Defaulted for continuum displacement Elements)
    iv) The number of integration Points -> (nGauss)= 0 (Defaulted for analytical integrated elements e.g. classical stiffnes beam) 
This has to be known at compile time.
*/

template<ushort Dim, ushort nNodes, ushort nDoF=Dim*nNodes, ushort nGauss=0> 
requires Topology::EmbeddableInSpace<Dim> 
class ElementBase: public Element{

  public:
    static constexpr ushort num_Nodes = nNodes;
    static constexpr ushort num_DoF   = nDoF  ;
    static constexpr ushort num_Gauss = nGauss;
           
    std::array<ushort , nNodes> nodes_;       // Node indexes in node domain array.
    
  protected:
    // Shape functions and derivatives.
    
  public:    
    ElementBase() = delete;
    ElementBase(int tag, std::array<ushort,num_Nodes> NodeTAGS): Element{tag}, nodes_{NodeTAGS}{};

    ~ElementBase(){};
};


#endif