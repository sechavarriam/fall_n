#ifndef FN_ELEMENTBASE_H
#define FN_ELEMENTBASE_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <utility>
#include <vector>
#include <concepts>

#include <iostream>

#include "../../numerics/Matrix.h"

#include "../../geometry/Topology.h"

#include "Element.h"
#include "../Node.h"

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

template<ushort Dim, ushort nNodes, ushort nDoF=Dim*nNodes> 
requires Topology::EmbeddableInSpace<Dim> 
class ElementBase{
  public:

    static constexpr ushort dim     = Dim   ;
    static constexpr ushort n_nodes = nNodes;
    static constexpr ushort n_nof   = nDoF  ;

    static constexpr ushort num_nodes(){return nNodes;}; // to be used in element concept 
    static constexpr ushort num_dof  (){return nDoF  ;}; // to be used in element concept

    uint  id()    const {return id_   ;};
    ushort const* nodes() const {return nodes_.data();};   

  private:
    uint id_ ; //tag
    std::array<ushort,nNodes> nodes_; //Array of node tags

  protected:
    // Shape functions and derivatives.
    
  public:    
    ElementBase() = delete;

    ElementBase(uint&  tag, std::array<ushort,nNodes>&  NodeTAGS): id_{tag}, nodes_{NodeTAGS}{};
    ElementBase(uint&& tag, std::array<ushort,nNodes>&& NodeTAGS)
      : id_{tag},
        nodes_{std::forward<std::array<ushort,nNodes>>(NodeTAGS)}{};


    ~ElementBase(){};

    // TODO: Implement static_asserts to check in derived elements that the imput matches nNodes.
    
};

#endif