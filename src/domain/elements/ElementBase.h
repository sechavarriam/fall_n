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
    
    private:       
    std::array<uint , nNodes> nodes_;       // Node indexes in node domain array.
    
    protected:

    // Shape functions and derivatives.
    std::array<std::function<double(double,double,double)>,nNodes> shape_function_ ; // Array of shape functions.
    
    
    // std::array<Node<Dim>*, nNodes> p_nodes_;// Pointers to nodes
    // std::array<uint,nDoF> dof_index_ ;      // dof Index position in domain array.
    //----------------------------------------------------------------------------------
    // std::array<IntegrationPoint<Dim>,nGauss> integration_points_ ; // Array of integration points. (Should be elements of the domain?) 
    // Initialized in zero by default static method to avoid garbage values.
    // SquareMatrix<nDoF> K_ = SquareMatrix<nDoF>::Zero(); 

  public:    
    
    ElementBase(){};
    ElementBase(int tag, std::array<uint,nNodes> NodeTAGS): Element(tag), nodes_(NodeTAGS){
      std::cout << "Element Base Copy Constructor" << std::endl;
    };

    
    ~ElementBase(){};
};


#endif