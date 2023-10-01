#ifndef FN_ELEMENTBASE_H
#define FN_ELEMENTBASE_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>
#include <vector>
#include <variant>
#include <concepts>
#include <initializer_list>

#include <iostream>

#include <Eigen/Dense>

#include "../Topology.h"

#include "Element.h"
#include "../Node.h"
#include "../IntegrationPoint.h"

typedef unsigned short ushort;
typedef unsigned int   uint  ;


template<ushort N> using SquareMatrix = Eigen::Matrix<double, N, N> ;

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
    std::array<Node<Dim>*, nNodes> p_nodes_;// Pointers to nodes
    std::array<uint,nDoF> dof_index_ ;      // dof Index position in domain array.

    //----------------------------------------------------------------------------------
    std::array<IntegrationPoint<Dim>,nGauss> integration_points_ ; // Array of integration points. (Should be elements of the domain?) 

    // Initialized in zero by default static method to avoid garbage values.
    SquareMatrix<nDoF> K_ = SquareMatrix<nDoF>::Zero(); 

  public:    
    //virtual void set_node_index(){};
    //virtual void set_dof_index(){};
    
    ElementBase(){};
    ElementBase(int tag, std::array<uint,nNodes> NodeTAGS): Element(tag), nodes_(NodeTAGS){
      std::cout << "Element Base Copy Constructor" << std::endl;
    };

    //Perfect Forwarding Constructor - Doesn't work yet.
    // requires clause to constrain a template parameter T to be a std::array<uint, nNodes> type.
    //template<typename T>
    //requires std::is_same_v<std::remove_cvref_t<T>,std::array<uint,nNodes>>
    //ElementBase(int tag, T&& args): Element(tag), nodes_{std::forward<T>(args)}{
    //  std::cout << "Element Base Perfect Forwarding Constructor" << std::endl;
    //};
    
    ~ElementBase(){};
};


#endif