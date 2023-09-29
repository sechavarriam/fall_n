#ifndef FN_ELEMENTBASE_H
#define FN_ELEMENTBASE_H

#include <vector>
#include <array>
#include <variant>

#include <iostream>
#include <concepts>

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
      
    std::array<uint,nDoF>   dof_index_ ; // dof   Index position in domain array.

    // Agrupar estos dos en un variant? ------------------------------------------------
    std::variant<uint,Node<Dim>*> nodes_; //Posición en el arreglo, o puntero directo al nodo. No los dos.
                                          // Para acceder a la info a través del nodo, o del índice.

    std::array<uint,nNodes>       nodes_index_  ;// nodes Index position in domain array.
    std::array<Node<Dim>*,nNodes> nodes_pointer_;// Pointer to nodes.
    //----------------------------------------------------------------------------------

    std::array<IntegrationPoint<Dim>,nGauss> integration_points_ ; // Array of integration points. (Should be elements of the domain?) 

    // Initialized in zero by default static method to avoid garbage values.
    SquareMatrix<nDoF> K_ = SquareMatrix<nDoF>::Zero(); 
    
    // TRANSFORMATION LOCAL DISPLACEMENTS TO GLOBAL DISPLACEMENTS
    // DISPLACEMENT INTERPOLATION FUNCTION. A functor with state?
    //                                      A lambda?
    //                                      A regular member function + coeficients.




  public:    
    virtual void set_node_index(){};
    virtual void set_dof_index(){};
    
    ElementBase(){};

    // TODO: Implement Perfec Forwarding.
    ElementBase(int tag, Node<Dim>** nodes): Element(tag), nodes_(nodes){};
    ElementBase(int tag, std::array<uint,nNodes> NodeTAGS): Element(tag), nodes_index_(NodeTAGS){};
    
    virtual ~ElementBase(){};
};

#endif