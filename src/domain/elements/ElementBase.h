#ifndef FN_ELEMENTBASE_H
#define FN_ELEMENTBASE_H


#include <cstddef>
#include <functional>
#include <utility>
#include <array>
#include <iostream>

#include <concepts>

//#include "../../numerics/Matrix.h"

#include "../../geometry/Topology.h"

#include "Element.h"
#include "../Node.h"
#include "../../numerics/numerical_integration/Quadrature.h"

typedef unsigned short ushort;
typedef unsigned int   uint  ;

/*
Every element type must have defined:
     i) His spacial Dimension ------------> (Dim)   
    ii) The number of nodes --------------> (nNodes)
   iii) The number of DoF ----------------> (nDoF)  = Dim*nNodes (Defaulted for continuum displacement Elements)
    iv) The number of integration Points -> (nGauss)= 0 (Defaulted for analytical integrated elements element.g. classical stiffnes beam) 
This has to be known at compile time.
*/

//template<ushort Dim, ushort nNodes, ushort nDoF=Dim*nNodes> 
//requires Topology::EmbeddableInSpace<Dim> 
//class ElementBase{
//  public:
//
//    static constexpr ushort dim     = Dim   ;
//    static constexpr ushort n_nodes = nNodes;
//    static constexpr ushort n_nof   = nDoF  ;
//
//    static constexpr ushort num_nodes(){return nNodes;}; // to be used in element concept 
//    static constexpr ushort num_dof  (){return nDoF  ;}; // to be used in element concept
//
//    uint  id()    const {return id_   ;};
//    ushort const* nodes() const {return nodes_.data();};   
//
//  private:
//    uint id_ ; //tag
//
//    //Array of node tags
//    std::array<ushort,nNodes> nodes_; 
//
//    //Gauss points
//    //std::array<IntegrationPoint<dim>,nGauss> gauss_points_;
//  protected:
//    // Shape functions and derivatives.
//    
//  public:    
//    ElementBase() = delete;
//
//    ElementBase(uint&  tag, std::array<ushort,nNodes>&  NodeTAGS): id_{tag}, nodes_{NodeTAGS}{};
//    ElementBase(uint&& tag, std::array<ushort,nNodes>&& NodeTAGS)
//      : id_{tag},
//        nodes_{std::forward<std::array<ushort,nNodes>>(NodeTAGS)}{};
//        
//    ~ElementBase(){};
//    // TODO: Implement static_asserts to check in derived elements that the imput matches nNodes.   
//};


// Static polymorphism version of ElementBase with CRTP and concepts.
template<typename E>
concept ElementType = requires(E element){
    {element.id()       } -> std::convertible_to<uint  >        ;
    {element.nodes()    } -> std::convertible_to<uint const*> ;
    {element.num_nodes()} -> std::convertible_to<ushort>        ;
    {element.num_dof()  } -> std::convertible_to<ushort>        ;
};


template<ElementType T, ushort Dim, ushort nNodes,ushort nDoF=Dim*nNodes> 
requires topology::EmbeddableInSpace<Dim>
class ElementBase{

  public:
};



// Curiously recurring template pattern (CRTP) version of ElementBase
//template<typename ElementType, 
//         ushort Dim,
//         ushort nNodes,
//         ushort nDoF=Dim*nNodes>
//requires ElementPrimitive<ElementType,Dim,nNodes,nDoF>
//class ElementBaseCRTP: public ElementBase<Dim,nNodes,nDoF>{
//  public:
//    using ElementBase<Dim,nNodes,nDoF>::ElementBase;
//};

#endif