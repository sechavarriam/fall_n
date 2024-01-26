#ifndef FN_ELEMENTBASE_H
#define FN_ELEMENTBASE_H

#include <cstddef>
#include <functional>
#include <utility>
#include <array>
#include <iostream>

#include <concepts>

//#include "../../numerics/Matrix.h"

#include "../../geometry/Topology.hh"

#include "Element.hh"
#include "../Node.hh"
#include "../../numerics/numerical_integration/Quadrature.hh"

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


template<ushort Dim, ushort nNodes, ushort nDoF=Dim*nNodes> 
requires topology::EmbeddableInSpace<Dim>
class ElementBase{
  public:
    //static constexpr ushort dim     = Dim   ;

    static constexpr ushort num_nodes(){return nNodes;}; // to be used in element concept 
    static constexpr ushort num_dof  (){return nDoF  ;}; // to be used in element concept

    uint  id()    const {return id_   ;};
    ushort const* nodes() const {return nodes_.data();};   


    void compute_measure(){};
    void compute_shape_functions(){};             //Policy
    void compute_shape_functions_derivatives(){}; //Policy
    void compute_stiffness_matrix(){};
    void compute_mass_matrix(){};
    void compute_damping_matrix(){};              //Policy

  private:
    std::size_t id_ ; //tag

    //Array of node tags
    std::array<ushort,nNodes> nodes_; 

  protected:
    // Shape functions and derivatives.
    

  public:    

    ~ElementBase(){};
    ElementBase() = delete;

    ElementBase(uint&  tag, std::array<ushort,nNodes>&  NodeTAGS): id_{tag}, nodes_{NodeTAGS}{};
    ElementBase(uint&& tag, std::array<ushort,nNodes>&& NodeTAGS)
      : id_{tag},
        nodes_{std::forward<std::array<ushort,nNodes>>(NodeTAGS)}{};
        
    
    // TODO: Implement static_asserts to check in derived elements that the imput matches nNodes.   
};


// Dejemos este acá para despues usarlo en implementaciónes de algoritmos genéricos que involucren elementos.
// Se debe completar después en función de lo que se agregue a ElementBase y en función de lo que se necesite
// en la interfaz de Element (Type Erasure Wrapper)

template<typename E>
concept ElementType = requires(E element){
    {element.id()       } -> std::convertible_to<uint  >        ;
    {element.nodes()    } -> std::convertible_to<uint const*>   ;
    {element.num_nodes()} -> std::convertible_to<ushort>        ;
    {element.num_dof()  } -> std::convertible_to<ushort>        ;
    {element.compute_measure()};
};


#endif