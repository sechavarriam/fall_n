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


/*
Every element type must have defined:
     i) His spacial Dimension ------------> (Dim)   
    ii) The number of nodes --------------> (nNodes)
   iii) The number of DoF ----------------> (nDoF)  = Dim*nNodes (Defaulted for continuum displacement Elements)
    iv) The number of integration Points -> (nGauss)= 0 (Defaulted for analytical integrated elements element.g. classical stiffnes beam) 
This has to be known at compile time.
*/


template<std::size_t Dim, std::size_t nNodes, std::size_t nDoF=Dim*nNodes> 
requires topology::EmbeddableInSpace<Dim>
class ElementBase{
  public:
    //static constexpr std::size_t dim     = Dim   ;

    static constexpr std::size_t num_nodes(){return nNodes;}; // to be used in element concept 
    static constexpr std::size_t num_dof  (){return nDoF  ;}; // to be used in element concept

    void set_num_dofs() 
    {
      std::cout << "Unreachable thing " << std::endl;
    };

    uint  id()    const {return id_   ;};
    std::size_t const* nodes() const {return nodes_.data();};   

    void set_material_integrator()
    {
      std::cout << "Unreachable thing " << std::endl;
    };


  private:
    std::size_t id_ ; //tag

    //Array of node tags
    std::array<std::size_t,nNodes> nodes_; 

  protected:
    // Shape functions and derivatives.
    

  public:    

    ~ElementBase(){};
    ElementBase() = delete;

    ElementBase(uint&  tag, std::array<std::size_t,nNodes>&  NodeTAGS): id_{tag}, nodes_{NodeTAGS}{};
    ElementBase(uint&& tag, std::array<std::size_t,nNodes>&& NodeTAGS)
      : id_{tag},
        nodes_{std::forward<std::array<std::size_t,nNodes>>(NodeTAGS)}{};
        
    
    // TODO: Implement static_asserts to check in derived elements that the imput matches nNodes.   
};


// Dejemos este acá para despues usarlo en implementaciónes de algoritmos genéricos que involucren elementos.
// Se debe completar después en función de lo que se agregue a ElementBase y en función de lo que se necesite
// en la interfaz de Element (Type Erasure Wrapper)

template<typename E>
concept ElementType = requires(E element){
    {element.id()       } -> std::convertible_to<uint  >        ;
    {element.nodes()    } -> std::convertible_to<uint const*>   ;
    {element.num_nodes()} -> std::convertible_to<std::size_t>        ;
    {element.num_dof()  } -> std::convertible_to<std::size_t>        ;
    {element.compute_measure()};
};


#endif