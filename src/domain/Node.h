#ifndef FN_NODE
#define FN_NODE

#include <array>
#include <cmath>

#include <iostream>
#include <memory> 
#include <concepts>
#include <initializer_list>
#include <utility>

#include "DoF.h"
#include "../geometry/Point.h"
#include "../geometry/Topology.h"

typedef unsigned short ushort;
typedef unsigned int   uint  ;

template<std::size_t Dim>//, ushort nDoF=Dim> 
class Node : public geometry::Point<Dim>{

  private:

    std::size_t id_{} ; 
    std::size_t num_dof_{0}; 
  
  public:

    domain::DoF_Interfase dof_;

  public:

    std::integral auto id()     {return id_     ;}
    std::integral auto num_dof(){return num_dof_;}

    void set_id     (const std::size_t& id) noexcept {id_ = id;};
    void set_num_dof(const std::size_t& n ) noexcept {num_dof_ = n;};    

    
    void set_dof_interfase(std::initializer_list<std::size_t>&& dofs_index)
    {
      dof_.set_index(std::forward<std::initializer_list<std::size_t>>(dofs_index));
    };

    void set_dof_interfase(){dof_.set_handler();};   


    
    Node() = delete;

    // forwarding constructor
    template<typename... Args> //This thing also defines copy and move constructors. If a copy constructor is defined, any member can't be (own) a unique_ptr.
    Node(int tag, Args&&... args) : 
      id_(tag),
      geometry::Point<Dim>(std::forward<Args>(args)...)
      {
        //std::cout << "Node Full Forwarding constructor" << std::endl;
      }


    Node(std::size_t tag, std::array<double,Dim>&& coord_list) : 
      id_{tag},
      geometry::Point<Dim>{std::forward<std::array<double,Dim>>(coord_list)}
      {
        //std::cout << "Node Coord Forwarding constructor" << std::endl;
      }; 


    ~Node(){} 

};



#endif