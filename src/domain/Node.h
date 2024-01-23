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
     // using Handler = domain::DoF_Handler;
    // 1. Create new handler using initializer list
      dof_.set_index(std::forward<std::initializer_list<std::size_t>>(dofs_index));
    //     // 2. Direct the interfase to the new handler (set_handler)
    //     dof_.set_handler<Handler>(*handler);
    //     // 3. 
    };   

    //  static_cast<domain::DoF_Handler<Dim>*>(dof_.dof_handler())->set_dof_index(dofs);
    //};
    
    Node() = delete;

    // forwarding constructor
    template<typename... Args> //This thing also defines copy and move constructors. If a copy constructor is defined, any member can be a unique_ptr.
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
        //std::cout << "Node Forwarding constructor" << std::endl;
      }; 


    ////Copy constructor and assignment operator.
    //Node(const Node& other) : 
    //  id_(other.id_),
    //  geometry::Point<Dim>(other)
    //  {
    //    std::cout << "Node Copy constructor" << std::endl;
    //  }
    //Node& operator=(const Node& other) {

    //  id_ = other.id_;
    //  geometry::Point<Dim>::operator=(other);
    //  return *this; 
    //};

    //Move constructor and assignment operator.
    //Node(Node&& other) noexcept : 
    //  id_(std::move(other.id_)),
    //  geometry::Point<Dim>(std::move(other))
    //  {
    //    std::cout << "Node Move constructor" << std::endl;
    //  }

    //Node& operator=(Node&& other) noexcept {
    //  id_ = std::move(other.id_);
    //  geometry::Point<Dim>::operator=(std::move(other));
    //  return *this; 
    //};

    //Destructor.

    ~Node(){} 

};



#endif