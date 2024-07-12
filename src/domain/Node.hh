#ifndef FN_NODE
#define FN_NODE

#include <array>

#include <stdexcept>
#include <initializer_list>
#include <utility>
#include <span>

#include "DoF.hh"
#include "../geometry/Point.hh"

template<std::size_t Dim>//, ushort nDoF=Dim> 
class Node : public geometry::Point<Dim>{
 
 using DoF_Interface = domain::DoF_Interface;
 
     
    std::size_t id_{} ;
    std::size_t num_dof_{}; 
    

    DoF_Interface dof_;
    
  public:

    std::size_t id()       {return id_     ;}
    std::size_t num_dof()  {return num_dof_;}
    //std::size_t num_dof_h(){return dof_.num();}

    std::span<double*> dofs(){return std::span<double*>(dof_.handler_->dofs_);};

    constexpr void set_id     (const std::size_t& id) noexcept {id_ = id;};
    

    constexpr void set_num_dof(std::size_t n ) noexcept {
      num_dof_ = n;
      dof_.set_num_dof(n);
      };

    constexpr void set_dof(std::size_t i, double* p_model_dof){
      dof_.handler_->dofs_[i] = p_model_dof;
    };

    //constexpr void set_dofs(std::ranges::contiguous_range auto&& dofs){
    //  dof_.handler_->set_dofs(dofs);
    //};

    auto dof_index(){
      if (!dof_.handler_) throw std::runtime_error("DoF Handler not set");
      return std::span<std::size_t>(dof_.handler_->dof_index_);
      };


    void set_dof_interface(std::initializer_list<std::size_t>&& dofs_index){
      dof_.set_index(std::forward<std::initializer_list<std::size_t>>(dofs_index));
    };

    void set_dof_interface(){dof_.set_handler();};   
    
    Node() = delete;

    //forwarding constructor
    template<std::floating_point... Args> //This thing also defines copy and move constructors. If a copy constructor is defined, any member can't be (own) a unique_ptr.
    Node(std::size_t tag, Args&&... args) : 
      geometry::Point<Dim>(std::forward<Args>(args)...),
      id_{tag}{}
  

    ~Node(){} 

};



#endif