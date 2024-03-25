#ifndef FN_NODE
#define FN_NODE

#include <array>

#include <stdexcept>
#include <initializer_list>
#include <utility>
#include <span>


#include "DoF.hh"
#include "../geometry/Point.hh"

typedef unsigned short ushort;
typedef unsigned int   uint  ;

template<std::size_t Dim>//, ushort nDoF=Dim> 
class Node : public geometry::Point<Dim>{
 
 using DoF_Interface = domain::DoF_Interface;
 
    std::size_t id_{} ; 
    std::size_t num_dof_{0}; 

    DoF_Interface dof_;

  public:

    std::integral auto id()     {return id_     ;}
    std::integral auto num_dof(){return dof_.handler_->dof_index_.size();}

    constexpr void set_id     (const std::size_t& id) noexcept {id_ = id;};
    constexpr void set_num_dof(const std::size_t& n ) noexcept {
      num_dof_ = n;
      if (!dof_.handler_) dof_.set_handler();
      dof_.handler_->set_num_dof(n);
      };

    auto dof_index(){
      if (!dof_.handler_) throw std::runtime_error("DoF Handler not set");
      return std::span<std::size_t>(dof_.handler_->dof_index_);
      };

    std::integral auto dof_index(std::size_t i){
      if (!dof_.handler_) throw std::runtime_error("DoF Handler not set");
      return dof_.handler_->dof_index_[i];
      };


    void set_dof_interface(std::initializer_list<std::size_t>&& dofs_index){
      dof_.set_index(std::forward<std::initializer_list<std::size_t>>(dofs_index));
    };

    void set_dof_interface(){dof_.set_handler();};   
    
    Node() = delete;

    //forwarding constructor
    template<std::floating_point... Args> //This thing also defines copy and move constructors. If a copy constructor is defined, any member can't be (own) a unique_ptr.
    Node(int tag, Args&&... args) : 
      id_(tag),
      geometry::Point<Dim>(std::forward<Args>(args)...)
      {}


    //Node(std::size_t tag, std::initializer_list<double>&& coord_list) : 
    //  id_{tag},
    //  geometry::Point<Dim>{std::forward<std::initializer_list<double>>(coord_list)}
    //  {
    //    std::cout << "Hiiii" << id_ << std::endl;
    //  };

    //Node(std::size_t tag, std::array<double,Dim>&& coord_list) : 
    //  id_{tag},
    //  geometry::Point<Dim>{std::forward<std::array<double,Dim>>(coord_list)}
    //  {
    //    std::cout << "Hi" << id_ << std::endl;
    //  }; 


    ~Node(){} 

};



#endif