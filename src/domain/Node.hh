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

  private:

    std::size_t id_{} ; 
    std::size_t num_dof_{0}; 

    domain::DoF_Interface dof_;

  public:

    std::integral auto id()     {return id_     ;}
    std::integral auto num_dof(){return num_dof_;}

    void set_id     (const std::size_t& id) noexcept {id_ = id;};
    void set_num_dof(const std::size_t& n ) noexcept {num_dof_ = n;};    

    std::span<std::size_t> dof_index(){
      if (!dof_.dof_handler_) {
        throw std::runtime_error("DoF Handler not set");
      }
      return std::span<std::size_t>(dof_.dof_handler_->dof_index_);
      };

    std::size_t dof_index(std::size_t i){
      if (!dof_.dof_handler_) {
        throw std::runtime_error("DoF Handler not set");
      }
      return dof_.dof_handler_->dof_index_[i];
      };


    void set_dof_interface(std::initializer_list<std::size_t>&& dofs_index)
    {
      dof_.set_index(std::forward<std::initializer_list<std::size_t>>(dofs_index));
      num_dof_ = dofs_index.size();
    };

    void set_dof_interface(){dof_.set_handler();};   
    
    Node() = delete;

    // forwarding constructor
    template<typename... Args> //This thing also defines copy and move constructors. If a copy constructor is defined, any member can't be (own) a unique_ptr.
    Node(int tag, Args&&... args) : 
      id_(tag),
      geometry::Point<Dim>(std::forward<Args>(args)...)
      {}


    Node(std::size_t tag, std::array<double,Dim>&& coord_list) : 
      id_{tag},
      geometry::Point<Dim>{std::forward<std::array<double,Dim>>(coord_list)}
      {}; 


    ~Node(){} 

};



#endif