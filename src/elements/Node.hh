#ifndef FN_NODE
#define FN_NODE

#include <array>

#include <stdexcept>
#include <initializer_list>
#include <utility>
#include <span>

#include "../model/DoF.hh"
#include "../geometry/Point.hh"

template<std::size_t Dim>//, ushort nDoF=Dim> 
class Node : public geometry::Point<Dim>{
 
 using DoF_Interface = domain::DoF_Interface;
 
     
    std::size_t id_{} ;
    
    DoF_Interface dof_;
    
  public:

    std::size_t id()       {return id_     ;}
    std::size_t num_dof()  {return dof_.handler_->num_dof();}

    //std::span<double*> dofs(){return std::span<double*>(dof_.handler_->dofs_);};

    constexpr void set_id     (const std::size_t& id) noexcept {id_ = id;};
    
    //constexpr void set_num_dof(std::size_t n ) noexcept {
    //  dof_.set_num_dof(n);
    //};


    constexpr void set_dof_index(std::size_t i, std::size_t dof_index){
      dof_.handler_->dof_index_[i] = dof_index;
    };

    constexpr void set_dof_index(std::ranges::range auto&& idxs)
    requires std::convertible_to<std::ranges::range_value_t<decltype(idxs)>, std::size_t>{
      dof_.handler_->set_index(idxs);
    };


    std::span<PetscInt> dof_index() const {
      if (!dof_.handler_) throw std::runtime_error("DoF Handler not set");
      return std::span<PetscInt>(dof_.handler_->dof_index_);
      };

    void fix_dof(std::size_t i){
      dof_.handler_->dof_index_[i] *= -1;
    };

    //void set_dof_interface(std::initializer_list<std::size_t>&& dofs_index){
    //  dof_.set_index(std::forward<std::initializer_list<std::size_t>>(dofs_index));
    //};

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