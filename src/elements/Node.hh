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
 
    std::size_t id_{}     ;
    std::size_t num_dof_{0};
    
    DoF_Interface dof_;
    
  public:

    static constexpr std::size_t dim = Dim;

    std::optional<PetscInt> sieve_id; // Optional sieve id for the node (vertex) inside DMPlex Mesh

    std::size_t id()      const noexcept {return id_     ;}
    std::size_t num_dof() const noexcept {return num_dof_;}

    //std::span<double*> dofs(){return std::span<double*>(dof_.handler_->dofs_);};

    constexpr void set_sieve_id(PetscInt id){sieve_id = id;};
    constexpr void set_id     (const std::size_t& id) noexcept {id_ = id;};
    
    
    void set_num_dof(std::size_t n) noexcept {
      num_dof_ = n;
      dof_.set_handler();
    };

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




    template<std::floating_point... Args>
    Node(std::size_t tag, Args... args) : 
    geometry::Point<Dim>(std::array<double, Dim>{args...}),
    id_{tag}{}

    //forwarding constructor
    template<std::floating_point... Args> //This thing also defines copy and move constructors. If a copy constructor is defined, any member can't be (own) a unique_ptr.
    Node(std::size_t tag, Args&&... args) : 
      geometry::Point<Dim>(std::forward<Args>(args)...),
      id_{tag}{}
  

    ~Node(){} 

};



#endif