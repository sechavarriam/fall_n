#ifndef FALL_N_DEGGREE_OF_FREEDOM_CONTAINER
#define FALL_N_DEGGREE_OF_FREEDOM_CONTAINER

#include <array>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>
#include <iostream>
#include <optional>
#include <span>
#include <ranges>
// #include "../numerics/Vector.h"

namespace domain {

class DoF_Handler {
  public:

  std::vector<std::size_t> dof_index_{};
  std::vector<double*>     dofs_{};     //Alt: GlobalDofs or ModelDofs. //Esto no permite un maneho facil de los dofs en un vector de PETSc.

  constexpr std::size_t num_dof() const {return dof_index_.size();};

  constexpr void set_size(std::size_t num_dof){dofs_.resize(num_dof,nullptr);};
  
  constexpr void set_dofs(std::ranges::range auto&& dofs){
    dofs_.reserve(std::ranges::size(dofs));
    std::ranges::move(dofs, std::back_inserter(dofs_));
  };


  constexpr void set_index(std::ranges::range auto&& idxs) //requires dofs are integral types.
  requires std::integral<std::ranges::range_value_t<decltype(idxs)>>{
    dof_index_.reserve(std::ranges::size(idxs));
    std::ranges::move(idxs, std::back_inserter(dof_index_));
  };
  // PETSc NOTE:
  // Negative indices may be passed in idxm and idxn, these rows and columns are simply ignored. 
  // This allows easily inserting element stiffness matrices with homogeneous Dirichlet boundary 
  // conditions that you don’t want represented in the matrix.

  DoF_Handler() = default;

};


class DoF_Interface{
  
  private:
  public:

    std::shared_ptr<DoF_Handler> handler_;// Node copy constructor forbids the use of unique_ptr in the way it is used here.
                                          // Beeng shared_pts has its advantages. It can be used to share the same handler 
                                          // between nodes, that is, manage the same dofs for two or more nodes.                                               

    constexpr void set_num_dof(std::size_t num_dof){handler_->set_size(num_dof);};
    
    constexpr void set_dofs(std::ranges::range auto&& dofs){
      //if(!handler_) set_handler();
      handler_->set_dofs(dofs);
    };


    //auto handler(){return handler_;};
    //class DoF_Handler;



    
    void set_handler(){ handler_ = std::make_shared<DoF_Handler>();};

    void set_index(std::ranges::range auto&& dofs){
      if(!handler_) set_handler();
      
      handler_->dof_index_.reserve(std::ranges::size(dofs));
      std::ranges::move(dofs, std::back_inserter(handler_->dof_index_));
    };

    void set_index(std::initializer_list<std::size_t>&& dofs){
      if(!handler_) set_handler();
      
      handler_->dof_index_.reserve(dofs.size());
      std::move(dofs.begin(), dofs.end(), std::back_inserter(handler_->dof_index_));
    };


    DoF_Interface(){
      handler_ = std::make_shared<DoF_Handler>();
    }
    
    ~DoF_Interface(){};

};












} // namespace domain

#endif