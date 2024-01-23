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
// #include "../numerics/Vector.h"

namespace domain {


class DoF_Handler {
  public:

  std::vector<std::size_t> dof_index_{}; 

  DoF_Handler() = default;

  DoF_Handler(std::initializer_list<std::size_t> dofs){
    dof_index_.reserve(dofs.size());
    std::move(dofs.begin(), dofs.end(), std::back_inserter(dof_index_));
  };
};


class DoF_Interfase{
  
  private:
  public:
    //class DoF_Handler;
    std::shared_ptr<DoF_Handler> dof_handler_{nullptr}; // It should be a raw pointer to avoid smart pointer overhead (?).  
                                                // Node copy constructor forbids the use of unique_ptr in the way it is used here.
                                                // Beeng shared_pts has its advantages. It can be used to share the same handler 
                                                // between nodes, that is, manage the same dofs for two or more nodes.                                               
    
    void set_handler(){
      dof_handler_ = std::make_shared<DoF_Handler>();
    };

    void set_index(std::initializer_list<std::size_t>&& dofs){
      if(!dof_handler_) set_handler();
      
      dof_handler_->dof_index_.reserve(dofs.size());
      std::move(dofs.begin(), dofs.end(), std::back_inserter(dof_handler_->dof_index_));

    };


    DoF_Interfase() = default;
    ~DoF_Interfase(){};

};












} // namespace domain

#endif