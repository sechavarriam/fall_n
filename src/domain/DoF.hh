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
  std::vector<double*>     dofs_{};     //Alt: GlobalDofs or ModelDofs.

  inline constexpr std::size_t num_dof() const {return dof_index_.size();};
  
  inline constexpr void set_num_dof(std::size_t num_dof){dofs_.resize(num_dof,nullptr);};

  DoF_Handler() = default;

  DoF_Handler(std::initializer_list<std::size_t> dofs){
    dof_index_.reserve(dofs.size());
    std::move(dofs.begin(), dofs.end(), std::back_inserter(dof_index_));
  };
};


class DoF_Interface{
  
  private:
  public:

    inline constexpr std::size_t num(){return handler_->num_dof();};
    //class DoF_Handler;
    std::shared_ptr<DoF_Handler> handler_; // Node copy constructor forbids the use of unique_ptr in the way it is used here.
                                           // Beeng shared_pts has its advantages. It can be used to share the same handler 
                                           // between nodes, that is, manage the same dofs for two or more nodes.                                               
    
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