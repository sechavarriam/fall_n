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


// https://stackoverflow.com/questions/39288891/why-is-shared-ptrvoid-legal-while-unique-ptrvoid-is-ill-formed 
//using unique_void_ptr = std::unique_ptr<void, void(*)(void const*)>;
//
//template<typename T>
//auto unique_void(T* ptr) -> unique_void_ptr
//{
//    return unique_void_ptr(ptr, [](void const* data) {
//         T const* p = static_cast<T const*>(data);
//         delete p;
//    });
//}
//
//template<typename T, typename... Args>
//auto make_unique_void(Args&&... args)
//{
//     return unique_void(new T(std::forward<Args>(args)...));
//}
// ============================================================================================================


namespace domain {

//template <std::size_t nDoF> 
class DoF_Handler {

  std::vector<std::size_t> dof_index_{}; //veiw?
public:

  void set_dof_index(std::initializer_list<std::size_t> dofs){
    std::move(dofs.begin(), dofs.end(), dof_index_.begin());
  };

  DoF_Handler() = default;

  DoF_Handler(std::initializer_list<std::size_t> dofs){
    dof_index_.reserve(dofs.size());
    std::move(dofs.begin(), dofs.end(), dof_index_.begin());
  };
};


class DoF_Interfase{
  
  private:
    //class DoF_Handler;
    std::shared_ptr<DoF_Handler> dof_handler_ ;

  public:

    //void set_handler(DoF_Handler&& handler){
    //  dof_handler_ = std::make_shared<DoF_Handler>(std::forward<DoF_Handler>(handler));
    //};

    void set_handler(std::initializer_list<std::size_t>&& dofs){
      dof_handler_ = std::make_shared<DoF_Handler>(std::forward<std::initializer_list<std::size_t>>(dofs));
    };


    DoF_Interfase() = default;
    ~DoF_Interfase(){};

};












} // namespace domain

#endif