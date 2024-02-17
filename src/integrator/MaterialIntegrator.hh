#ifndef FALL_N_MATERIAL_INTEGRATOR_ABSTRACTION
#define FALL_N_MATERIAL_INTEGRATOR_ABSTRACTION


#include <concepts>
#include <cstddef>
#include <type_traits>



//template<typename T>
//concept MaterialIntegrator = requires(T t){
//  requires std::invocable<T>;
//};



class MaterialIntegrator {
  public:
    MaterialIntegrator() = default;
    ~MaterialIntegrator() = default;

};



#endif // FALL_N_MATERIAL_INTEGRATOR_ABSTRACTION
