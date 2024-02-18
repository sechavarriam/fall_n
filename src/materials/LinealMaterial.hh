
#ifndef FALL_N_LINEAL_MATERIAL
#define FALL_N_LINEAL_MATERIAL


#include <cstddef>
#include <type_traits>
#include <concept>

namespace material{

//template<typename M>
//concept MaterialPolicy = requires(M m){
//    {m.stress()} -> std::same_as<double>;
//    {m.strain()} -> std::same_as<double>;
//    {m.elastic_modulus()} -> std::same_as<double>;
//    {m.poisson_ratio()} -> std::same_as<double>;
//    {m.yield_stress()} -> std::same_as<double>;
//    {m.hardening_modulus()} -> std::same_as<double>;
//    {m.strain_rate()} -> std::same_as<double>;
//    {m.stress_rate()} -> std::same_as<double>;
//    {m.temperature} -> std::same_as<double>;
//    {m.temperature_rate} -> std::same_as<double>;
//};

//template <typename T>
//concept StoragePolicy = requires(T t){
//    {t.id()} -> std::same_as<std::size_t>;
//    {t.name()} -> std::same_as<std::string>;
//    {t.description()} -> std::same_as<std::string>;
//};


//Some concept
template<typename MaterialPolicy> //Continuum, Uniaxial, Plane, etc. 
class LinealMaterial{
    
    MaterialPolicy policy_{}; 

    std::size_t id_{0};
    
    



};


}



#endif // FALL_N_LINEAL_MATERIAL