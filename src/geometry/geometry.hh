#ifndef FALL_N_GEOMETRY_CONCEPTS
#define FALL_N_GEOMETRY_CONCEPTS

#include "Topology.hh"
#include <concepts>

namespace geometry{

//template <typename Entity>
//concept Point = requires(Entity e){
//    {e.coordinates()} -> std::convertible_to<double>;
//    {e.coordinates}   -> std::convertible_to<double>;
//};

template<typename Entity>
concept Measurable = requires(Entity e){
    {e.measure()} -> std::convertible_to<double>;
};





}



#endif // FALL_N_GEOMETRY_CONCEPTS