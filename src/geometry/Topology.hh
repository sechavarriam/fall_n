#ifndef FN_TOPOLOGY_H
#define FN_TOPOLOGY_H

#include <concepts>
#include <cstddef>

namespace topology{ // Topology related concepts and definitions

template<std::size_t Dim> static constexpr bool EmbeddableInSpace{Dim > 0 && Dim<4};
template<std::size_t Dim> static constexpr bool InPlane = Dim==2; // Unused Yet 
template<std::size_t Dim> static constexpr bool InSpace = Dim==3; // Unused Yet

template<std::size_t Dim1, std::size_t Dim2=Dim1>
class Dimension{
    static constexpr std::size_t Space    = Dim1;
    static constexpr std::size_t Lagrange = Dim2; 

    //static constexpr double Hausdorff; // For Fractal Objects (Not needed yet)
};
}




#endif







