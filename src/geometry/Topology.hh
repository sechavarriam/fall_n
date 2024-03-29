#ifndef FN_TOPOLOGY_H
#define FN_TOPOLOGY_H


#include <concepts>
#include <cstddef>

typedef unsigned short ushort;
typedef unsigned int   uint  ;

namespace topology{ // Topology related concepts and definitions

template<std::size_t Dim> static constexpr bool EmbeddableInSpace{Dim > 0 && Dim<4};
template<std::size_t Dim> static constexpr bool InPlane = Dim==2; // Unused Yet 
template<std::size_t Dim> static constexpr bool InSpace = Dim==3; // Unused Yet



template<unsigned int Dim1, unsigned int Dim2=Dim1>
class Dimension{
    static constexpr uint Space    = Dim1;
    static constexpr uint Lagrange = Dim2; 

    //static constexpr double Hausdorff; // For Fractal Objects (Not needed yet)
};
}




#endif







