#ifndef FN_TOPOLOGY_H
#define FN_TOPOLOGY_H


#include <concepts>


//template<unsigned int Dim> static constexpr bool EmbeddableInSpace = Dim<4;

namespace Topology{

template<unsigned short Dim> static constexpr bool EmbeddableInSpace{Dim<4};
template<unsigned short Dim> static constexpr bool InPlane = Dim==2; // Unused Yet 
template<unsigned short Dim> static constexpr bool InSpace = Dim==3; // Unused Yet


template<unsigned int Dim1, unsigned int Dim2=Dim1>
class Dimension{
    static constexpr unsigned int Space    = Dim1;
    static constexpr unsigned int Lagrange = Dim2; 

    //static constexpr double Hausdorff; // For Fractal Objects (Not needed yet)
};
}




#endif







