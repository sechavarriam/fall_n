#ifndef FALL_N_SIMPLEX
#define FALL_N_SIMPLEX

#include <array>
#include <utility>

#include "../numerics/Vector.h"

template<unsigned short Dim, unsigned short nPoints = Dim + 1>
class Simplex {
  
  public:
    
    static_assert(Dim>0, "Dimension must be greater than 0");
    static_assert(nPoints>0, "Number of points must be greater than 0");
    static_assert(nPoints<=Dim+1, "Number of points must be less than or equal to Dim+1");
    
    Simplex(){};
    ~Simplex(){};
    
    template<typename... Args>
    Simplex(Args&&... args):points_(std::forward<Args>(args)...){};
        

    std::array<std::array<double,Dim>,nPoints> points_;

};


















#endif