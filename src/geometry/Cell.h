#ifndef FALL_N_CELL
#define FALL_N_CELL


#include <array>
#include <cmath>
#include <utility>

#include "Point.h"


template <unsigned short Dim>
class Cell {
    public:
    
    static constexpr ushort n_points = std::pow(2,Dim);
    
    std::array<Point<Dim>, n_points> vertices_;

        Cell(){};
        ~Cell(){};


};

#endif
