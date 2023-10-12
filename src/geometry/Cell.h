#ifndef FALL_N_CELL
#define FALL_N_CELL


#include <array>
#include <utility>

#include "../numerics/Vector.h"


template <unsigned short Dim, unsigned short nPoints>
class Cell {
    public:

        static_assert(Dim>0, "Dimension must be greater than 0");
        static_assert(nPoints>0, "Number of points must be greater than 0");
        static_assert(nPoints<=Dim+1, "Number of points must be less than or equal to Dim+1");

        Cell(){};
        ~Cell(){};

        template<typename... Args>
        Cell(Args&&... args):points_(std::forward<Args>(args)...){};



        std::array<std::array<double,Dim>,nPoints> points_;

};

#endif
