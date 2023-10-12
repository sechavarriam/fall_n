#ifndef FALL_N_REFERENCE_SIMPLEX
#define FALL_N_REFERENCE_SIMPLEX

#include <array>
#include <utility>

#include "Point.h"

#include "../numerics/Vector.h"

template<unsigned short Dim>
class Simplex {

  public:
    static constexpr ushort n_points = Dim+1;
    std::array<Point<Dim>, n_points> vertices_;

    Simplex(){};
    ~Simplex(){};
};

#endif