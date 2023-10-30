#ifndef FALL_N_REFERENCE_SIMPLEX
#define FALL_N_REFERENCE_SIMPLEX

#include <array>
#include <utility>

#include "Point.h"

#include "../numerics/Vector.h"

namespace geometry {
  
template<unsigned short Dim>
class Simplex {
  
  using Point = geometry::Point<Dim>;

  public:
    static constexpr ushort n_points = Dim+1;
    std::array<Point, n_points> vertices_;

    Simplex(){};
    ~Simplex(){};
};

} // namespace geometry
#endif