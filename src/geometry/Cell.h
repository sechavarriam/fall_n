#ifndef FALL_N_CELL
#define FALL_N_CELL

#include <array>
#include <cmath>
#include <compare>
#include <cstddef>
#include <utility>

#include "Point.h"
#include "Topology.h"

// Cada uno debe ser un Singleton (Solo debería poderse crear un único objeto de
// esa clase.) Será que si? Porque cada punto de integración debe tener un
// material...

typedef unsigned short ushort;
typedef unsigned int uint;

namespace geometry {


// Helper function to compute the number of points in the reference cell in terms of 
// the two interpolation orders sintaxes allowed.
template <ushort Dim, ushort... InterpolationOrders>
  requires(sizeof...(InterpolationOrders) == Dim ||
           sizeof...(InterpolationOrders) == 1)
consteval std::size_t n_points_(){
    auto constexpr n_params = sizeof...(InterpolationOrders);
    if constexpr (n_params == 1) {
      auto constexpr nx = ((InterpolationOrders + 1)* ...);
      if constexpr (Dim == 0) {
        return 1;
      } else if constexpr (Dim == 1) {
        return nx;
      } else if constexpr (Dim == 2) {
        return nx*nx;
      } else if constexpr (Dim == 3) {
        return nx*nx*nx;
      } else { // Unreachable for C++23 ( TODO: compilation directive)
        static_assert(topology::EmbeddableInSpace<Dim>, "Dimension not supported (yet...)");
      }
    } else if constexpr (n_params == Dim) {
      return ((InterpolationOrders + 1)* ...);
    };
  }



template <ushort Dim, ushort... InterpolationOrders>
  requires(sizeof...(InterpolationOrders) == Dim ||
           sizeof...(InterpolationOrders) == 1)
class Cell { // Reference Cell (Unit Cube) of dimension Dim.
  using Point = geometry::Point<Dim>;

private:  
  

public:
  // Order of the cell (1 for linear, 2 for quadratic, etc.) in each direction
  
  static constexpr uint n_points = n_points_<Dim, InterpolationOrders...>();

  std::array<ushort, Dim> order{InterpolationOrders...};
  
  std::array<Point, n_points> points;
  

  Cell() {
    if constexpr (sizeof...(InterpolationOrders) == 1) {
      order.fill(InterpolationOrders...);
    }
  };

  ~Cell(){};
};

template <unsigned short Dim, unsigned short... Order> class ReferenceCell {};

} // namespace geometry

#endif
