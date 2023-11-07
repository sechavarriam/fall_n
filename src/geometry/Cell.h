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
template <ushort Dim, ushort... InterpolantOrder>
  requires(sizeof...(InterpolantOrder) == Dim ||
           sizeof...(InterpolantOrder) == 1)
consteval std::size_t n_nodes_(){
    auto constexpr n_params = sizeof...(InterpolantOrder);
    if constexpr (n_params == 1) {
      auto constexpr n = ((InterpolantOrder + 1)* ...);
      if constexpr (Dim == 0) {return 1;} 
      else if constexpr (Dim == 1) {return n;} 
      else if constexpr (Dim == 2) {return n*n;}
      else if constexpr (Dim == 3) {return n*n*n;} 
      else { // Unreachable for C++23 ( TODO: compilation directive)
        static_assert(Dim<0||~topology::EmbeddableInSpace<Dim>, "Dimension not supported (yet...)");
      }
    } 
    else if constexpr (n_params == Dim) {return ((InterpolantOrder + 1)* ...);};
  }

template <ushort Dim, ushort... QuadratureOrder>
  requires(sizeof...(QuadratureOrder) == Dim ||
           sizeof...(QuadratureOrder) == 1)
consteval std::size_t n_GaussPoints_(){
    auto constexpr n_params = sizeof...(QuadratureOrder);
    if constexpr (n_params == 1) {
      auto constexpr n = ((QuadratureOrder)* ...);
      if constexpr (Dim == 0) {return 1;} 
      else if constexpr (Dim == 1) { return n;}
      else if constexpr (Dim == 2) { return n*n;} 
      else if constexpr (Dim == 3) { return n*n*n;}
      else { // Unreachable for C++23 ( TODO: compilation directive)
        static_assert(Dim<0||~topology::EmbeddableInSpace<Dim>, "Dimension not supported (yet...)");
      }
    } 
    else if constexpr (n_params == Dim) {return ((QuadratureOrder)* ...);
    };
  }


template<ushort Dim, ushort... num_nodes_per_direction> //IntegrationPolicy (IntegrationStrategy)
  requires (sizeof...(num_nodes_per_direction) == Dim ||
            sizeof...(num_nodes_per_direction) == 1)
class Cell{
  using Point = geometry::Point<Dim>;

private:  
  
  std::array<ushort, Dim> nodes_per_direction{num_nodes_per_direction...}; //nx, ny, nz.
  
public:
  static constexpr uint n_nodes = n_nodes_<Dim, num_nodes_per_direction...>();
  std::array<Point , n_nodes> reference_nodes;

  Cell() {
    if constexpr (sizeof...(num_nodes_per_direction) == 1) {
      nodes_per_direction.fill(num_nodes_per_direction...);
    }
  };

  ~Cell(){};
};
// ==================================================================================================

template <unsigned short Dim, unsigned short... Order> class ReferenceCell {};

} // namespace geometry

#endif
