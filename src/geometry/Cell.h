#ifndef FALL_N_CELL
#define FALL_N_CELL

#include <array>
#include <cstddef>

#include "Point.h"
#include "Topology.h"

#include <ranges>
#include <algorithm>
#include <numeric>

// Cada uno debe ser un Singleton (Solo debería poderse crear un único objeto de
// esa clase.) Será que si? Porque cada punto de integración debe tener un
// material...

typedef unsigned short ushort;
typedef unsigned int uint;

namespace geometry::cell {

// Helper function to compute the number of points in the reference cell in
// terms of the two interpolation orders sintaxes allowed.

template <ushort Dim, ushort... num_node_per_direction>
  requires(sizeof...(num_node_per_direction) == Dim ||
           sizeof...(num_node_per_direction) == 1)
static consteval std::size_t n_nodes_() {
    auto constexpr n_params = sizeof...(num_node_per_direction);
    if constexpr (n_params == 1) 
    {
      auto constexpr n = ((num_node_per_direction) * ...); // Es necesario reducirlo así sea solo un parametro.
      if constexpr (Dim == 0) {
        return 1;
      } 
      else if constexpr (Dim == 1) {
        return n;
      } 
      else if constexpr (Dim == 2) {
        return n * n;
      } 
      else if constexpr (Dim == 3) {
        return n * n * n;
      } 
      else { // Unreachable for C++23 ( TODO: compilation directive)
        static_assert(Dim < 0 || ~topology::EmbeddableInSpace<Dim>,
                      "Dimension not supported (yet...)");
      }
    } else if constexpr (n_params == Dim) {
      return ((num_node_per_direction) * ...);
    };
}

static inline constexpr double delta_i(int n ) {
  constexpr double interval_size = 2.0;
  if (n == 1)
    return 0.0;
  return interval_size / (n - 1);
}; // Interval size per direction i.


template <ushort... dimensions> 
static inline constexpr std::array<double, sizeof...(dimensions)>  xi(std::array<std::size_t, sizeof...(dimensions)> index)
{ 
  constexpr std::size_t Dim =  sizeof...(dimensions);
  std::array<double,Dim> coordinates{dimensions...};

  for(int position = 0; position<Dim; ++position){
    coordinates[position] = -1 + position*delta_i(coordinates[position]);
  };

  return coordinates;
};


template <ushort Dim, ushort... n>
static inline constexpr Point<Dim> node_ijk(std::array<std::size_t,Dim> md_array) {
  return Point<Dim>(xi<n ...>(md_array));
};


template <ushort... N> // TODO: Optimize using ranges and fold expressions.
static inline constexpr std::size_t md_index_2_list(auto... ijk) 
requires(sizeof...(N) == sizeof...(ijk))
{
  constexpr std::size_t array_dimension = sizeof...(N);

  constexpr std::array<std::size_t, array_dimension> array_limits{N...  };
  constexpr std::array<std::size_t, array_dimension> md_index    {ijk...}; 

  constexpr std::size_t index = 0;
  auto n = md_index[0];

  for (auto i = 1; i < array_dimension; ++i) {
    index = md_index[i] * n;
    n *= array_limits[i];
  }
  return index;
}; // NOTE: Untested and unused yet.


template <uint... N> //<Nx, Ny, Nz ,...>
static inline constexpr std::array<std::size_t, sizeof...(N)> list_2_md_index(const int index) {
  using IndexTuple = std::array<std::size_t, sizeof...(N)>;

  constexpr std::size_t array_dimension = sizeof...(N); 
  
  IndexTuple array_limits{N...};
  IndexTuple md_index; // to return.

  std::integral auto num_positions = std::ranges::fold_left(array_limits, 1, std::multiplies<int>());
  std::integral auto divisor = num_positions/int(array_limits.back()); //TODO: Check if this is integer division or use concepts  

  std::integral auto I = index;

  for (auto n = array_dimension-1; n>0; --n) {
    md_index[n] = I/divisor; //TODO: Check if this is integer division or use concepts
    I %= divisor;
    divisor /= array_limits[n-1]; 
  }
  md_index[0] = I;

  return md_index;
}; // Untested. //TODO: Check list_2_md_index and md_index_2_list are inverse functions.
                //TODO: Check index is in the range of the cell.
  
template<ushort Dim, ushort... n> requires(sizeof...(n) == Dim)
consteval std::array<Point<Dim>, n_nodes_<Dim, n...>()> cell_nodes(){
  constexpr std::size_t num_nodes = n_nodes_<Dim, n...>();

  std::array<Point<Dim>, num_nodes> nodes;

  for (auto i = 0; i < num_nodes; ++i) // TODO: Convert loop to range. 
  {
    nodes[i] = node_ijk<Dim, n...>( list_2_md_index<n...>(i)); 
  }
  return nodes;
};



// ==================================================================================================

template <ushort Dim, ushort... n> // n: Number of nodes per direction.
  requires(sizeof...(n) == Dim || sizeof...(n) == 1)
class Cell { // (Lagrangian Cell?)
  using Point = geometry::Point<Dim>;

private:
  static constexpr std::array<ushort, Dim> nodes_per_direction{n...}; // nx, ny, nz.

public:
  static constexpr uint n_nodes = n_nodes_<Dim, n...>();
  static constexpr std::array<Point, n_nodes> reference_nodes{cell_nodes<Dim, n...>()};

  consteval Cell() {
    if constexpr (sizeof...(n) == 1) {
      nodes_per_direction.fill(n...);
    }
    
  };
  constexpr ~Cell(){};
};
// ==================================================================================================

template <unsigned short Dim, unsigned short... Order> class ReferenceCell {};

} // namespace geometry::cell

#endif
