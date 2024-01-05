#ifndef FALL_N_CELL
#define FALL_N_CELL

#include <array>
#include <cstddef>

#include "Point.h"
#include "Topology.h"

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
    if constexpr (n_params == 1) {
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

//template <ushort n> 
//static inline constexpr double xi(auto i) {
//  return -1.0 + i*delta_i(n);
//}; // Coordinate of the i-th node in the reference cell.

template <ushort... dimensions> 
static inline constexpr 
std::array<double, sizeof...(dimensions)> 
xi(std::array<double, sizeof...(dimensions)> index)
{ 
  constexpr std::size_t Dim =  sizeof...(dimensions);
  std::array<double,Dim> coordinates{dimensions...};

  for(int position = 0; position<Dim; ++position){
    coordinates[position] = -1 + position*delta_i(coordinates[position]);
  };

  return coordinates;
};

//template <ushort Dim, ushort... n>
//static inline constexpr Point<Dim> node_ijk(auto... index) {
//  static_assert(sizeof...(n) == sizeof...(index),
//                "Number of indices must match the number of template "
//                "parameters (minus one)");
//  static_assert(sizeof...(index) == Dim,
//                "Number of indices must match the dimension of the cell");
//  return Point<Dim>( xi<n>(index)... );
//};

template <ushort Dim, ushort... n>
static inline constexpr Point<Dim> node_ijk(std::array<uint,Dim> ijk) {
  std::array<double,Dim> coordinates;
  return Point<Dim>(xi<n ...>(coordinates));
};


template<ushort Dim, ushort... n> requires(sizeof...(n) == Dim) // Dim se debe poder extraer del numero de parámetros.
static constexpr inline std::array<uint, Dim> index_2_ijk(auto list_index){
  constexpr std::array<uint, Dim> nx_ny_nz{n...}; //Number of nodes per direction.
  
  std::array<uint, Dim> ijk;
  ijk[0] = 0; //nx_ny_nz[0]%list_index==0?  nx_ny_nz[0] : list_index%nx_ny_nz[0];
  if constexpr(Dim>1){
    ijk[1] = 0 ;//(list_index-ijk[0])/nx_ny_nz[0] + 1;
    if constexpr(Dim>2){
      ijk[2] = 0;
    };
  };
  return ijk;
}

template<ushort Dim, ushort... n> requires(sizeof...(n) == Dim)
consteval std::array<Point<Dim>, n_nodes_<Dim, n...>()> cell_nodes(){
  constexpr std::size_t num_nodes = n_nodes_<Dim, n...>();

  std::array<Point<Dim>, num_nodes> nodes;

  for (auto i = 0; i < num_nodes; ++i) 
  {
    nodes[i] = node_ijk<Dim, n...>( index_2_ijk<Dim,n...>(i));
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
