#ifndef FALL_N_CELL
#define FALL_N_CELL

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdio>
//#include <print> //Not yet implemented in Clang.
#include <iostream>
#include <ranges>
#include <tuple>

#include "Point.h"
#include "Topology.h"

#include "../numerics/Interpolation/LagrangeInterpolation.h"


// Cada uno debe ser un Singleton (Solo debería poderse crear un único objeto de
// esa clase.) Será que si? Porque cada punto de integración debe tener un
// material...

typedef unsigned short ushort;
typedef unsigned int uint;

namespace geometry::cell {

// Helper function to compute the number of points in the reference cell in
// terms of the two interpolation orders sintaxes allowed.

template <ushort Dim, ushort... num_node_per_direction> //DEPRECATED.
  requires(sizeof...(num_node_per_direction) == Dim ||
           sizeof...(num_node_per_direction) == 1)
static consteval std::size_t n_nodes_() {
    auto constexpr n_params = sizeof...(num_node_per_direction);
    if constexpr (n_params == 1) 
    {
      auto constexpr n = ((num_node_per_direction) * ...); // Es necesario reducirlo así sea solo un parametro.
      if constexpr (Dim == 0) {return 1;} 
      else if constexpr (Dim == 1) {return n;} 
      else if constexpr (Dim == 2) {return n*n;} 
      else if constexpr (Dim == 3) {return n*n*n;} 
      else { // Unreachable for C++23 ( TODO: compilation directive)
        static_assert(Dim < 0 || ~topology::EmbeddableInSpace<Dim>,
                      "Dimension not supported (yet...)");
      }
    } 
    else if constexpr (n_params == Dim) 
    {
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
static inline constexpr std::array<double, sizeof...(dimensions)>  
coordinate_xi(std::array<std::size_t, sizeof...(dimensions)> index_ijk)
{ 
  constexpr std::size_t Dim =  sizeof...(dimensions);

  std::array<double,Dim> coordinates{dimensions...};

  for(int position = 0; position<Dim; ++position){
    coordinates[position] = - 1 + index_ijk[position]*delta_i(coordinates[position]);
  };

  return coordinates;
};



template <ushort Ni> // TODO: Redefine as policy.
static inline constexpr auto equally_spaced_coordinates() 
{
  std::array<double, Ni> coordinates;
  for (auto i = 0; i < Ni; ++i) {
    coordinates[i] = -1 + i * delta_i(Ni);
  }
  return coordinates;
};



template <ushort... n>
static inline constexpr Point<sizeof...(n)> 
node_ijk(std::array<std::size_t,sizeof...(n)> md_array) {
  return Point<sizeof...(n)>(coordinate_xi<n ...>(md_array));
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
}; // NOTE: Untested.


template <ushort... N> //<Nx, Ny, Nz ,...>
static inline constexpr std::array<std::size_t, sizeof...(N)> list_2_md_index(const int index) {
  using IndexTuple = std::array<std::size_t, sizeof...(N)>;

  constexpr std::size_t array_dimension = sizeof...(N); 
  
  IndexTuple array_limits{N...};
  IndexTuple md_index; // to return.

  std::integral auto num_positions = std::ranges::fold_left(array_limits, 1, std::multiplies<int>());

  try { // TODO: Improve this error handling.
    if (index >= num_positions) {
      throw index >= num_positions;
    }
  } catch (bool out_of_range) {
    std::cout << "Index out of range. Returning zero array." << std::endl;
    return std::array<std::size_t, sizeof...(N)>{0};
  }
  
  std::integral auto divisor = num_positions/int(array_limits.back()); //TODO: Check if this is integer division or use concepts  
  std::integral auto I       = index;

  for (auto n = array_dimension-1; n>0; --n) 
  {
    md_index[n] = I/divisor; //TODO: Check if this is integer division or use concepts
    I %= divisor;
    divisor /= array_limits[n-1]; 
  }
  md_index[0] = I;

  return md_index;
}; // Untested. //TODO: Check list_2_md_index and md_index_2_list are inverse functions.
                //TODO: Check index is in the range of the cell.
  
template<ushort... n>
consteval std::array<Point<sizeof...(n)>, (n * ...)>cell_nodes(){
  constexpr std::size_t num_nodes = (n * ...);

  static constexpr std::size_t Dim = sizeof...(n);

  std::array<Point<Dim>, num_nodes> nodes;

  for (auto i = 0; i < num_nodes; ++i) // TODO: Convert loop to range. 
  {
    nodes[i] = node_ijk<n...>( list_2_md_index<n...>(i)); 
  }
  return nodes;
};

//inline constexpr std::array<double, 1> xi_coordinates() { return {1.0}; };
// ==================================================================================================

template <ushort... n> // n: Number of nodes per direction.
class LagrangianCell {

  static constexpr std::size_t dim = sizeof...(n);

  using Point = geometry::Point<dim>;

  template<ushort... num_nodes_in_each_direction>
  using Basis = interpolation::LagrangeBasis_ND<num_nodes_in_each_direction...>;

private:
  
public:

  static constexpr std::array<ushort, dim> nodes_per_direction{n...}; // nx, ny, nz.

  static constexpr uint n_nodes{(n*...)};

  static constexpr std::array<Point, n_nodes> reference_nodes{cell_nodes<dim, n...>()};


public:

  static constexpr Basis<n...> basis{equally_spaced_coordinates<n>()...}; //n funtors that generate lambdas



  static inline constexpr auto evaluate_basis_function(const std::array<double,dim>& X) noexcept
  {
    return std::apply([&X](auto&&... args) 
    { 
      return std::make_tuple(args(X)...); 
    },
    basis.L);
  };



  static constexpr void print_node_coords() noexcept
  {
    for (auto node : reference_nodes){
      for (auto j: node.coord()){
        std::cout << j << " ";
      };
      printf("\n");
    }
  };

  // Constructor
  consteval LagrangianCell() noexcept {
    if constexpr (sizeof...(n) == 1) {
      nodes_per_direction.fill(n...);
    }
    
  };
  constexpr ~LagrangianCell(){};
};
// ==================================================================================================

//template <unsigned short Dim, unsigned short... Order> class ReferenceCell {};

} // namespace geometry::cell

#endif
