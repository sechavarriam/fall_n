#ifndef FALL_N_REFERENCE_CELL
#define FALL_N_REFERENCE_CELL

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdio>
//#include <print> //Not yet implemented in Clang.
#include <iostream>
#include <petscsnes.h>
#include <ranges>
#include <tuple>

#include "Point.hh"
#include "Topology.hh"


#include "../utils/index.hh"
#include "../numerics/Interpolation/LagrangeInterpolation.hh"

namespace geometry::cell {

// HELPER FUNCTIONS ================================================================================
static inline constexpr double delta_i(int n ) {
  constexpr double interval_size = 2.0;
  if (n == 1)
    return 0.0;
  return interval_size / (n - 1);
}; // Interval size per direction i.


template <std::size_t... dimensions> 
static inline constexpr std::array<double, sizeof...(dimensions)>  
coordinate_xi(std::array<std::size_t, sizeof...(dimensions)> index_ijk)
{ 
  constexpr std::size_t dim =  sizeof...(dimensions);

  std::array<double,dim> coordinates{dimensions...};

  for(std::size_t position = 0; position < dim; ++position){
    coordinates[position] = - 1 + index_ijk[position]*delta_i(coordinates[position]);
  };

  return coordinates;
};

template <std::size_t Ni> // TODO: Redefine as policy.
static inline constexpr auto equally_spaced_coordinates() 
{
  std::array<double, Ni> coordinates;
  for (std::size_t i = 0; i < Ni; ++i) coordinates[i] = -1 + i * delta_i(Ni);

  return coordinates;
};


template <std::size_t... n>
static inline constexpr Point<sizeof...(n)> node_ijk(std::array<std::size_t,sizeof...(n)> md_array){
  return Point<sizeof...(n)>(coordinate_xi<n ...>(md_array));
};

  
template<std::size_t... n>
consteval std::array<Point<sizeof...(n)>, (n * ...)>cell_nodes(){

  std::array<Point<sizeof...(n)>, (n*...)> nodes;
  for (std::size_t i = 0; i < (n*...); ++i) nodes[i] = node_ijk<n...>( utils::list_2_md_index<n...>(i)); 

  return nodes;
};


// CLASS DEFINITION ================================================================================

template <std::size_t... n> // n: Number of nodes per direction.
class LagrangianCell {

  static constexpr std::size_t dim    {sizeof...(n)};
  static constexpr std::size_t num_nodes_{(n*...)};


  using Point = geometry::Point<dim>;
  
  template<ushort... num_nodes_in_each_direction>
  using Basis = interpolation::LagrangeBasis_ND<num_nodes_in_each_direction...>;

  
public:  
  static constexpr std::array<Point, num_nodes_> reference_nodes{cell_nodes<n...>()};
  
  static constexpr Basis<n...> basis{equally_spaced_coordinates<n>()...}; //n funtors that generate lambdas

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
  consteval LagrangianCell() = default;
    
  constexpr ~LagrangianCell(){};
};
// ==================================================================================================



} // namespace geometry::cell




#endif // FALL_N_REFERENCE_CELL
