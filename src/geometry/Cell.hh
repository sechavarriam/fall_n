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


#include <vtkCellType.h>


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


//https://stackoverflow.com/questions/71422709/metafunction-to-check-if-all-parameter-pack-arguments-are-the-same
//template <typename T, typename...U> 
//using is_all_same = std::integral_constant<bool, (... && std::is_same_v<T,U>)>;

template <std::size_t... n> // TODO: Extract to utilities
consteval bool are_equal(){ //check if all n are equal{
  std::array<std::size_t, sizeof...(n)> arr{n...};
  return std::all_of(arr.begin(), arr.end(), [&arr](unsigned int i) { return i == arr[0]; });
};

// CLASS DEFINITION ================================================================================

template <std::size_t... n> // n: Number of nodes per direction.
class LagrangianCell {

  static constexpr auto dimensions = std::array{n...};

  static constexpr std::size_t dim    {sizeof...(n)};
  static constexpr std::size_t num_nodes_{(n*...)};

  using Point = geometry::Point<dim>;
  
  template<ushort... num_nodes_in_each_direction>
  using Basis = interpolation::LagrangeBasis_ND<num_nodes_in_each_direction...>;

public:  
  static constexpr std::array<Point, num_nodes_> reference_nodes{cell_nodes<n...>()};
  
  static constexpr Basis<n...> basis{equally_spaced_coordinates<n>()...}; //n funtors that generate lambdas

  static constexpr unsigned int VTK_cell_type(){
    if constexpr (dim == 1) {
      if      constexpr (dimensions[0] == 2) return VTK_LINE;
      else if constexpr (dimensions[0] == 3) return VTK_QUADRATIC_EDGE;
      else if constexpr (dimensions[0]  > 3) return VTK_LAGRANGE_CURVE; // or could be VTK_HIGHER_ORDER_CURVE
    }
    else if constexpr (dim == 2)
      if constexpr (are_equal<n...>()){
        if      constexpr (dimensions[0] == 2) return VTK_QUAD;
        else if constexpr (dimensions[0] == 3) return VTK_QUADRATIC_QUAD;
        else if constexpr (dimensions[0]  > 3) return VTK_LAGRANGE_QUADRILATERAL; // or could be VTK_HIGHER_ORDER_QUADRILATERAL 
      }
      else return VTK_EMPTY_CELL;
    else if constexpr (dim == 3){
      if constexpr (are_equal<n...>()){
        if      constexpr (dimensions[0] == 2) return VTK_HEXAHEDRON;
        else if constexpr (dimensions[0] == 3) return VTK_TRIQUADRATIC_HEXAHEDRON;
        else if constexpr (dimensions[0]  > 3) return VTK_LAGRANGE_HEXAHEDRON; // or could be VTK_HIGHER_ORDER_HEXAHEDRON 
      }
      else return VTK_EMPTY_CELL;
    } 
    else return VTK_EMPTY_CELL; // unsupported dimension
  }

  static constexpr std::array<std::size_t, num_nodes_> VTK_node_ordering()
  {
    using Array = std::array<std::size_t, num_nodes_>;
    if constexpr (dim == 1) {
      if      constexpr (dimensions[0] == 2) return Array{0, 1};
      else if constexpr (dimensions[0] == 3) return Array{0, 2, 1};
      else if constexpr (dimensions[0]  > 3) return Array{0};
    }
    else if constexpr (dim == 2)
      if constexpr (are_equal<n...>()){
        if      constexpr (dimensions[0] == 2) return Array{0, 1, 3, 2};
        else if constexpr (dimensions[0] == 3) return Array{0, 1, 3, 2, 4, 5, 7, 6};
        else if constexpr (dimensions[0]  > 3) return Array{0};
      }
      else return Array{0};
    else if constexpr (dim == 3){
      if constexpr (are_equal<n...>()){
        if      constexpr (dimensions[0] == 2) return Array{0, 1, 3, 2, 4, 5, 7, 6};
        else if constexpr (dimensions[0] == 3) return Array{0, 2, 8, 6, 18, 20, 26, 24, 1, 5, 7, 3, 19, 23 ,25 ,21 ,9 ,11, 17, 15, 12, 14, 10, 16, 4, 22, 13};
        else if constexpr (dimensions[0]  > 3) return Array{0};
      }
      else return Array{0};
    }
    else return Array{0}; // unsupported dimension    
  };

  // Constructor
  consteval LagrangianCell() = default;
  constexpr ~LagrangianCell() = default;
};


// ==================================================================================================

} // namespace geometry::cell

#endif // FALL_N_REFERENCE_CELL
