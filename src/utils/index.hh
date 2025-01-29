#ifndef FALL_N_INDEXING_UTILITIES
#define FALL_N_INDEXING_UTILITIES

#include <iostream>
#include <array>
#include <functional>
#include <ranges>

namespace utils
{



template <ushort... N> // TODO: Optimize using ranges and fold expressions.
static inline constexpr std::size_t md_index_2_list(auto... ijk) 
requires(sizeof...(N) == sizeof...(ijk))
{
  constexpr std::size_t dim = sizeof...(N);

  std::array<std::size_t, dim> array_limits{N...  };
  std::array<std::size_t, dim> md_index    {ijk...}; 
  
  auto n = md_index[0];

  //Row major order
  for (std::size_t i = 1; i < dim; ++i) {
    n *= array_limits[i];
    n += md_index[i];
  }

  ////Column major order
  //for (std::size_t i = 1; i < dim; ++i) {
  //  n += md_index[i] * array_limits[i];
  //}
  
  return n;
}; 


template <std::size_t... N> //<Nx, Ny, Nz ,...> 
static inline constexpr std::array<std::size_t, sizeof...(N)>
list_2_md_index(const std::size_t index) {
  
  using IndexTuple = std::array<std::size_t, sizeof...(N)>;

  constexpr std::size_t dim = sizeof...(N);

  IndexTuple array_limits{N...};
  IndexTuple md_index; // to return.

  std::size_t num_positions =
      std::ranges::fold_left(array_limits, 1, std::multiplies<int>());

  try { // TODO: Improve this error handling.
    if (index >= num_positions) {
      throw index >= num_positions;
    }
  } catch (bool out_of_range) {
    std::cout << "Index out of range. Returning zero array." << std::endl;
    return std::array<std::size_t, sizeof...(N)>{0};
  }

  std::integral auto divisor = num_positions/int(array_limits.back()); // TODO: Check if this is integer division or
                                                                       // use concepts
  std::integral auto I = index;

  for (auto n = dim - 1; n > 0; --n) {
    md_index[n] =
        I / divisor; // TODO: Check if this is integer division or use concepts
    I %= divisor;
    divisor /= array_limits[n - 1];
  }
  md_index[0] = I;

  return md_index;
}; // TODO:  IN CELL TOO (REMOVE BOTH AND MOVE TO UTILS)


}














#endif//FALL_N_INDEXING_UTILITIES