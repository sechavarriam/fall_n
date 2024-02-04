#ifndef SMALL_MATH_HH
#define SMALL_MATH_HH

#include <array>
#include <concepts>
#include <cstddef>

#include <utility>

namespace utils{

template<std::size_t dim> requires (dim == 1 || dim == 2 || dim == 3)
inline constexpr auto det(std::array<std::array<double, dim>, dim> A) noexcept
{
  if constexpr (dim == 1){
    return A[0][0];}
  else if constexpr (dim == 2){
    return A[0][0] * A[1][1] - A[0][1] * A[1][0];}
  else if constexpr (dim == 3){
    return A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) 
         - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) 
         + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);}
  else {std::unreachable();}
};

}


#endif // SMALL_MATH_HH