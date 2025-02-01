#ifndef SMALL_MATH_HH
#define SMALL_MATH_HH

#include <array>
#include <concepts>
#include <cstddef>


#ifdef __clang__ 
  #include <format>
  #include <print>
#endif

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

    //std::println( "Jacobian Matrix: {0:> 2.8e} {1:> 2.8e} {2:> 2.8e}\n"
    //              "                 {3:> 2.8e} {4:> 2.8e} {5:> 2.8e}\n"
    //              "                 {6:> 2.8e} {7:> 2.8e} {8:> 2.8e}\n",
    //              A[0][0], A[0][1], A[0][2],
    //              A[1][0], A[1][1], A[1][2],
    //              A[2][0], A[2][1], A[2][2]);

    return A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) 
         - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) 
         + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);}
  else {std::unreachable();}
};

}


#endif // SMALL_MATH_HH