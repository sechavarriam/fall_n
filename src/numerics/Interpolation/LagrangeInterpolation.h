
#ifndef FN_LAGRANGE_INTERPOLATION_H
#define FN_LAGRANGE_INTERPOLATION_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <numeric>
#include <ranges>

#include <concepts>
#include <functional>
#include <tuple>
#include <utility>

// #include "../../utils/constexpr_function.h"

namespace interpolation {

template <unsigned short nPoints> class LagrangeBasis_1D { // constexpr funtor

  std::array<double, nPoints> xPoints{};

public:
  std::size_t size() const noexcept { return nPoints; };

  constexpr auto operator[](std::integral auto i) const noexcept {
    return [&, i](std::floating_point auto x) {
      double L_i = 1.0;
      for (auto j = 0; j < nPoints; ++j) {
        (j != i) ? L_i *= (x - xPoints[j]) / (xPoints[i] - xPoints[j])
                 : L_i *= 1.0;
      }
      return L_i;
    };
  };

  consteval LagrangeBasis_1D(
      const std::array<double, nPoints> &xCoordinates) noexcept
      : xPoints{xCoordinates} {};

  constexpr ~LagrangeBasis_1D(){};
};

template <unsigned short nPoints> class LagrangeInterpolator_1D {

  using Array = std::array<double, nPoints>;
  using Basis = LagrangeBasis_1D<nPoints>;

  Basis L{}; // Owing or reference?
  Array fValues{};

public:
  constexpr double operator()(double x) const noexcept {
    auto value = 0;

    for (auto i = 0; i < nPoints; ++i) {
      value += fValues[i] * L[i](x);
    }
    return value;
  };

  consteval LagrangeInterpolator_1D(const Array &xPoints_,
                                    const Array &yValues_) noexcept
      : L{std::forward<const Array &>(xPoints_)}, fValues{yValues_} {};

  consteval LagrangeInterpolator_1D(const Basis &basis_,
                                    const Array &yValues_) noexcept
      : L{basis_}, fValues{yValues_} {};

  constexpr ~LagrangeInterpolator_1D(){};
};

template <
    class Tuple, class
    F> // From
       // https://www.fluentcpp.com/2021/03/05/stdindex_sequence-and-its-improvement-in-c20/
       // TODO: Move to utils folder and namespace.
       constexpr decltype(auto) for_each(Tuple &&tuple, F &&f) {
  return []<std::size_t... I>(Tuple &&tuple, F &&f, std::index_sequence<I...>) {
    (f(std::get<I>(tuple)), ...);
    return f;
  } // End of lambda
  ( /*inmediate invocation (use std::invoke to clarify?)*/
   std::forward<Tuple>(tuple), std::forward<F>(f),
   std::make_index_sequence<
       std::tuple_size<std::remove_reference_t<Tuple>>::value>{});
}

template <class Tuple, class Array,
          class F> // TODO: requires size of tuple = size of array
constexpr decltype(auto) for_each_tuple_pair(Tuple &&tuple, Array &&array,
                                             F &&f) {
  return []<std::size_t... I>(Tuple &&tuple, Array &&array, F &&f,
                              std::index_sequence<I...>) {
    (f(std::get<I>(tuple), std::get<I>(array)), ...);
    return f;
  }(std::forward<Tuple>(tuple), std::forward<Array>(array), std::forward<F>(f),
         std::make_index_sequence<
             std::tuple_size<std::remove_reference_t<Tuple>>::value>{});
}

template <unsigned short... Ni> class LagrangeBasis_ND {

  template <unsigned short n> using Array = std::array<double, n>;

  template <unsigned short n> using Basis = interpolation::LagrangeBasis_1D<n>;

  static constexpr auto dim = sizeof...(Ni);

public:
  std::tuple<Array<Ni>...> coordinates_i{};
  std::tuple<Basis<Ni>...> L{};

  consteval LagrangeBasis_ND(const Array<Ni> &...xCoordinates) noexcept
      : coordinates_i{xCoordinates...}, L{Basis<Ni>{xCoordinates}...} {};

  constexpr ~LagrangeBasis_ND(){};
};

template <ushort... N> //<Nx, Ny, Nz ,...> // TODO: Put and classify in utils.
static inline constexpr std::array<std::size_t, sizeof...(N)>
list_2_md_index(const int index) {
  using IndexTuple = std::array<std::size_t, sizeof...(N)>;

  constexpr std::size_t array_dimension = sizeof...(N);

  IndexTuple array_limits{N...};
  IndexTuple md_index; // to return.

  std::integral auto num_positions =
      std::ranges::fold_left(array_limits, 1, std::multiplies<int>());

  try { // TODO: Improve this error handling.
    if (index >= num_positions) {
      throw index >= num_positions;
    }
  } catch (bool out_of_range) {
    std::cout << "Index out of range. Returning zero array." << std::endl;
    return std::array<std::size_t, sizeof...(N)>{0};
  }

  std::integral auto divisor =
      num_positions /
      int(array_limits.back()); // TODO: Check if this is integer division or
                                // use concepts
  std::integral auto I = index;

  for (auto n = array_dimension - 1; n > 0; --n) {
    md_index[n] =
        I / divisor; // TODO: Check if this is integer division or use concepts
    I %= divisor;
    divisor /= array_limits[n - 1];
  }
  md_index[0] = I;

  return md_index;
}; // TODO:  IN CELL TOO (REMOVE BOTH AND MOVE TO UTILS)

template <unsigned short... Ni>
class LagrangeInterpolator_ND { // In regular grid (define as policy?)

  template <unsigned short n> using Array = std::array<double, n>;

  template <unsigned short... n>
  using Basis = interpolation::LagrangeBasis_ND<n...>;

  static constexpr auto dim = sizeof...(Ni);

  // Using structured bindings ?

public:
  Basis<Ni...> basis{};
  Array<(Ni * ...)> fValues{}; // posible usage for md_span

  constexpr auto operator()(const Array<dim> &X) const noexcept {
    std::floating_point auto value{0.0};

    for (auto i = 0; i < (Ni * ...); ++i) {
      auto md_index = list_2_md_index<Ni...>(i);

      value +=[&]<std::size_t... I>(const auto &x, std::index_sequence<dim>){// Templated Lambda
          (std::get<I>(basis).L[md_index[I]](x[I]) *...); // Fold expression
          }(                                              // Inmediate invocation
          std::forward<const Array<dim> &>(X),
          std::make_index_sequence<dim>{});
    };
    return value;
  };

  consteval LagrangeInterpolator_ND(const Basis<Ni...> &basis_,
                                    Array<(Ni * ...)> fi_)
      : basis{basis_}, fValues{fi_} {};

  constexpr ~LagrangeInterpolator_ND(){};
};

} // End of namespace interpolation

#endif // __LAGRANGE_INTERPOLATION_H__
