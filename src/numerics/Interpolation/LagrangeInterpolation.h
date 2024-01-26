
#ifndef FN_LAGRANGE_INTERPOLATION_H
#define FN_LAGRANGE_INTERPOLATION_H

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <iostream>
#include <limits>
#include <numeric>
#include <ranges>


#include <tuple>
#include <utility>


#include "../../utils/index.hh"

namespace interpolation {

template <std::size_t nPoints> class LagrangeBasis_1D; // Forward declaration


//template <std::size_t nPoints> 
//class d_dx
//{ 
//  using Basis = LagrangeBasis_1D<nPoints>;
//  
//  public:
//  constexpr static void* owner {nullptr};
//
//    constexpr auto operator[](std::integral auto i) const noexcept
//    {
//      return [&, i](std::floating_point auto x) {
//        double dL_i = 0.0;
//        for (auto j = 0; j < nPoints; ++j) {
//          (j != i) ? dL_i += 1.0/(static_cast<Basis*>(owner)->x(i) - static_cast<Basis*>(owner)->x(j))
//                   : dL_i += 0.0;
//        }
//        return dL_i * static_cast<Basis*>(owner)->operator[](i)(x);
//      };
//    };
//
//   constexpr d_dx() = default;
//   constexpr d_dx(Basis* owner_) noexcept : owner{owner_} {};
//   constexpr ~d_dx() = default;
//};


template <std::size_t nPoints> 
class LagrangeBasis_1D { // constexpr funtor

std::array<double, nPoints> xPoints{};

public:

constexpr double x(std::size_t i) const noexcept { return &xPoints[i]; };

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

  constexpr auto derivative(std::size_t i) const noexcept 
  {
    return [&, i](double x) 
    {
      double dL_i{0.0};
      double num{1.0};
      double den{1.0};

      for (auto j = 0; j < nPoints; ++j){
        if (j != i){
          for (auto k = 0; k < nPoints; ++k){
            if (k != i){
              den *= (xPoints[i] - xPoints[k]);
              if (k != j) {
                num *= (x - xPoints[k]);
              }
            }
          }
          dL_i += num / den;
          num = 1.0;
          den = 1.0;
        }
      }
      return dL_i;
    };
  };


  consteval LagrangeBasis_1D( const std::array<double, nPoints> &xCoordinates) noexcept : 
    xPoints{xCoordinates}
    {};
  
  constexpr ~LagrangeBasis_1D(){};
};





template <std::size_t nPoints> class LagrangeInterpolator_1D {

  using Array = std::array<double, nPoints>;
  using Basis = LagrangeBasis_1D<nPoints>;

  Basis L{}; // Owing or reference?
  Array fValues{};

public:
  constexpr double operator()(double x) const noexcept {
    double value = 0.0;

    for (auto i = 0; i < nPoints; ++i) {
      value += fValues[i] * L[i](x);
    }
    return value;
  };

  constexpr double derivative(const double x) const noexcept {
    double value = 0.0;

    for (auto i = 0; i < nPoints; ++i) {
      value += fValues[i] * std::invoke(L.derivative(i),x);//L.derivative(i)(x);
      std::cout << "Li="<< std::invoke(L.derivative(i),x) << std::endl;      
    }
    std::cout << "------------" << std::endl;
    std::cout << "x="<<x<<"  value="<< value << std::endl;

    std::cout << "-------------------------------------------" << std::endl;
    return value;
  };

  consteval LagrangeInterpolator_1D(const Array &xPoints_,
                                    const Array &yValues_) noexcept : 
      L{std::forward<const Array &>(xPoints_)},
      fValues{yValues_} 
      {};

  consteval LagrangeInterpolator_1D(const Basis &basis_,
                                    const Array &yValues_) noexcept
      : L{basis_}, fValues{yValues_} {};

  constexpr ~LagrangeInterpolator_1D(){};
};

template <std::size_t... Ni> class LagrangeBasis_ND {

  template <std::size_t n> using Array = std::array<double, n>;
  template <std::size_t n> using Basis = interpolation::LagrangeBasis_1D<n>;

  static constexpr auto dim = sizeof...(Ni);

public:
  std::tuple<Array<Ni>...> coordinates_i{};
  std::tuple<Basis<Ni>...> L{};

  consteval LagrangeBasis_ND(const Array<Ni> &...xCoordinates) noexcept
      : coordinates_i{xCoordinates...}, L{Basis<Ni>{xCoordinates}...} {};

  constexpr ~LagrangeBasis_ND(){};
};



template <unsigned short... Ni>
class LagrangeInterpolator_ND { // In regular grid (define as policy?)

  template <unsigned short n> using Array = std::array<double, n>;

  template <unsigned short... n>
  using Basis = interpolation::LagrangeBasis_ND<n...>;

  static constexpr std::size_t dim = sizeof...(Ni);


public:
  Basis<Ni...> basis{};
  Array<(Ni * ...)> fValues{}; // posible usage for md_span

  constexpr auto operator()(const Array<dim> &X) const noexcept {
    std::floating_point auto value{0.0};

    for (auto i = 0; i < (Ni * ...); ++i) {
      auto md_index = utils::list_2_md_index<Ni...>(i);

      value +=[&]<std::size_t... I>(const auto &x, std::index_sequence<I...> ){// Templated Lambda
          return (fValues[i]*(std::get<I>(basis.L)[md_index[I]](x[I]) * ... )); // Fold expression
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
