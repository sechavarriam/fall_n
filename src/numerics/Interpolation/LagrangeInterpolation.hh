
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


namespace interpolation{

  template <std::size_t nPoints>
  class LagrangeBasis_1D{  // constexpr funtor
  
    
  std::array<double, nPoints> xPoints{};

  public:

    constexpr std::size_t size() const noexcept { return nPoints; };
    constexpr double x(std::size_t i) const noexcept { return xPoints[i]; };
    constexpr const std::array<double, nPoints>& points() const noexcept { return xPoints; };

    constexpr auto operator[](std::size_t i) const noexcept{
      return [this, i](const double&  x){   
        double L_i = 1.0;
        for (std::size_t j = 0; j < nPoints; ++j){
          (j != i) ? L_i *= (x - xPoints[j]) / (xPoints[i] - xPoints[j])
                   : L_i *= 1.0;
        }
        return L_i;
      };
    };

    constexpr auto derivative(std::size_t i) const noexcept{
      return [this, i](const double &x) -> double
      {
        std::size_t j,k;
        
        double dL_i{0.0};
        double num{1.0};
        double den{1.0};

        for (j = 0; j < nPoints; ++j){
          if (j != i){
            for (k = 0; k < nPoints; ++k){
              if (k != i){
                den *= (xPoints[i] - xPoints[k]);
                if (k != j){
                  num *= (x - xPoints[k]);
                }
              }
            }
            dL_i += num / den;
            num = 1.0;
            den = 1.0;
          }
        }
        //std::println("deriv i = {0}, xPoints[i] = {1}, x = {2}, dL_i = {3}", i, xPoints[i], x, dL_i);
        return dL_i;
      };
    };

    constexpr LagrangeBasis_1D(const std::array<double, nPoints> &xCoordinates) noexcept : xPoints{xCoordinates} {};
    constexpr LagrangeBasis_1D() noexcept = default;
    constexpr ~LagrangeBasis_1D() = default;
  };

  template <std::size_t nPoints>
  class LagrangeInterpolator_1D
  {

    using Array = std::array<double, nPoints>;
    using Basis = LagrangeBasis_1D<nPoints>;

    Basis L      {}; 
    Array fValues{};

  public:
    constexpr double operator()(double x) const noexcept
    {
      double value{0.0};
      for (std::size_t i = 0; i < nPoints; ++i)
        value += fValues[i] * L[i](x);
      return value;
    };

    constexpr double derivative(const double x) const noexcept
    {
      double value{0.0};
      for (std::size_t i = 0; i < nPoints; ++i) value += fValues[i] * std::invoke(L.derivative(i), x);

      return value;
    };

    constexpr LagrangeInterpolator_1D(const Array &xPoints_, const Array &yValues_) noexcept : 
      L{std::forward<const Array &>(xPoints_)},fValues{yValues_} 
      {};

    constexpr LagrangeInterpolator_1D(const Basis &basis_,const Array &yValues_) noexcept : 
      L{basis_}, fValues{yValues_} 
      {};

    constexpr ~LagrangeInterpolator_1D() {};
  };

// =================================================================================================
// =================================================================================================
// =================================================================================================

  template <std::size_t... Ni> 
  class LagrangeBasis_ND{

    template <std::size_t... Ns> // Helper para especialización parcial
    struct first;

    template <std::size_t N0, std::size_t... Ns>
    struct first<N0, Ns...> : std::integral_constant<std::size_t, N0> {};

    static constexpr auto dim = sizeof...(Ni);

    template <std::size_t n> using Array = std::array<double, n>;
    template <std::size_t n> using Basis = interpolation::LagrangeBasis_1D<n>;

    using CoordinateType = std::conditional_t<dim == 1, Array<first<Ni...>::value>, std::tuple<Array<Ni>...>>;
    using BasisType      = std::conditional_t<dim == 1, Basis<first<Ni...>::value>, std::tuple<Basis<Ni>...>>;

  public:

    CoordinateType coordinates_i{};
    BasisType                  L{};
    
    constexpr auto shape_function(std::size_t i) const noexcept {

      if constexpr (dim == 1){
        return [this, i](const auto &x) -> double {
          return L[i](x[0]);
        };
      }
      else{
        auto md_index = utils::list_2_md_index<Ni...>(i);
        return [this, md_index](const std::ranges::range auto &x) -> double {
          return [&]<std::size_t... I>(std::index_sequence<I...>) -> double {
            return (std::get<I>(L)[md_index[I]](x[I]) * ...);
          }(std::make_index_sequence<dim>{});
        }; 
      }
    };

    constexpr auto shape_function_derivative(std::size_t i, std::size_t j) const noexcept{ // $frac{\partial h_i}{\partial x_j}$
      if constexpr (dim == 1){
        return [this, i](const std::ranges::range auto &x) -> double {
          return L.derivative(i)(x[0]);
        };
      }
      else{
        auto md_index = utils::list_2_md_index<Ni...>(i);

        return [this, md_index, j](const std::ranges::range auto &x) -> double {
          return [&]<std::size_t... I>(std::index_sequence<I...>) -> double {
            return (((I != j)
                         ? std::invoke(std::get<I>(L)[md_index[I]], x[I])
                         : std::invoke(std::get<I>(L).derivative(md_index[I]), x[I])) *
                    ...);
          }(std::make_index_sequence<dim>{});
        };
      }
    };

    constexpr inline auto interpolate(const std::ranges::contiguous_range auto& F, const Array<dim> &X) const noexcept{
      
      double value{0.0};
      
      if constexpr (dim == 1){
        for (std::size_t i = 0; i < (Ni * ...); ++i)
          value += F[i] * L[i](X[0]);
      }
      else{
      for (std::size_t i = 0; i < (Ni * ...); ++i){
        auto md_index = utils::list_2_md_index<Ni...>(i);

        value += [&]<std::size_t... I>(const auto &x, std::index_sequence<I...>) { // Templated Lambda
          return (F[i] * (std::get<I>(L)[md_index[I]](x[I]) * ...));   // Fold expression
        }( X , std::make_index_sequence<dim>{});
      };
      }
      return value;
    };

    // UNTESTED + UNUSED YET
    constexpr inline auto interpolate_derivative(const std::ranges::contiguous_range auto& F, const Array<dim> &X, std::size_t j) const noexcept{
      
      double value{0.0};  
      for (std::size_t i = 0; i < (Ni * ...); ++i){
        auto md_index = utils::list_2_md_index<Ni...>(i);

        value += [&]<std::size_t... I>(const auto &x, std::index_sequence<I...>) { // Templated Lambda
          return (F[i] * (std::get<I>(L).derivative(md_index[I],j)(x[I]) * ...));   // Fold expression
                  }( X , std::make_index_sequence<dim>{});
      };

      return value;
    };

    consteval LagrangeBasis_ND(const Array<Ni> &...xCoordinates) noexcept
        : coordinates_i{xCoordinates...}, L{Basis<Ni>{xCoordinates}...} {};

    constexpr ~LagrangeBasis_ND() = default;
  
  };


  // ===================================================
  template <std::size_t... Ni>
  class LagrangeInterpolator_ND
  { // In regular grid (define as policy?)

    template <std::size_t n>
    using Array = std::array<double, n>;

    template <std::size_t... n>
    using Basis = interpolation::LagrangeBasis_ND<n...>;

    static constexpr std::size_t dim = sizeof...(Ni);

  public:
    Basis<Ni...> basis{};
    Array<(Ni * ...)> fValues{}; // posible usage for md_span

    constexpr auto operator()(const Array<dim> &X) const noexcept{
      std::floating_point auto value{0.0};

      for (std::size_t i = 0; i < (Ni * ...); ++i)
      {
        auto md_index = utils::list_2_md_index<Ni...>(i);

        value += [&]<std::size_t... I>(const auto &x, std::index_sequence<I...>) { // Templated Lambda
          return (fValues[i] * (std::get<I>(basis.L)[md_index[I]](x[I]) * ...));   // Fold expression
        }(                                                                         // Inmediate invocation
                     std::forward<const Array<dim> &>(X), std::make_index_sequence<dim>{});
      };
      return value;
    };

    consteval LagrangeInterpolator_ND(const Basis<Ni...> &basis_, Array<(Ni * ...)> fi_)
        : basis{basis_}, fValues{fi_} {};

    constexpr ~LagrangeInterpolator_ND() = default;
  };

} // End of namespace interpolation

#endif // __LAGRANGE_INTERPOLATION_H__


/*  TO TEST IN MAIN:

    std::cout << "-- 1D INTERPOLATOR ---------------------------------" << std::endl;
    //auto F = interpolation::LagrangeInterpolator_1D<2>{ interpolation::LagrangeBasis_1D<2>{{-10, 10}} , {2.0,4.0} };
    auto F = interpolation::LagrangeInterpolator_1D<3>{ {-1.0, 0.0, 1.0} , {1.0, 0.0, 1.0} };

    //auto f = interpolation::LagrangeInterpolator_ND<2>{ interpolation::LagrangeBasis_ND<2>{ {-10, 10}} , {2.0,4.0}};

    using namespace matplot;
    auto xx  = linspace(-1 , 1, 101);
    //auto y  = transform(x, [=](double x) { return F(x); });

    auto yy = transform(xx, [=](double xx) { return F.derivative(xx); });

    //auto z = transform(x, [=](double x) { return f(std::array<double,1>{x}); }); //OK!
    plot(xx, yy);
    //plot(x, z);
    show();
*/
/*
    std::cout << "-- ND INTERPOLATOR ---------------------------------" << std::endl;

    static constexpr interpolation::LagrangeBasis_ND <2,2> L2_2({0.0,1.0},{0.0, 1.0});

    interpolation::LagrangeInterpolator_ND<2,2> F2_2(L2_2, {1.0, 0.5, -1.0, 2.0});
    std::cout << F2_2({0.5,0.5}) << std::endl;
    static constexpr interpolation::LagrangeBasis_ND <3,4> L3_4({-1.0,0.0,1.0},{-1.0, -2.0/3.0, 2.0/3.0, 1.0});
    interpolation::LagrangeInterpolator_ND<3,4> F3_4(L3_4, {-4,2,4,
                                                            5,-5,3,
                                                            2,3,4,
                                                            1,2,3})
    using namespace matplot;
    auto [X, Y] = meshgrid(linspace(-1, 1, 100), linspace(-1, 1, 100));
    auto Z = transform(X, Y, [=](double x, double y) {
        return F3_4({x,y});
    });
    surf(X, Y, Z);
    show();
*/


// HELPERS

/*
// utility to avoid serializing function evaluation with comma operator (arguments of a function have no order of evaluation)
template<class T>
constexpr void evaluateFoldExpression(std::initializer_list<T>&&)
{}

// utility to expand integer sequence over a functor
template<class F, class I, I... i>
decltype(auto) constexpr unpackIntegerSequence(F&& f, std::integer_sequence<I, i...> sequence)
{
    return f(std::integral_constant<I, i>()...);
}

// utility to evaluate an index sequence at compile time (i.e. unfold a loop with templates)
template<class I, I... i, class F>
void forEach(std::integer_sequence<I, i...> sequence, F&& f){
    unpackIntegerSequence([f](auto... is){
        evaluateFoldExpression<int>({(f(std::integral_constant<I,i>()), 0)...});
    }, sequence);
}
*/
