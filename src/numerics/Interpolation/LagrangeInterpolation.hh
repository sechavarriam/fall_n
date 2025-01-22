
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


namespace interpolation
{

  template <std::size_t nPoints>
  class LagrangeBasis_1D
  { // constexpr funtor

    std::array<double, nPoints> xPoints{};

  public:
    constexpr std::size_t size() const noexcept { return nPoints; };

    constexpr double x(std::size_t i) const noexcept { return &xPoints[i]; };

    
    constexpr auto operator[](std::size_t i) const noexcept{ //take the index and return a lambda of x.
      //return [&, i](std::floating_point auto x){
      return [&, i](const double&  x){   
        double L_i = 1.0;
        for (std::size_t j = 0; j < nPoints; ++j){
          (j != i) ? L_i *= (x - xPoints[j]) / (xPoints[i] - xPoints[j])
                   : L_i *= 1.0;
        }
        //std::println("shape i = {0}, xPoints[i] = {1}, x = {2}, L_i = {3}", i, xPoints[i], x, L_i);
        return L_i;
      };
    };

    constexpr auto derivative(std::size_t i) const noexcept{
      return [&, i](const double &x) -> double
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

    consteval LagrangeBasis_1D(const std::array<double, nPoints> &xCoordinates) noexcept : xPoints{xCoordinates} {};

    constexpr ~LagrangeBasis_1D() = default;
  };

  template <std::size_t nPoints>
  class LagrangeInterpolator_1D
  {

    using Array = std::array<double, nPoints>;
    using Basis = LagrangeBasis_1D<nPoints>;

    Basis L{}; // Owing or reference?
    Array fValues{};

  public:
    constexpr double operator()(double x) const noexcept
    {
      double value{0.0};
      for (std::size_t i = 0; i < nPoints; ++i)
        value += fValues[i] * L[i](x);
      std::cout << "x= " << x << std::endl;
      std::cout << "HI! Value = " << value << std::endl;
      return value;
    };

    constexpr double derivative(const double x) const noexcept
    {
      double value{0.0};
      for (std::size_t i = 0; i < nPoints; ++i)
        value += fValues[i] * std::invoke(L.derivative(i), x);
      std::cout << "x= " << x << std::endl;
      std::cout << "Derivative value = " << value << std::endl;

      return value;
    };

    consteval LagrangeInterpolator_1D(const Array &xPoints_, const Array &yValues_) noexcept : 
      L{std::forward<const Array &>(xPoints_)},fValues{yValues_} 
      {};

    consteval LagrangeInterpolator_1D(const Basis &basis_,const Array &yValues_) noexcept : 
      L{basis_}, fValues{yValues_} 
      {};

    constexpr ~LagrangeInterpolator_1D() {};
  };

  template <std::size_t... Ni>
  class LagrangeBasis_ND{

    template <std::size_t n>
    using Array = std::array<double, n>;
    
    template <std::size_t n>
    using Basis = interpolation::LagrangeBasis_1D<n>;

    static constexpr auto dim = sizeof...(Ni);

  public:
    std::tuple<Array<Ni>...> coordinates_i{};
    std::tuple<Basis<Ni>...> L{};

    // template<std::size_t i>
    constexpr auto shape_function(std::size_t i) const noexcept {
      static auto md_index = utils::list_2_md_index<Ni...>(i);

      return [&](const auto &x){
        if constexpr (dim == 1)
          return std::get<0>(L)[md_index[0]](x); // TODO:Revisar
        else
          return [&]<std::size_t... I>(std::index_sequence<I...>){
            return (std::get<I>(L)[md_index[I]](x[I]) * ...);
          }(std::make_index_sequence<dim>{});
      };

    };

    template <std::size_t J> requires (J < dim)
    constexpr auto shape_function_derivative(std::size_t i) const noexcept
    {                                                          // $frac{\partial h_i}{\partial x_j}$
      auto md_index = utils::list_2_md_index<Ni...>(i);
      return [&](const auto &x)
      {
        if constexpr (dim == 1){
          return std::get<0>(L).derivative(md_index[0])(x); // TODO:Revisar
        }
        else
        {
          return [&]<std::size_t... I>(std::index_sequence<I...>){
            // Print index sequence
            
            return (((I != J) ? std::invoke(std::get<I>(L)           [md_index[I]], x[I]) :
                                std::invoke(std::get<I>(L).derivative(md_index[I]), x[I])) * ...);
          }(std::make_index_sequence<dim>{});
        } 
      };
    };

    
    constexpr auto shape_function_derivative(std::size_t i, std::size_t j) const noexcept{                                                          // $frac{\partial h_i}{\partial x_j}$
      static std::size_t J{0};                                 // static initialization: only performed once.
      static auto md_index = utils::list_2_md_index<Ni...>(0); // static initialization: only performed once.
      J = j;

      // std::println("Computing derivative of shape function at node {0} with respect to x_{1} = {2}.", i, j, J);

      md_index = utils::list_2_md_index<Ni...>(i);

      return [&](const std::ranges::range auto &x){
        if constexpr (dim == 1)
        {
          return std::get<0>(L).derivative(md_index[0])(x); // TODO:Revisar
        }
        else{
            return 
            [&]<std::size_t... I>(std::index_sequence<I...>)->double{
              //std::cout << "Index sequence: "; (std::cout << ... << I) << std::endl;
              //auto bool_test = std::array<bool, dim>{(I != J)...};
              //std::cout << "Bool test: "; for (auto b : bool_test) std::cout << b << " "; std::cout << std::endl;
              //std::cout << "md_index:  "; for (auto m : md_index)  std::cout << m << " "; std::cout << std::endl;
              return 
              (((I != J) ? std::invoke(std::get<I>(L)[md_index[I]], x[I]) :
                           std::invoke(std::get<I>(L).derivative(md_index[I]), x[I])) * ...);
            }(std::make_index_sequence<dim>{}); 
        }
      };
    };

    constexpr inline auto interpolate(const std::ranges::contiguous_range auto& F, const Array<dim> &X) const noexcept{
      
      double value{0.0};
      
      for (std::size_t i = 0; i < (Ni * ...); ++i){
        auto md_index = utils::list_2_md_index<Ni...>(i);

        value += [&]<std::size_t... I>(const auto &x, std::index_sequence<I...>) { // Templated Lambda
          return (F[i] * (std::get<I>(L)[md_index[I]](x[I]) * ...));   // Fold expression
        }( X , std::make_index_sequence<dim>{});
      };
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

    constexpr ~LagrangeBasis_ND() {};

  };

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

      for (auto i = 0; i < (Ni * ...); ++i)
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

    constexpr ~LagrangeInterpolator_ND() {};
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