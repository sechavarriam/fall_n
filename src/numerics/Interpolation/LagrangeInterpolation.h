
#ifndef FN_LAGRANGE_INTERPOLATION_H
#define FN_LAGRANGE_INTERPOLATION_H

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <ranges>
#include <numeric>

#include <concepts>
#include <functional>
#include <tuple>
#include <variant>
//#include "../../utils/constexpr_function.h"


namespace interpolation{


template<unsigned short nPoints>
class LagrangeBasis_1D{ //constexpr funtor

  std::array<double, nPoints>xPoints{};

  public:
    constexpr auto operator[](std::integral auto i) const noexcept
    {
        return [&,i](std::floating_point auto x){
            double L_i = 1.0;
            for (auto j = 0; j < nPoints; ++j)
            {
                (j != i)? 
                    L_i *= (x - xPoints[j]) / (xPoints[i] - xPoints[j]) : 
                    L_i *= 1.0;
            }
            return L_i;
        }; 
    };

    consteval LagrangeBasis_1D(const std::array<double, nPoints>& xCoordinates) noexcept
     : xPoints{xCoordinates} {};

    constexpr ~LagrangeBasis_1D(){};
};

template<unsigned short... Ni>
class LagrangeBasis_ND{ 
    
    template<unsigned short n>
    using Array = std::array<double, n>;
    
    template<unsigned short n>
    using Basis = interpolation::LagrangeBasis_1D<n>;

    static constexpr auto Dim = sizeof...(Ni);

  public:

    std::tuple<Array<Ni>...> coordinates_i{};
    std::tuple<Basis<Ni>...> L{};  

  public:

    consteval LagrangeBasis_ND(const Array<Ni>&... xCoordinates) noexcept : 
        coordinates_i{xCoordinates...},
        L{Basis<Ni>{xCoordinates}...}
        {};

    constexpr ~LagrangeBasis_ND(){};

};




/*

template<unsigned short... Ni>
class LagrangeBasis_ND_v2{ 
    
    template<unsigned short n>
    using Array = std::array<double, n>;
    
    template<unsigned short n>
    using Basis = interpolation::LagrangeBasis_1D<n>;

    static constexpr auto Dim = sizeof...(Ni);

  public:

    std::array<std::variant<Array<Ni>...>, Dim> coordinates_i{};
    //std::array<std::variant<Basis<Ni>>..., Dim> L{};


  public:
  
    consteval LagrangeBasis_ND_v2(const Array<Ni>&... xCoordinates) noexcept : 
        coordinates_i{std::variant<Array<Ni>...>{xCoordinates}...}
        //L{std::variant<Basis<Ni>>{Basis<Ni>{xCoordinates}}...}
        {};


    constexpr ~LagrangeBasis_ND_v2(){};

};



inline constexpr auto lagrange_interpolation_1d
(
    std::ranges::range auto const&&  X,
    std::ranges::range auto const&& FX,
    auto const&& x
)
{
    return std::transform_reduce
    (
        std::ranges::begin(X), 
        std::ranges::end(X),
        0.0,
        std::plus{},
        [&](auto xi) 
            {
            auto L = std::transform_reduce
            (
                std::ranges::begin(X),
                std::ranges::end(X),
                1.0,
                std::multiplies{},
                [x, xi](auto xj)
                {
                    return xi != xj ? (x - xj) / (xi - xj) : 1.0; 
                }
            );
            return FX[std::ranges::distance(std::ranges::begin(X), std::ranges::find(X, xi))] * L;
            });
};
*/

}



#endif // __LAGRANGE_INTERPOLATION_H__
