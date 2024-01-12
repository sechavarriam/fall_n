
#ifndef FN_LAGRANGE_INTERPOLATION_H
#define FN_LAGRANGE_INTERPOLATION_H

#include <array>
#include <cmath>
#include <limits>
#include <ranges>
#include <numeric>

#include <concepts>
#include <functional>

//#include "../../utils/constexpr_function.h"


namespace interpolation{

template<unsigned short nPoints>
class LagrangeBasis{ //constexpr funtor

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

    consteval explicit LagrangeBasis(const std::array<double, nPoints>& xCoordinates) noexcept
     : xPoints{xCoordinates} {};

    constexpr ~LagrangeBasis(){};
    
};



template <unsigned short nPoints>
class LagrangeShapeFunctions{
    
  private:
   
    //std::array<double, nPoints> xPoints_; // Maybe not needed
    std::array<std::function<double(double)>, nPoints> L; // Lagrange basis

  public:
    constexpr LagrangeShapeFunctions(const std::array<double, nPoints>& xPoints)// : xPoints_(xPoints)
    {
        for (unsigned short i = 0; i < nPoints; ++i)
        {
            L[i] = [&, i](double x)
            {
                double L_i = 1.0;
                for (auto j = 0; j < nPoints; ++j)
                {
                    (j != i)? L_i *= (x - xPoints[j]) / (xPoints[i] - xPoints[j]) : L_i *= 1.0;
                }
                return L_i;
            };
        }
    };

    constexpr double operator()(const std::array<double, nPoints>& yValues, double x) const
    {
        return std::inner_product
        (
            std::begin(yValues),
            std::end  (yValues),
            std::begin(L),
            0.0,
            std::plus<>(),
            [&](auto y, auto l)
            {
                return y * l(x);
            }
        );
    };

    constexpr ~LagrangeShapeFunctions(){};
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



}



#endif // __LAGRANGE_INTERPOLATION_H__
