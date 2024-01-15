
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


template<unsigned short nPoints>
class LagrangeInterpolator_1D{
    
    using Array = std::array<double, nPoints>;
    using Basis = LagrangeBasis_1D<nPoints>;

    Basis L      {}; // Owing or reference? 
    Array yValues{};

    public:
    

    constexpr double operator()(double x) const noexcept
    {
        auto value = 0;
        for (auto i = 0; i < nPoints; ++i)
        {
            value += yValues[i] * L[i](x);
        }
        return value;
    };

    consteval LagrangeInterpolator_1D(const Array& xPoints_, const Array& yValues_) noexcept:
        L{std::forward<const Array&>(xPoints_)},
        yValues{yValues_} 
        {};

    consteval LagrangeInterpolator_1D(const Basis& basis_, const Array& yValues_ ) noexcept:
        L{basis_},
        yValues{yValues_} 
        {};

    constexpr ~LagrangeInterpolator_1D(){};


};




template <class Tuple, class F> // From https://www.fluentcpp.com/2021/03/05/stdindex_sequence-and-its-improvement-in-c20/
                                // TODO: Move to utils folder and namespace.
    constexpr decltype(auto) for_each(Tuple&& tuple, F&& f)
    {
        return [] <std::size_t... I>
        (Tuple&& tuple, F&& f, std::index_sequence<I...>)
        {
            (f(std::get<I>(tuple)), ...);
            return f;
        } // End of lambda
        (/*inmediate invocation (use std::invoke to clarify?)*/
            std::forward<Tuple>(tuple),
            std::forward<F>(f),
            std::make_index_sequence<std::tuple_size<std::remove_reference_t<Tuple>>::value>{}
        );
    }

template <class Tuple, class Array, class F> // TODO: requires size of tuple = size of array
    constexpr decltype(auto) for_each_tuple_each_arrayelem(Tuple&& tuple, Array&& array, F&& f)
    {
        return [] <std::size_t... I>
        (Tuple&& tuple, Array&& array, F&& f, std::index_sequence<I...>)
        {
            (f(std::get<I>(tuple), array[I]), ...);
            return f;
        } // End of lambda
        (/*inmediate invocation (use std::invoke to clarify?)*/
            std::forward<Tuple>(tuple),
            std::forward<Array>(array),
            std::forward<F>(f),
            std::make_index_sequence<std::tuple_size<std::remove_reference_t<Tuple>>::value>{}
        );
    }

template<unsigned short... Ni>
class LagrangeBasis_ND{ 
    
    template<unsigned short n>
    using Array = std::array<double, n>;
    
    template<unsigned short n>
    using Basis = interpolation::LagrangeBasis_1D<n>;

    static constexpr auto dim = sizeof...(Ni);

  public:

    std::tuple<Array<Ni>...> coordinates_i{};
    std::tuple<Basis<Ni>...> L{};  


    constexpr decltype(auto) evaluate_basis_function (const Array<dim>& X) const noexcept
    {
        return for_each_tuple_each_arrayelem(L, X,
            [](auto&& basis, auto&& x){
                return basis[x];
                }
            );
    };


    consteval LagrangeBasis_ND(const Array<Ni>&... xCoordinates) noexcept : 
        coordinates_i{xCoordinates...},
        L{Basis<Ni>{xCoordinates}...}
        {};

    constexpr ~LagrangeBasis_ND(){};

};



}



#endif // __LAGRANGE_INTERPOLATION_H__
