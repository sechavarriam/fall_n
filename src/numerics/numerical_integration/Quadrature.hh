#ifndef FN_GAUSS_CUADRATURE
#define FN_GAUSS_CUADRATURE

#include <array>
#include <concepts>
#include <iostream>
#include <functional> 
#include <algorithm>  
#include <numeric>   
#include <utility>
#include <cstdlib>

#include <tuple>

#include "GaussLegendreNodes.hh"
#include "GaussLegendreWeights.hh"

#include "../../geometry/Point.hh"

#include "../../utils/index.hh"


template<std::size_t nPoints> 
using defaultWeightContainer = std::array<double, nPoints>;

template<std::size_t nPoints> 
using defaultPointsContainer = std::array<double, nPoints>;

template<std::size_t nPoints, typename P=defaultPointsContainer<nPoints>, typename W=defaultWeightContainer<nPoints>>
class Quadrature{
    public:
    
    P evalPoints_;
    W weights_   ; 

    template<std::invocable<decltype(evalPoints_[0])> F>
    constexpr std::invoke_result_t<F, decltype(evalPoints_[0])> operator()(F&& function2eval) const noexcept {
        using returnType = std::invoke_result_t<F, decltype(evalPoints_[0])>;
        
        if constexpr(std::is_same_v<returnType, void>){
            std::cerr << "Quadrature Error: The function to evaluate must return a value (e.g. double, Vector, Matrix)" << std::endl;
            std::exit(EXIT_FAILURE);
        } 
        else 
        {
            if constexpr(std::is_same_v<returnType,double>)
            {
                return std::inner_product(weights_.begin(), weights_.end(), evalPoints_.begin(), double(0.0),  std::plus<>(),
                    [&](const auto& w, const auto& x){
                    return w*function2eval(x);});
            }
            else 
            {
                auto result = weights_[0]*function2eval(evalPoints_[0]);     

                for(std::size_t i = 1; i < nPoints; ++i) {
                    result += weights_[i]*function2eval(evalPoints_[i]);
                    }
                return result;       
            }
        }
    }

    constexpr Quadrature(){};
    constexpr ~Quadrature(){};

};



#endif