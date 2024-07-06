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

    //template<typename F>   
    constexpr auto operator()(std::invocable<decltype(evalPoints_[0])> auto&& function2eval) const noexcept {
        return std::inner_product(weights_.begin(), weights_.end(), evalPoints_.begin(), double(0.0),std::plus<>(),
            [&](const auto& w, const auto& x){
             return w*function2eval(x);});
    }

    constexpr Quadrature(){};
    constexpr ~Quadrature(){};

};



#endif