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

// in clang, print
#ifdef __clang__ 
#include <format>
#include <print>
#endif

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
    constexpr decltype(auto) operator()(F&& function2eval) const noexcept {
        using returnType = std::invoke_result_t<F, decltype(evalPoints_[0])>;
        
        if constexpr(std::is_same_v<returnType, void>){ 
            std::cerr << "Quadrature Error: The function to evaluate must return a value (e.g. double, Vector, Matrix)" << std::endl;
            std::exit(EXIT_FAILURE);
        } 
        else {
            if constexpr(std::is_same_v<returnType,double>) {
                return std::inner_product(weights_.begin(), weights_.end(), evalPoints_.begin(), double(0.0),  std::plus<>(),
                    [&](const auto& w, const auto& x){
                    return w*function2eval(x);});
            }
            else if constexpr(std::is_base_of_v<Eigen::MatrixBase<returnType>, returnType>) // Eigen Matrices and Vectors 
            {
                using MatrixType = Eigen::Matrix<double, returnType::RowsAtCompileTime, returnType::ColsAtCompileTime>;
                return [&]<std::size_t... I>(std::index_sequence<I...>)->MatrixType{
                    return ((function2eval(evalPoints_[I])*weights_[I]) + ...);
                }(std::make_index_sequence<nPoints>{});
            }
            else //default
            {
                returnType result = function2eval(evalPoints_[0])*weights_[0];  
                for(std::size_t i = 1; i < nPoints; ++i) {
                    result += function2eval(evalPoints_[i])*weights_[i];}
                return result;       
            }
        }
    }

    constexpr Quadrature(){};
    constexpr ~Quadrature(){};

};



#endif