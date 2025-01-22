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

#include <print>

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
                auto result = function2eval(evalPoints_[0])*weights_[0];  // ESTE OPERACION PUJEDDE ESTAS MAL DEFINIDA!
                //auto result = function2eval(evalPoints_[0]);  // ESTE OPERACION PUJEDDE ESTAS MAL DEFINIDA!     

                //if constexpr(std::is_same_v<returnType,Matrix>){
                //    std::println("First Evaluation");
                //    std::cout << "Eval Point: "; for (auto&& j:evalPoints_[0]){std::cout << j << " ";}std::cout << std::endl;
                //    MatView(function2eval(evalPoints_[0]).mat_ , PETSC_VIEWER_STDOUT_WORLD);
                //    //MatView(result.mat_ , PETSC_VIEWER_STDOUT_WORLD);
                //}

                for(std::size_t i = 1; i < nPoints; ++i) {

                    //std::println(" Evaluation {0}",i);
                    //std::println(" weights_[{0}] = {1}",i,weights_[i]);
                    result += function2eval(evalPoints_[i])*weights_[i];
                    //result += function2eval(evalPoints_[i]);
                        //if constexpr(std::is_same_v<returnType,Matrix>){
                        //    std::cout << "Eval Point: ";
                        //    for (auto&& j:evalPoints_[i]){
                        //        std::cout << j << " ";
                        //    }std::cout << std::endl;
                        //    //MatView(result.mat_ , PETSC_VIEWER_STDOUT_WORLD);
                        //    MatView(function2eval(evalPoints_[i]).mat_ , PETSC_VIEWER_STDOUT_WORLD);
                        //}
                }
                return result;       
            }
        }
    }

    constexpr Quadrature(){};
    constexpr ~Quadrature(){};

};



#endif