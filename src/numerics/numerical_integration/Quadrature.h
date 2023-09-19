#ifndef FN_GAUSS_CUADRATURE
#define FN_GAUSS_CUADRATURE

#include <iostream>
#include <functional> // https://en.cppreference.com/w/cpp/header/functional
#include <algorithm>  // https://en.cppreference.com/w/cpp/algorithm
#include <numeric>    // https://en.cppreference.com/w/cpp/header/numeric



// Algoritmos candidatos a ser usados.
// https://en.cppreference.com/w/cpp/algorithm/accumulate
// https://en.cppreference.com/w/cpp/algorithm/inner_product


// a Class?
// a Functor?
// a Function Template?
template<typename T, typename F>
class Quadrature{
    T weights_;
    F function2eval_; // Only one function

  public:


    double operator()(T evalPoints){
        return std::inner_product(
            weights_.begin(),weights_.end(),
            evalPoints.begin(), 
            double(0),
            std::plus<>(),
            [&](const auto& w, const auto& x){return w*function2eval_(x);}
            );
    };

    double operator()(T& evalPoints){
        return std::inner_product(
            weights_.begin(), weights_.end(),
            evalPoints.begin(),0,
            std::plus<>(),
            [&](const double& w, const double& x){
                return w*function2eval_(x);}
            );
    };

    //double operator()(T&& evalPoints){
    //    return std::inner_product(
    //        weights_.begin(), weights_.end(),
    //        evalPoints.begin(),0,
    //        std::plus<>(),
    //        [&](T& w, T& x){
    //            return std::forward<T>(w)*function2eval_(std::forward<T>(x));}
    //        );
    //};

    Quadrature(){};

    //Quadrature(T w, F f):weights_(w),function2eval_(f){
    //    std::cout << "value constructor called." << std::endl;
    //} ;

    Quadrature(T& w, F& f):weights_(w),function2eval_(f){
        std::cout << "ref constructor called." << std::endl;
    } ;
    
    Quadrature(T&& w, F&& f):
        weights_(std::forward<T>(w)),
        function2eval_(std::forward<F>(f)){
            std::cout << "rvalue ref constructor called." <<std::endl;
        } ;
    ~Quadrature(){};

};

/* Possible Usage

Quadrature<std::array<double>, ShapeFunction> Q; //No es tan interesante, en la cuadratura los puntos son fijos.
Quadrature<std::array<double>, std::array<points>> Q; 
Quadrature<std::array<double>, std::array<GaussPoints>, JACOBIAN?> Q(shape_function); 

Quadrature<std::array<double>, std::array<GaussPoints>, JACOBIAN?> Q(shape_function, jacobian?);
dim? order?



Q()

*/

#endif