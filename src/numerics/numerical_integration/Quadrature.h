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
template<typename W, typename P, typename F>
class Quadrature{
    W weights_   ; // Punteros a un contenedor estático?
    P evalPoints_;

  public:
    double operator()(F function2eval){
        return std::inner_product(
            weights_.begin(),weights_.end(),
            evalPoints_.begin(), 
            double(0),
            std::plus<>(),
            [&](const auto& w, const auto& x){return w*function2eval(x);}
            );
    };
//
    double operator()(F& function2eval){
        return std::inner_product(
            weights_.begin(), weights_.end(),
            evalPoints_.begin(),0,
            std::plus<>(),
            [&](const double& w, const double& x){
                return w*function2eval(x);}
            );
    };

    double operator()(F&& function2eval){
        return std::inner_product(
            weights_.begin(), weights_.end(),
            evalPoints_.begin(),0,
            std::plus<>(),
            [&](W& w, P& x){
                return std::forward<W>(w)*function2eval(std::forward<P>(x));}
            );
    };

    Quadrature(){};

    Quadrature(W w, P p):weights_(w),evalPoints_(p){
        std::cout << "value constructor called." << std::endl;
    } ;

    Quadrature(W& w, P& p):weights_(w),evalPoints_(p){
        std::cout << "ref constructor called." << std::endl;
    } ;
    
    Quadrature(W&& w, P&& p):
        weights_(std::forward<W>(w)),
        evalPoints_(std::forward<P>(p)){
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