#ifndef FN_GAUSS_CUADRATURE
#define FN_GAUSS_CUADRATURE

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

    double operator()(T&& evalPoints){
        return std::inner_product(
            weights_.begin(), weights_.end(),
            evalPoints.begin(),0,
            std::plus<>(),
            [&](T& w, T& x){return std::forward<T>(w)*function2eval_(std::forward<T>(x));}
            );
    };

    Quadrature(){};

    Quadrature(T w, F f):weights_(w),function2eval_(f){} ;

    ~Quadrature(){};

};

/* Possible Usage

Quadrature<std::array<double>, ShapeFunction> Q; 

Q()

*/

#endif