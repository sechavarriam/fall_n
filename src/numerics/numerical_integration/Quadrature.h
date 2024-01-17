#ifndef FN_GAUSS_CUADRATURE
#define FN_GAUSS_CUADRATURE

#include <array>
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
typedef unsigned short ushort;

template<ushort Dim,ushort Order> 
using defaultWeightContainer = std::array<double, Order*Dim>;

template<ushort Dim,ushort Order> 
using defaultPointsContainer = std::array<double, Order*Dim>;

template<ushort Dim, ushort Order, typename W=defaultWeightContainer<Dim,Order>, typename P=defaultPointsContainer<Dim,Order>>
class Quadrature{
    W weights_   ; // Punteros a un contenedor estático?
    P evalPoints_;

  public:

   template<typename F>
   constexpr double operator()(F function2eval){
       return std::inner_product(
           weights_.begin(),weights_.end(),
           evalPoints_.begin(), 
           double(0),
           std::plus<>(),
           [&](const auto& w, const auto& x){return w*function2eval(x);}
           );
   };

   // DEFINIR EL TERMINOS DE PUNTERO A FUNCION?
   //template<typename F>
   //double operator()(F& function2eval){
   //    return std::inner_product(
   //        weights_.begin(), weights_.end(),
   //        evalPoints_.begin(),0,
   //        std::plus<>(),
   //        [&](const double& w, const double& x){
   //            return w*function2eval(x);}
   //        );
   //};


    constexpr Quadrature(){};

    //Quadrature(W w, P p):weights_(w),evalPoints_(p){
    //    std::cout << "value constructor called." << std::endl;
    //} ;

    constexpr Quadrature(const W& w,const P& p):weights_(w),evalPoints_(p){
        std::cout << "ref constructor called." << std::endl;
    } ;
    
    constexpr Quadrature(W&& w, P&& p):
        weights_(std::forward<W>(w)),
        evalPoints_(std::forward<P>(p)){
            std::cout << "rvalue ref constructor called." <<std::endl;
        } ;
    
    constexpr ~Quadrature(){};

};

    //std::function<double(double)> Fn = [](double x){return x*x;};
    //for (int i = 0; ++i, i<10;)std::cout << i << ' ' << Fn(i) << std::endl;
    //constexpr short order = 15;
    //Quadrature<1,order> GaussOrder3(GaussLegendre::Weights1D<order>(),GaussLegendre::evalPoints1D<order>());
    //std::cout << GaussOrder3([](double x){return x*x;}) << std::endl;
    //std::cout << GaussOrder3(Fn) << std::endl;
    //std::cout << "____________________________" << std::endl;
    //auto W_2D = GaussLegendre::Weights<1,3>();
    //for(auto&& i:W_2D) std::cout << i << std::endl;
    

#endif