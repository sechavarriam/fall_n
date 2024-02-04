#ifndef FN_GAUSS_CUADRATURE
#define FN_GAUSS_CUADRATURE

#include <array>
#include <iostream>
#include <functional> 
#include <algorithm>  
#include <numeric>   
#include <utility>
#include <cstdlib>

#include "../../geometry/Point.hh"


template<std::size_t Dim,std::size_t nPoints> 
using defaultWeightContainer = std::array<double, nPoints*Dim>;

template<std::size_t Dim,std::size_t nPoints> 
using defaultPointsContainer = std::array<double, nPoints*Dim>;

template<std::size_t Dim, std::size_t nPoints, typename P=defaultPointsContainer<Dim,nPoints>, typename W=defaultWeightContainer<Dim,nPoints>>
class Quadrature{
    
    P evalPoints_;
    W weights_   ; // Punteros a un contenedor estático?
    

  public:

   template<typename F>
   
   constexpr double operator()(F function2eval){
       return std::inner_product(weights_.begin(), weights_.end(), evalPoints_.begin(), double(0.0),std::plus<>(),
           [&](const auto& w, const auto& x){
            return w*function2eval(x);});
   };
    constexpr Quadrature(){};

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

template <std::size_t... n> // n: Number of quadrature points per direction.
class CellIntegrator : public Quadrature<sizeof...(n), (n*...), std::array<geometry::Point<sizeof...(n)>*,(n*...)>>
{
  
  using Point = geometry::Point<sizeof...(n)>;

  static constexpr std::size_t dim    {sizeof...(n)};
  static constexpr std::size_t n_nodes{(n*...)};

  constexpr void set_integration_points(){

  };
  
  std::array<std::size_t,dim> orders {n...};



  //using Point = geometry::Point<dim>;

  //std::array<Point*, n_nodes> integration_points_; //Pointer to IntegrationPoint base class.
  //std::array<double, n_nodes> weights_;  

    public:


    
        constexpr CellIntegrator()
        {
            std::cout << "CellIntegrator constructor called." << std::endl;

        };
    
        constexpr ~CellIntegrator(){};
    
        //template<typename F>
        //constexpr double operator()(F function2eval){
        //    return std::accumulate(integration_points_.begin(), integration_points_.end(), double(0.0),
        //        [&](const auto& sum, const auto& point){
        //            return sum + function2eval(*point);
        //        });
        //};


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