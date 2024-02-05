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



   template<typename F>   
   constexpr auto operator()(F function2eval){
       return std::inner_product(weights_.begin(), weights_.end(), evalPoints_.begin(), double(0.0),std::plus<>(),
           [&](const auto& w, const auto& x){
            return w*function2eval(x);});
   };



    constexpr Quadrature(){};

    /*
    constexpr Quadrature(const W& w,const P& p):weights_(w),evalPoints_(p){
        std::cout << "ref constructor called." << std::endl;
    } ;
    
    constexpr Quadrature(W&& w, P&& p):
        weights_(std::forward<W>(w)),
        evalPoints_(std::forward<P>(p)){
            std::cout << "rvalue ref constructor called." <<std::endl;
        } ;
    */    
    constexpr ~Quadrature(){};

};

template <std::size_t... n> // n: Number of quadrature points in each direction.
class CellIntegrator : public Quadrature<(n*...), std::array<geometry::Point<sizeof...(n)>*,(n*...)>>
{ 
  
  static constexpr std::size_t dim    {sizeof...(n)};
  static constexpr std::size_t n_nodes{(n*...)};
  
  using Point       = geometry::Point<dim>;
  using pPointArray = std::array<Point*, n_nodes>;

  using Quadr = Quadrature<n_nodes, pPointArray>;
  

public:
  using Quadr::operator();

  static constexpr std::tuple< std::array<double,n>...> dir_eval_coords{GaussLegendre::evalPoints<n>()...}; //Coordinates in each direction.
  static constexpr std::tuple< std::array<double,n>...> dir_weights    {GaussLegendre::Weights   <n>()...}; //Weights in each direction.

  //using Quadrature;
  //static constexpr std::array<std::size_t,dim> orders {(n-1)...};

  constexpr void set_weights() noexcept{
    for(auto i = 0; i < n_nodes; ++i){
        [&]<std::size_t... Is>(std::index_sequence<Is...>){
            Quadr::weights_[i] = (std::get<Is>(dir_weights)[utils::list_2_md_index<n...>(i)[Is]] * ...);
        }(std::make_index_sequence<dim>{});
    };
  };
  
  constexpr void set_integration_points() noexcept
    {
      for(auto i = 0; i < n_nodes; ++i){
          [&]<std::size_t... Is>(std::index_sequence<Is...>){
              //Quadr::evalPoints_[i] = new Point{std::get<Is>(dir_eval_coords)[utils::list_2_md_index<n...>(i)[Is]]...};
              Quadr::evalPoints_[i] -> set_coord({std::get<Is>(dir_eval_coords)[utils::list_2_md_index<n...>(i)[Is]]...});
          }(std::make_index_sequence<dim>{});
      };
    };

  //std::array<Point*, n_nodes> integration_points_; //Pointer to IntegrationPoint base class.
  //std::array<double, n_nodes> weights_;  

  public: 

    CellIntegrator() = delete;

    constexpr CellIntegrator(pPointArray&& p):Quadr(std::forward<pPointArray>(p)){
        set_weights();
        set_integration_points();
    };

    using pPointVector = std::vector<Point*>;
    constexpr CellIntegrator(pPointVector& p){
        std::move(p.begin(), p.end(), Quadr::evalPoints_.begin());
        set_weights();
        set_integration_points();
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