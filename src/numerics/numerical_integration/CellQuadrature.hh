#ifndef FALL_N_CELL_QUADRATURE_HH
#define FALL_N_CELL_QUADRATURE_HH


#include "Quadrature.hh"

namespace GaussLegendre {

template <std::size_t... n> // n: Number of quadrature points in each direction.
class CellQuadrature : public Quadrature<(n*...), std::array<geometry::Point<sizeof...(n)>,(n*...)>>
{ 
  
  static constexpr std::size_t dim    {sizeof...(n)};
  static constexpr std::size_t n_nodes{(n*...)};
  
  using Point       = geometry::Point<dim>;
  using pPointArray = std::array<Point, n_nodes>;

  using Quadr = Quadrature<n_nodes, pPointArray>;
  

public:
  using Quadr::operator();

  static constexpr std::tuple< std::array<double,n>...> dir_eval_coords{GaussLegendre::evaluation_points<n>()...}; //Coordinates in each direction.
  static constexpr std::tuple< std::array<double,n>...> dir_weights    {GaussLegendre::weights          <n>()...}; //Weights in each direction.

  //using Quadrature;
  //static constexpr std::array<std::size_t,dim> orders {(n-1)...};

  constexpr void set_weights() noexcept{
    for(std::size_t i = 0; i < n_nodes; ++i){
        [&]<std::size_t... Is>(std::index_sequence<Is...>){
            Quadr::weights_[i] = (std::get<Is>(dir_weights)[utils::list_2_md_index<n...>(i)[Is]] * ...);
        }(std::make_index_sequence<dim>{});
    };
  };
  
  constexpr void set_integration_points() noexcept
    {
      for(std::size_t i = 0; i < n_nodes; ++i){
          [&]<std::size_t... Is>(std::index_sequence<Is...>){
              Quadr::evalPoints_[i].set_coord({std::get<Is>(dir_eval_coords)[utils::list_2_md_index<n...>(i)[Is]]...});
          }(std::make_index_sequence<dim>{});
      };
    };

  public: 

    constexpr CellQuadrature(){
      set_weights();
      set_integration_points();
      };
        
    constexpr ~CellQuadrature(){};  
};


}

#endif // FALL_N_CELL_QUADRATURE_HH