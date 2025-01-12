#ifndef FALL_N_CELL_QUADRATURE_HH
#define FALL_N_CELL_QUADRATURE_HH


#include "Quadrature.hh"

namespace GaussLegendre {

//template <std::size_t... n> // n: Number of quadrature points in each direction.
//class CellQuadrature : public Quadrature<(n*...), std::array<geometry::Point<sizeof...(n)>,(n*...)>>  //For nodes as geometry::Point
template <std::size_t... n> 
class CellQuadrature : public Quadrature<(n*...), std::array<std::array<double, sizeof...(n)>,(n*...)>> //For nodes as array of coordinates
{ 
  static constexpr std::size_t dim    {sizeof...(n)};
  //using Point      = geometry::Point<dim>;
  
public:
  

  static constexpr std::size_t num_points{(n*...)};
  using Point      = std::array<double, dim>;
  using PointArray = std::array<Point , num_points>;

  using Quadr = Quadrature<num_points, PointArray>;
  using Quadr::operator();

  static constexpr std::tuple< std::array<double,n>...> dir_eval_coords{GaussLegendre::evaluation_points<n>()...}; //Coordinates in each direction.
  static constexpr std::tuple< std::array<double,n>...> dir_weights    {GaussLegendre::weights          <n>()...}; //Weights in each direction.

  //using Quadrature;
  //static constexpr std::array<std::size_t,dim> orders {(n-1)...};

  constexpr void set_weights() noexcept{
    for(std::size_t i = 0; i < num_points; ++i){
        [&]<std::size_t... Is>(std::index_sequence<Is...>){
            Quadr::weights_[i] = (std::get<Is>(dir_weights)[utils::list_2_md_index<n...>(i)[Is]] * ...);
        }(std::make_index_sequence<dim>{});
    };
  };
  
  constexpr void set_integration_points() noexcept
    {
      for(std::size_t i = 0; i < num_points; ++i){
          [&]<std::size_t... Is>(std::index_sequence<Is...>){
              Quadr::evalPoints_[i] = std::array{std::get<Is>(dir_eval_coords)[utils::list_2_md_index<n...>(i)[Is]]...};  //Nodes as array of coordinates
              //Quadr::evalPoints_[i].set_coord({std::get<Is>(dir_eval_coords)[utils::list_2_md_index<n...>(i)[Is]]...}); //Nodes as geomety::Point
          }(std::make_index_sequence<dim>{});
      };
    };

  public: 

  // TODO: REVISAR QUE TAN VALIDO ES DEVOLVER UNA REFERENCIA.
  constexpr Point get_point_coords(std::size_t i) const noexcept{
    return Quadr::evalPoints_[i];
  };

  constexpr double get_point_weight(std::size_t i) const noexcept{
    return Quadr::weights_[i];
  };


    consteval CellQuadrature(){
      set_weights();
      set_integration_points();
      };
        
    constexpr ~CellQuadrature () = default;
};


}

#endif // FALL_N_CELL_QUADRATURE_HH