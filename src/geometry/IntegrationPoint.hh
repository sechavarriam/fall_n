#ifndef FN_INTEGRATION_POINT
#define FN_INTEGRATION_POINT

#include <concepts>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <optional>

#include "../geometry/Point.hh"
#include "../geometry/Topology.hh"

template <std::size_t dim>
class IntegrationPoint{

     using Array = std::array<double, dim>;

     std::size_t id_;
     Array       coord_;
     double      weight_{0.0};

public:

     bool is_coordinated_{false}; // Flag to check if the material point has the coordinates set.
     bool is_weighted_   {false}; // Flag to check if the material point has weights set.

     inline constexpr auto id() const noexcept { return id_; };
     inline constexpr auto set_id(const std::size_t id) noexcept { id_ = id; }; // No deberia ser necesario puesto que el ID se asigna desde constructor.

     inline constexpr auto coord() const noexcept { return coord_; };
     inline constexpr auto coord(const std::size_t i) const noexcept { return coord_[i]; };

     inline constexpr void set_coord(std::floating_point auto &...coord) noexcept
          requires(sizeof...(coord) == dim){
          coord_ = {coord...};
          is_coordinated_ = true;
     }

     inline constexpr void set_coord(const std::array<double, dim> &coord) noexcept{
          coord_ = coord;
          is_coordinated_ = true;
     }

     inline constexpr void set_coord(const double *coord) noexcept{
          std::copy(coord, coord + dim, coord_.begin());
          is_coordinated_ = true;
     }

     inline constexpr void set_weight(const double weight) noexcept{
          weight_ = weight;
          is_weighted_ = true;
     }

     // Constructors 
     //IntegrationPoint( std::size_t id) noexcept : id_{id} 
     //     {}

     IntegrationPoint() = default;

     ~IntegrationPoint() = default;
};

#endif