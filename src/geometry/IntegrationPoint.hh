#ifndef FN_INTEGRATION_POINT
#define FN_INTEGRATION_POINT

#include <algorithm>
#include <concepts>
#include <initializer_list>
#include <iostream>
#include <array>

#include <vtkType.h>

#include "../geometry/Topology.hh"

template <std::size_t Dim>
class IntegrationPoint{

     // Numerical sample site used by quadrature.  It is intentionally distinct
     // from mesh vertices/nodes even when collocated with them (e.g. Lobatto
     // endpoints), because integration role and nodal-interpolation role are
     // not the same concept.

     using Array = std::array<double, Dim>;

     vtkIdType   id_{0};
     Array       coord_{};
     double      weight_{0.0};
     bool        is_coordinated_{false};
     bool        is_weighted_{false};

public:
     static constexpr std::size_t dim = Dim;

     inline constexpr auto id()   const noexcept { return id_; };
     inline constexpr auto id_p() const noexcept { return &id_; }; // Pointer to the id... Reqiured for VTK 
     inline constexpr auto set_id(const std::size_t id) noexcept { id_ = id; }; // No deberia ser necesario puesto que el ID se asigna desde constructor.

     inline constexpr const Array& coord() const noexcept { return coord_; };
     inline constexpr double coord(const std::size_t i) const noexcept { return coord_[i]; };
     inline constexpr const Array& coord_ref() const noexcept { return coord_; }
     inline constexpr const double* data() const noexcept { return coord_.data(); }
     inline constexpr double* data() noexcept { return coord_.data(); }
     inline constexpr auto weight() const noexcept { return weight_; }
     inline constexpr bool is_weighted() const noexcept { return is_weighted_; }
     inline constexpr bool is_coordinated() const noexcept { return is_coordinated_; }

     inline constexpr void set_coord(std::floating_point auto &...coord) noexcept
          requires(sizeof...(coord) == Dim){
          coord_ = {coord...};
          is_coordinated_ = true;
     }

     inline constexpr void set_coord(const Array &coord) noexcept{
          coord_ = coord;
          is_coordinated_ = true;
     }

     inline constexpr void set_coord(const double *coord) noexcept{
          std::copy(coord, coord + Dim, coord_.begin());
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
