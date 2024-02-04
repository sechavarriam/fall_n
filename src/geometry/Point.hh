#ifndef FN_POINT
#define FN_POINT

#include <array>
#include <concepts>
#include <cstddef>
#include "Topology.hh"

typedef unsigned short ushort;
typedef unsigned int   uint  ;

namespace geometry {
  
  template<std::size_t dim> requires topology::EmbeddableInSpace<dim>
  class Point {
   
    private:
      //Vector<dim> coord_; //Use of Eigen vector to facilitate operaitons.
      std::array<double, dim> coord_;

    public:

      //Getters.
      inline constexpr double coord(const std::size_t i) const { return coord_[i]; };


      inline constexpr void set_coord(const std::size_t i, const double value) { coord_[i] = value; };

      inline constexpr void set_coord(const std::array<double,dim>&& coord_array) { 
        std::move(coord_array.begin(),coord_array.end(),coord_.begin()); 
        };

      //inline constexpr std::array<double,dim> coord() const { return &coord_; };



      constexpr Point(){}; 
  
      inline constexpr Point(std::array<double,dim> coord_array) : coord_{coord_array}{}; 
      inline constexpr Point(std::array<double,dim>&& coord_array) : coord_{std::move(coord_array)}{}; 


 
      template<std::floating_point... Args> // Giving error with this constructor.
      inline constexpr Point(Args&&... args) : coord_{std::forward<Args>(args)...}{}
   
      ////Copy constructor and assignment operator.
      //constexpr Point(const Point& other) : coord_{other.coord_}{};
      //constexpr Point& operator=(const Point& other) { coord_ = other.coord_; return *this; };
      //
      ////Move constructor and assignment operator.
      //constexpr Point(Point&& other) noexcept : coord_{std::move(other.coord_)}{};
      //constexpr Point& operator=(Point&& other) noexcept { coord_ = std::move(other.coord_); return *this; };

      //Destructor.
      constexpr ~Point(){} 
  
  };

} // namespace geometry


#endif