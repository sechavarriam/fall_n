#ifndef FN_POINT
#define FN_POINT

#include <array>
#include "Topology.h"

typedef unsigned short ushort;
typedef unsigned int   uint  ;

namespace geometry {
  
  template<ushort Dim> requires topology::EmbeddableInSpace<Dim>
  class Point {
   
    private:
      //Vector<Dim> coord_; //Use of Eigen vector to facilitate operaitons.
      std::array<double, Dim> coord_;

    public:

      //Getters.
      inline constexpr double coord(const ushort i) const { return coord_[i]; };

      inline constexpr std::array<double,Dim> coord() const { return coord_; };



      constexpr Point(){}; 
  
      inline constexpr Point(std::array<double,Dim> coord_array) : coord_{coord_array}{}; 

      template<typename... Args> // Giving error with this constructor.
      inline constexpr Point(Args&&... args) : coord_{std::forward<Args>(args)...}{}
   
      //Copy constructor and assignment operator.
      constexpr Point(const Point& other) : coord_{other.coord_}{};
      constexpr Point& operator=(const Point& other) { coord_ = other.coord_; return *this; };

      //Move constructor and assignment operator.
      constexpr Point(Point&& other) noexcept : coord_{std::move(other.coord_)}{};
      constexpr Point& operator=(Point&& other) noexcept { coord_ = std::move(other.coord_); return *this; };

      //Destructor.
      constexpr ~Point(){} 
  
  };

} // namespace geometry


#endif