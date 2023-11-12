#ifndef FN_POINT
#define FN_POINT

#include <array>
#include <cmath>

#include <iostream> // Header that defines the standard input/output stream objects.
#include <concepts>

#include "../numerics/Vector.h"

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
  
      //void print_coords() const {
      //  for (auto const& c : coord_) {
      //    std::cout << c << " ";
      //  }
      //  std::cout << std::endl;
      //}

      Point(){}; 
  
      template<typename... Args>
      inline constexpr Point(Args&&... args) : coord_{std::forward<Args>(args)...}{}
   
      constexpr ~Point(){} 
  
  };

} // namespace geometry


#endif