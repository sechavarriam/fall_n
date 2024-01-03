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
      Point(){}; 
  
      template<typename... Args>
      inline constexpr Point(Args&&... args) : coord_{std::forward<Args>(args)...}{}
   
      constexpr ~Point(){} 
  
  };

} // namespace geometry


#endif