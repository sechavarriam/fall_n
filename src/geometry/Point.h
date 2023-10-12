#ifndef FN_POINT
#define FN_POINT

#include <cmath>

#include <iostream> // Header that defines the standard input/output stream objects.
#include <concepts>

#include "../numerics/Vector.h"

#include "Topology.h"

typedef unsigned short ushort;
typedef unsigned int   uint  ;


template<ushort Dim> requires Topology::EmbeddableInSpace<Dim>
class Point {
 
  private:

    Vector<Dim> coord_; //Use of Eigen vector to facilitate operaitons.

  public:

    Point(){}; 

    template<typename... Args>
    Point(Args&&... args) : coord_(std::forward<Args>(args)...){}





    ~Point(){} 

};



#endif