#ifndef FN_POINT
#define FN_POINT

#include <cmath>

#include <iostream> // Header that defines the standard input/output stream objects.
#include <concepts>

#include <Eigen/Dense>

#include "Topology.h"

typedef unsigned short ushort;
typedef unsigned int   uint  ;


template<ushort Dim> requires Topology::EmbeddableInSpace<Dim>
class Point {
 
  private:

    uint id_ ; //Considerar quitar el id de nodo si los puntos de integración no lo necesitan.
    Eigen::Matrix<double, Dim, 1> coord_; //Use of Eigen vector to facilitate operaitons.

  public:

    virtual inline void set_id (int t){id_=t;}
    virtual inline void set_tag(int t){id_=t;}
  
    virtual inline int id (){return id_;}
    virtual inline int tag(){return id_;}
  
    virtual inline double* coord(int i){return &coord_[i];}

    Point(){}; 
    Point(int tag, double Coord1, double Coord2):id_(tag),coord_({Coord1,Coord2}) 
    {
      static_assert(Topology::InPlane<Dim>, "Using constructor for 2D node");
    } 
  
    Point(int tag, double Coord1, double Coord2, double Coord3):id_(tag),coord_({Coord1,Coord2,Coord3}) 
    {
      static_assert(Topology::InSpace<Dim>, "Using constructor for 3D node");
    } 
    
    ~Point(){} 

};



#endif