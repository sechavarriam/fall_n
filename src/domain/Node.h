#ifndef FN_NODE
#define FN_NODE

#include <cmath>

#include <iostream> // Header that defines the standard input/output stream objects.
#include <concepts>

#include <Eigen/Dense>


//#include "Domain.h"
#include "Point.h"
#include "Topology.h"

typedef unsigned short ushort;
typedef unsigned int   uint  ;

template<ushort Dim, ushort nDoF=Dim> 
class Node : public Point<Dim>{
 public:
    static constexpr unsigned int dim = Dim; // Dimention (2 or 3). Its topological dimension is 0

 private:

 public:

    Node(){}; //Private con Friend Domain? Para que sean solo construibles por el dominio?    
    Node(int tag, double Coord1, double Coord2):Point<2>(tag,Coord1,Coord2)
    {
      static_assert(Topology::InPlane<Dim>, "Using constructor for 2D node");
    } 
  
    Node(int tag, double Coord1, double Coord2, double Coord3):Point<3>(tag,Coord1,Coord2,Coord3)
    {
      static_assert(Topology::InSpace<Dim>, "Using constructor for 3D node");
      std::cout << "Construido Nodo 3D: " << tag << "\n"; 
    } 
    
    ~Node(){} 

};



#endif