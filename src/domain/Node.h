#ifndef FN_NODE
#define FN_NODE

#include <cmath>

#include <iostream> 
#include <concepts>

#include <Eigen/Dense>

#include "DoF.h"
#include "Point.h"
#include "Topology.h"

typedef unsigned short ushort;
typedef unsigned int   uint  ;

template<ushort Dim, ushort nDoF=Dim> 
class Node : public Point<Dim>{

 private:

    //DoF<nDoF> DoF_; TODO.
    Eigen::Matrix<double, nDoF, 1> DoF_; // e.g. [u,v,w]  current state? Should have a containter for all times? Recorder...  
                                         // Should be a class itself? Maybe
 
 public:
    Node(){}; //Private con Friend Domain? Para que sean solo construibles por el dominio?    
    Node(int tag, double Coord1, double Coord2):Point<2>(tag,Coord1,Coord2)
    {
      static_assert(Topology::InPlane<Dim>, "Using constructor for 2D node");
    } 
  
    Node(int tag, double Coord1, double Coord2, double Coord3):Point<3>(tag,Coord1,Coord2,Coord3)
    {
      static_assert(Topology::InSpace<Dim>, "Using constructor for 3D node");
    } 
    
    ~Node(){} 

};



#endif