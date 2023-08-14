#ifndef FN_NODE
#define FN_NODE

#include <cmath>

#include <iostream> // Header that defines the standard input/output stream objects.
#include <concepts>

#include <Eigen/Dense>


//#include "Domain.h"
#include "Topology.h"

typedef unsigned short u_short;
typedef unsigned int   u_int  ;

template<u_short Dim, u_short nDoF=Dim> requires Topology::EmbeddableInSpace<Dim>
class Node {
 public:
    static constexpr unsigned int dim = Dim; // Dimention (2 or 3). Its topological dimension is 0

  private:

    u_int id_ ;
    
    Eigen::Matrix<double, Dim, 1> coord_; //Use of Eigen vector to facilitate operaitons.

  public:

    inline void set_id (int t){id_=t;}
    inline void set_tag(int t){id_=t;}
  
    inline int id (){return id_;}
    inline int tag(){return id_;}
  
    inline double* coord(int i){return &coord_[i];}

  //private:
    
    Node(){}; //Private con Friend Domain? Para que sean solo construibles por el dominio?    
    Node(int tag, double Coord1, double Coord2):id_(tag),coord_({Coord1,Coord2}) 
    {
      static_assert(Topology::InPlane<Dim>, "Using constructor for 2D node");
    } 
  
    Node(int tag, double Coord1, double Coord2, double Coord3):id_(tag),coord_({Coord1,Coord2,Coord3}) 
    {
      static_assert(Topology::InSpace<Dim>, "Using constructor for 3D node");
      std::cout << "Construido Nodo 3D: " << tag << "\n"; 
    } 
    
    ~Node(){} 

};



#endif