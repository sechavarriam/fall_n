#ifndef FN_INTEGRATION_POINT
#define FN_INTEGRATION_POINT

#include <iostream> // Header that defines the standard input/output stream objects.

#include <Eigen/Dense>



#include "Point.h"
#include "Topology.h"

typedef unsigned short ushort;
typedef unsigned int   uint  ;

template<ushort Dim> 
class IntegrationPoint : public Point<Dim>{

 private:


    //Eigen::Matrix<double, nDoF, 1> DoF_; // e.g. [u,v,w]  current state? Should have a containter for all times? Recorder...  
    //                                     // Should be a class itself? Maybe
 public:

    IntegrationPoint(){};  
    IntegrationPoint(int tag, double Coord1, double Coord2):Point<2>(tag,Coord1,Coord2)
    {
      static_assert(Topology::InPlane<Dim>, "Using constructor for 2D IntegrationPoint");
    } 
  
    IntegrationPoint(int tag, double Coord1, double Coord2, double Coord3):Point<3>(tag,Coord1,Coord2,Coord3)
    {
      static_assert(Topology::InSpace<Dim>, "Using constructor for 3D IntegrationPoint");
      std::cout << "Construido Nodo 3D: " << tag << "\n"; 
    } 
    
    ~IntegrationPoint(){} 

};



#endif