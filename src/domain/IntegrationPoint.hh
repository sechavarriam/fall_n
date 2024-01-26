#ifndef FN_INTEGRATION_POINT
#define FN_INTEGRATION_POINT

#include <iostream> 

#include "../geometry/Point.hh"
#include "../geometry/Topology.hh"

typedef unsigned short ushort;
typedef unsigned int   uint  ;

template<ushort Dim> 
class IntegrationPoint : public geometry::Point<Dim>{
  
    using Point = geometry::Point<Dim>;

  private:

  public:

    IntegrationPoint(){};  
    IntegrationPoint(int tag, double Coord1, double Coord2):Point(tag,Coord1,Coord2){} 
  
    IntegrationPoint(int tag, double Coord1, double Coord2, double Coord3):Point(tag,Coord1,Coord2,Coord3){} 
    
    ~IntegrationPoint(){} 

};



#endif