#ifndef FN_NODE
#define FN_NODE

#include <cmath>

#include <iostream>
#include <memory> 
#include <concepts>

#include "DoF.h"
#include "../geometry/Point.h"
#include "../geometry/Topology.h"

typedef unsigned short ushort;
typedef unsigned int   uint  ;

template<ushort Dim, ushort nDoF=Dim> 
class Node : public geometry::Point<Dim>{

 private:

    uint id_ ; 

 // DOF?

 public:
    Node(){}; //Private con Friend Domain? Para que sean solo construibles por el dominio?    
    //Node(int tag, double Coord1, double Coord2): id_(tag), Point<2>{Coord1,Coord2}{} 
    //Node(int tag, double Coord1, double Coord2, double Coord3): id_(tag), Point<3>(Coord1,Coord2,Coord3){} 

    // forwardeing constructor
    template<typename... Args>
    Node(int tag, Args&&... args) : id_(tag), geometry::Point<Dim>(std::forward<Args>(args)...){}
    
    ~Node(){} 

};



#endif