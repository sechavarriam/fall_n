#ifndef FN_ELEMENTBASE_H
#define FN_ELEMENTBASE_H

#include <vector>
#include <array>
#include <iostream>
#include <concepts>

#include <Eigen/Dense>

#include "../Topology.h"

#include "Element.h"
#include "../Node.h"
#include "../IntegrationPoint.h"

typedef unsigned short ushort;
typedef unsigned int   uint  ;

/*
Every element type must have defined:
     i) His spacial Dimension ------------> (Dim)
    ii) The number of nodes --------------> (nNodes)
   iii) The number of DoF ----------------> (nDoF)
    iv) The number of integration Points -> (nGauss)
This has to be known at compile time.
*/

template<ushort Dim, ushort nNodes, ushort nDoF, ushort nGauss=0> 
requires Topology::EmbeddableInSpace<Dim> 
class ElementBase: public Element{
  public:
  
  private: 

    //https://stackoverflow.com/questions/11134497/constant-sized-vector
    std::array<uint      ,nNodes> nodes_index_;   
    std::array<Node<Dim>*,nNodes> nodes_                         ; // Fixed size?
    std::array<IntegrationPoint<Dim>,nGauss> integration_points_ ; // Fixed size?


    // Eigen Matrix. Initialized in zero by default static method to avoid garbage values.
    Eigen::Matrix<double, nDoF, nDoF> K_ = Eigen::Matrix<double, nDoF, nDoF>::Zero();
    
    //virtual void set_K(Eigen::Matrix<double, nDoF, nDoF> mat){
    //    this->K_ = mat;
    //};
    
  public:

    //virtual void compute_K(){};
    //virtual int num_dof(){return this->num_dof_;};   
    //virtual double measure(){return this->measure_;};  
  
  //protected:
    // This is an abstract base class. Pure elements should not be constructed.
    ElementBase(){};
    ElementBase(int tag, Node<Dim>** nodes): Element(tag), nodes_(nodes){};
    ElementBase(int tag, std::array<uint,nNodes> NodeTAGS): Element(tag), nodes_index_(NodeTAGS){};
    
    virtual ~ElementBase(){};
};

#endif