
#ifndef FN_ELEMENTBASE_H
#define FN_ELEMENTBASE_H


#include <vector>
#include <array>
#include <iostream> // Header that defines the standard input/output stream objects.
#include <concepts>

#include <Eigen/Dense>

#include "../Topology.h"

#include "Element.h"
#include "../Node.h"


typedef unsigned short u_short;
typedef unsigned int   u_int  ;


/*
Every element type must have define:
     i) His spacial Dimension (Dim)
    ii) The number of nodes   (nNodes)
   iii) The number of nodes   (Dim)
This has to be known at compile time.
*/

template<u_short Dim, u_short nNodes, u_short nDoF> requires Topology::EmbeddableInSpace<Dim> 
class ElementBase: protected Element{
  public:
  
  private: 

    //https://stackoverflow.com/questions/11134497/constant-sized-vector
    std::array<u_int,nNodes> nodes_index_;   

    Node<Dim>*  nodes_[nNodes];
    
    //Node<Dim>**  node_; //Pointer to array of Node pointers.
    
    //static constexpr unsigned int num_nodes_ = nNodes    ; //Común a cada clase de elemento.
    //unsigned int num_dof_ = nDoF; //const static in each subclass? 
    
    // Eigen Matrix. Initialized in zero by default static method to avoid garbage values.
    Eigen::Matrix<double, nDoF, nDoF> K_ = Eigen::Matrix<double, nDoF, nDoF>::Zero();

    //virtual Node<Dim>** node(){return node_;}
    //virtual void set_num_nodes(unsigned int n){this->num_nodes_ = n;};
    
    virtual void set_K(Eigen::Matrix<double, nDoF, nDoF> mat){
        this->K_ = mat;
    };
    
    
    virtual void compute_K(){};

  public:

  
    //virtual int num_dof(){return this->num_dof_;};   
    //virtual double measure(){return this->measure_;};  
  
  //protected:
    // This is an abstract base class. Pure elements should not be constructed.
    ElementBase(){};

    ElementBase(int tag, Node<Dim>** nodes): Element(tag), nodes_(nodes){};
    
    ElementBase(int tag, std::array<u_int,nNodes> NodeTAGS): Element(tag), nodes_index_(NodeTAGS){};
    
    
    virtual ~ElementBase(){};
};

#endif