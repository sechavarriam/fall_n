
#ifndef FN_ELEMENT_H
#define FN_ELEMENT_H

#include <vector>
#include <iostream> // Header that defines the standard input/output stream objects.
#include <concepts>

#include <Eigen/Dense>

#include "../Topology.h"
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
class Element{
  public:
    static constexpr unsigned int dim  = Dim;
    //static u_short num_dof = nDoF; //Run time...
  
  private:
    int id_ ; //tag    

    std::size_t* node_positions;
    Node<Dim>**  node_; //Pointer to array of Node pointers.
    
    //static constexpr unsigned int num_nodes_ = nNodes    ; //Común a cada clase de elemento.
    //unsigned int num_dof_ = nDoF; //const static in each subclass? 
    
    double measure_ = 0; // Length: for topological 1D element (like truss or beam).
                         // Area  : for topological 2D element (like shell or plate).
                         // Volume: for topological 3D element (like brik element).

    // Eigen Matrix. Initialized in zero by default static method to avoid garbage values.
    Eigen::Matrix<double, nDoF, nDoF> K_ ;//= Eigen::Matrix<double, nDoF, nDoF>::Zero();

  protected:

    void set_id (int t){id_=t;}
    void set_tag(int t){id_=t;}

    //virtual Node<Dim>** node(){return node_;}
    virtual void set_num_nodes(unsigned int n){this->num_nodes_ = n;};
    
    virtual void set_K(Eigen::Matrix<double, nDoF, nDoF> mat){
        this->K_ = mat;
    };
    
    virtual void compute_measure(){};
    virtual void compute_K(){};

  public:

    virtual int id() {return id_;};
    virtual int tag(){return id_;};
    virtual int num_dof(){return this->num_dof_;};   
    
    virtual double measure(){return this->measure_;};  
  
  protected:
    // This is an abstract base class. Pure elements should not be constructed.
    Element(){};
    Element(int tag, Node<Dim>** nodes): id_(tag),node_(nodes){};
    virtual ~Element(){};
};

#endif