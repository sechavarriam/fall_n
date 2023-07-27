#ifndef FN_LINE_ELEMENT
#define FN_LINE_ELEMENT

#include "../Node.h"
#include "StructuralElement.h"

/*Line elements or 1D elements have an initial node ni, a final node nk and 
  num_nodes-2 optional internal nodes nk*/

template<unsigned int Dim, unsigned int nDoF>
class LineElement: virtual public StructuralElement<Dim,nDoF>{


 private:
    static const int topo_dim = 1; // Topological dimension
                                   // 1D for line elements
                                   // 2D for surface elements

    Node<Dim>* ni;
    Node<Dim>* nj;

    //double xi, xj, yi, yj ; 

 protected:

    //virtual void set_num_nodes(int n){this->set_num_nodes(int n);};

     LineElement(){};
     LineElement(int tag, Node<Dim> **nodes): StructuralElement<Dim,nDoF>(tag,nodes){}

 public:
    virtual ~LineElement(){};

};

#endif