#ifndef FN_LINE_ELEMENT
#define FN_LINE_ELEMENT

#include "../Node.h"
#include "StructuralElement.h"

/*Line elements or 1D elements have an initial node ni, a final node nk and 
  num_nodes-2 optional internal nodes nk*/

template<u_short Dim, u_short nNodes, u_short nDoF>
class LineElement: virtual public StructuralElement<Dim,nNodes,nDoF>{


 private:
    static const int topo_dim = 1; // Topological dimension
                                   // 1D for line elements
                                   // 2D for surface elements

    Node<Dim>* ni; //First node
    Node<Dim>* nj; //Last node

    //double xi, xj, yi, yj ; 

 protected:

    //virtual void set_num_nodes(int n){this->set_num_nodes(int n);};

     LineElement(){};
     //LineElement(int tag, Node<Dim>** nodes): StructuralElement<Dim,nNodes,nDoF>(tag,nodes){}
     LineElement(int tag, std::array<u_int,nNodes> NodeTAGS): StructuralElement<Dim,nNodes,nDoF>(tag,NodeTAGS){}


 public:
    virtual ~LineElement(){};

};

#endif