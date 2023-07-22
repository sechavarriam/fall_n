#ifndef STRUCTURAL_ELEMENT_H
#define STRUCTURAL_ELEMENT_H

#include "Element.h"

class StructuralElement: public Element{

 private:
    //static int topo_dim ; // Topological dimension
                          // 1D for line elements
                          // 2D for surface elements

 protected:
    //virtual void set_topo_dim(int n){this->topo_dim = n;};

    //virtual void set_num_nodes(int n){this->set_num_nodes(int n);};

     StructuralElement(){};
     StructuralElement(int tag, Node **nodes): Element(tag,nodes){}

 public:
    virtual ~StructuralElement(){};

};

#endif