
#ifndef FALL_N_ELEMENT_H
#define FALL_N_ELEMENT_H


#include "../Node.h"
#include <vector>

class Element{
    
    private:
    
    int dim;
    int tag;

    double** K = NULL; // Stiffness matrix
    double** M = NULL;
    double** C = NULL;

    // Apuntar a nodo o coleccionar índices?
    //std::vector<Node> node;

    public:

    virtual int get_tag(){return tag;};



    Element(){};

};


#endif