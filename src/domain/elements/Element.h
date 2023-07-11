
#ifndef FALL_N_ELEMENT_H
#define FALL_N_ELEMENT_H


//#include "../../numeric_utils/Matrix.h" 
//typedef Matrix<double,2> Mat2D;



#include <vector>
#include <iostream> // Header that defines the standard input/output stream objects.


// Forward Declarations
class Node;
template<typename, int> class Matrix;

//template(typename T)
//Matrix<double,2>


class Element{
    
    private:

    int id; //tag    
    int dim;

    Node**  node; // En vez de punteros podría ser solo una lista de índices

    //int const n_nodes;
    //int nodeTags[n_nodes];


    Matrix<double, 2>* K   ; // Stiffness matrix
    
    // Apuntar a nodo o coleccionar índices?
    //std::vector<Node> node;

    public:

    Element(){};

    Element(int tag, Node** nodes): id(tag),node(nodes){
        std::cout << "Elemento Generico Construido: " << tag << "\n"; 
    };

    ~Element(){};

    virtual int tag(){return id;};

    

};


#endif