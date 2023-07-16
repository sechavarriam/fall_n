
#ifndef FALL_N_ELEMENT_H
#define FALL_N_ELEMENT_H

//#include "../../numeric_utils/Matrix.h" 
//typedef Matrix<double,2> Mat2D;

#include <vector>
#include <iostream> // Header that defines the standard input/output stream objects.


// Forward Declarations
class Node;
template<typename, int> class Matrix;

//template<typename> class Transformation{}; //Not defined yet.


//template<typename T>
class Element{
    
 private:

    unsigned int id; //tag    
    int dim;

    
    Node**  node; //Pointer to array of Node pointers.

    unsigned int num_nodes; //Común a cada clase de elemento.
    
    /*
    En vez de punteros podría ser solo una lista de índices?
        int const n_nodes;
        int nodeTags[n_nodes]

    O definir un genérico!!!? 
        T** node; //?
    */

    double measure; // Length: for topological 1D element (like truss or beam).
                    // Area  : for topological 2D element (like shell or plate).
                    // Volume: for topological 3D element (like brik element).

    Matrix<double, 2>* K  ;// Stiffness matrix

    protected:

    // Protected constructors.
    // This is an abstract class. Pure elements should not be constructed.
    Element(){};
    //Element(int tag, Node** nodes): id(tag),node(nodes){};
    virtual ~Element(){};

    virtual void set_num_nodes(unsigned int n){this->num_nodes = n;};

    public:

    Element(int tag, Node** nodes): id(tag),node(nodes){}; // Here only for test

    // Implementations to be defined in separate .cxx file.
    virtual int tag(){return id;};

    //virtual Matrix<double, 2>* set_K();
    

};


#endif