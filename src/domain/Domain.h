//
//
//
//
#include <iostream>  
#include <vector>

#include "Topology.h"
#include "Node.h"



// Templatizar tipo contenedor de nodos y elementos usado concepto Is_Container?
// Template dim?

//Contar cuantos nodos hay. y con esto separar un espacio en memoria con algun contenedor.
//Los elementos guardarían solo en índice en el arreglo en vez de un puntero al nodo?

template<unsigned int Dim>
class Domain{ //Spacial Domain. Where the simulation takes place
    
    int num_nodes_;
    int num_elements_;

    
    std::vector<Node<Dim>> nodes_;


  public:
    Domain(){};


    ~Domain(){};



};































