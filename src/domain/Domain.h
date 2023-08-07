

#include <iostream>  
#include <vector>

#include "Topology.h"
#include "Node.h"
#include "elements/Element.h"


// Templatizar tipo contenedor de nodos y elementos usado concepto Is_Container?
// Template dim?

//Contar cuantos nodos hay. y con esto separar un espacio en memoria con algun contenedor.
//Los elementos guardarían solo en índice en el arreglo en vez de un puntero al nodo?

template<unsigned int Dim> requires Topology::EmbeddableInSpace<Dim>
class Domain{ //Spacial Domain. Where the simulation takes place
    
    int num_nodes_;
    int num_elements_;

    std::vector<Node<Dim>> nodes_; //Could have an init preallocating parameter for eficiency!

    //std::vector<Element<Dim,2> > elements_; //Vector de punteros a clase base...
                                              //porque en sí puede contener varios tipos derivados

  public:

    //https://stackoverflow.com/questions/23717151/why-emplace-back-is-faster-than-push-back
    //void add_node(Node<Dim> x){nodes_.push_back(x);};
    void add_node(Node<Dim> x){nodes_.emplace_back(x);}; //Constructs the node directly in the container


    Domain(){};


    ~Domain(){};



};































