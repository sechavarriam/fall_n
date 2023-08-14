#ifndef FN_DOMAIN
#define FN_DOMAIN



#include <format>
#include <iostream>  
#include <memory>
#include <vector>

#include "Topology.h"
#include "Node.h"
#include "elements/ElementBase.h"


// Templatizar tipo contenedor de nodos y elementos usado concepto Is_Container?
// Template dim?

//Contar cuantos nodos hay. y con esto separar un espacio en memoria con algun contenedor.
//Los elementos guardarían solo en índice en el arreglo en vez de un puntero al nodo?

template<unsigned short Dim> requires Topology::EmbeddableInSpace<Dim>
class Domain{ //Spacial Domain. Where the simulation takes place

    friend class Node<Dim>;
    friend class Element; //?

    u_int num_nodes_;
    u_int num_elements_;

    //unique_ptr<int> p = make_unique<int>(42);

    std::vector<Node<Dim>> nodes_     ; //Could have an init preallocating parameter for eficiency!
    std::vector<std::unique_ptr<Element>>  elements_ ; 
  public:

    //Node<Dim>* node(u_int i){return &nodes_[i] ;}; 

    template<typename ElementType, typename... Args >
    void add_element(Args... constructorArgs){
        elements_.emplace_back(
            std::make_unique<ElementType>(constructorArgs... )
        );
    };


    //https://stackoverflow.com/questions/23717151/why-emplace-back-is-faster-than-push-back
    //void add_node(Node<Dim> x){nodes_.push_back(x);};

    //Constructs the node directly in the container
    void add_node(Node<Dim> node){nodes_.emplace_back(node);}; 
    
    // Se usa push_back porque el elemento no se crea en el arreglo. Se crea el puntero.
    



    // https://cplusplus.com/reference/vector/vector/capacity/
    // https://cplusplus.com/reference/vector/vector/reserve/
    // Tol increases capacity by default in 20%.
    void preallocate_node_capacity(int n, double tol=1.20){
    // Use Try and Catch to allow this operation if the container is empty.
        try {
            if(nodes_.empty()) nodes_.reserve((int) n*tol);
            else throw nodes_.empty();
        } catch (bool NotEmpty) {
            std::cout << "Preallocation should be done only before any node definition. Doing nothing." << std::endl;
        }     
    };
    



    Domain(){};


    ~Domain(){};



};



#endif
























