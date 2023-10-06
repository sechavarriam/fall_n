#ifndef FN_DOMAIN
#define FN_DOMAIN


#include <functional>
#include <utility>
#include <optional>
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
typedef unsigned short ushort;
typedef unsigned int   uint  ;

template<ushort Dim> requires Topology::EmbeddableInSpace<Dim>
class Domain{ //Spacial Domain. Where the simulation takes place

    //friend class Node<Dim>;//?
    //friend class Element;  //?

    uint num_nodes_{0};
    uint num_elements_{0};
    
    public:

    std::vector<Node<Dim>> nodes_     ; //Could have an init preallocating parameter for eficiency!
    std::vector<double>    dof_vector_;

  public:

    std::vector<std::unique_ptr<Element>> elements_  ; 

    // ===========================================================================================================
    template<typename ElementType, typename ...Args >
    void make_element(uint&& tag,std::array<ushort,ElementType::num_Nodes>&& nodeTags,Args&&... constructorArgs){
        elements_.emplace_back(
            std::make_unique<ElementType>(
                std::forward<int>(tag),
                std::forward<std::array<ushort,ElementType::num_Nodes>>(nodeTags),
                std::forward<Args>(constructorArgs)...)
        );
    };
    template<typename ElementType, typename... Args >
    void make_element(Args&&... args){
        elements_.emplace_back( 
            std::make_unique<ElementType>(std::forward<Args>(args)...)
            );
    };
    



    void add_node(Node<Dim>&& node){nodes_.emplace_back(std::forward<Node<Dim>>(node));};
    
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
    

    // TODO: Preallocated constructor.
    //Domain(uint estimatedNodes, estimatedElements,estimatedDofs){};
    //
    Domain(){};
    ~Domain(){};



};



#endif

