#ifndef FN_DOMAIN
#define FN_DOMAIN


#include <cstddef>
#include <functional>
#include <utility>
#include <optional>
#include <format>
#include <iostream>  
#include <memory>
#include <vector>

#include "../geometry/Topology.hh"
#include "Node.hh"
#include "elements/ElementBase.hh"



namespace domain{

// Templatizar tipo contenedor de nodos y elementos usado concepto Is_Container?
// Template dim?

//Contar cuantos nodos hay. y con esto separar un espacio en memoria con algun contenedor.
//Los elementos guardarían solo en índice en el arreglo en vez de un puntero al nodo?
typedef unsigned short ushort;
typedef unsigned int   uint  ;

template<std::size_t Dim> //requires topology::EmbeddableInSpace<Dim>
class Domain{ //Spacial (Phisical) Domain. Where the simulation takes place

    static constexpr std::size_t dim = Dim;

    std::size_t num_nodes_{0};
    std::size_t num_elements_{0};
    
    std::vector<Node<dim>> nodes_     ; 
    std::vector<Element  > elements_  ; 

    


  public:

    // Getters
    Node<Dim>* node_p(std::size_t i){return &nodes_[i];};
    Node<Dim>  node  (std::size_t i){return  nodes_[i];};

    // ===========================================================================================================
    template<typename ElementType, typename IntegrationStrategy, typename ...Args>
    void make_element(IntegrationStrategy&& integrator, uint&& tag,std::array<ushort,ElementType::n_nodes>&& nodeTags,Args&&... constructorArgs){
        elements_.emplace_back(
            Element(
                ElementType{
                std::forward<uint>(tag),
                std::forward<std::array<ushort,ElementType::n_nodes>>(nodeTags),
                std::forward<Args>(constructorArgs)...
                },
                std::forward<IntegrationStrategy>(integrator)
            )
        );
    };

    //template<typename ElementType,typename... Args>
    //void make_element(Args&&... args){
    //    elements_.emplace_back(ElementType{std::forward<Args>(args)...});
    //};
    
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

} // namespace Domain

#endif

