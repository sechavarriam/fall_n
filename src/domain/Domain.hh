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
#include "../mesh/Mesh.hh"

#include "../elements/Node.hh"

#include "../elements/element_geometry/ElementGeometry.hh"

#include "../mesh/gmsh/ReadGmsh.hh"
// #include "../mesh/gmsh/GmshDomainBuilder.hh"

template <std::size_t dim> requires topology::EmbeddableInSpace<dim>
class Domain
{ // Spacial (Phisical) Domain. Where the simulation takes place
    static constexpr std::size_t dim_ = dim;

    std::vector<Node<dim>>            nodes_   ;
    std::vector<ElementGeometry<dim>> elements_;

public:

    Mesh mesh;

    std::size_t num_nodes() const { return nodes_.size(); };
    std::size_t num_elements() const { return elements_.size(); };

    // Getters
    //Node<dim> *node_p(std::size_t i) { return &nodes_[i];};

    //Node<dim>& node  (std::size_t i) { return  nodes_[i];}; //Esto no es la iesima posicion!!!! debe devolver el nodo con id i.
                                                              //Esto se podria hacer asi si el vector de nodos fuera desde cero hasta el max id, y dejando espacios vacios.
                                                              //Otra opcion es almacenar un position_index en el nodo referente al dominio. Para llamar directamente sin buscar.
                                                              //Otra opcion es un unordered_map con los ids como keys.
                                                              //Por ahora se hara con un find_if, y luego se cambiara por alguna de las anteriores luego de un profiling.
    
    Node<dim> *node_p(std::size_t i) {
        // El iterador inicial podria tener un atajo si se ordenan los nodos por id.
        auto pos = std::find_if(nodes_.begin(), nodes_.end(), [&](auto &node){return node.id() == i;});
        
        std::cout << "Node id: " << pos->id() << std::endl;
        };


    Node<dim>& node  (std::size_t i) {
        // El iterador inicial podria tener un atajo si se ordenan los nodos por id.
        return *std::find_if(nodes_.begin(), nodes_.end(), [&](auto &node){return node.id() == i;});
        };

    std::span<Node<dim>>            nodes()    { return std::span<Node<dim>>           (nodes_);    };  
    std::span<ElementGeometry<dim>> elements() { return std::span<ElementGeometry<dim>>(elements_); };

    // ===========================================================================================================

    void link_nodes_to_elements(){
        for (auto &e : elements_){
            for (std::size_t i = 0; i < e.num_nodes(); i++){
                // Find position of node i in domain   
                auto pos = std::find_if(nodes_.begin(), nodes_.end(), [&](auto &node){return PetscInt(node.id()) == e.node(i);});
                e.bind_node(i, std::addressof(*pos));
            }
        
        // PONER UN MODEEEE!!!! Y UN CENTINELA!!! PARA PODER SACAR ESTO DE ACA Y USAR SOLO SI SE NECESITA!
        // MODE: 0 -> NO SE HACE NADA
        // MODE: 1 -> SE HACE ESTO
        e.set_integration_point_coordinates(); 
        }


    }

    void assemble_sieve() {

        link_nodes_to_elements();


        mesh.set_size(PetscInt(num_nodes() + num_elements())); // Number of DAG points = nodes + edges + faces + cells
                                                                // Uninterpoleated topology by now (no edges or faces).

        PetscInt sieve_point_idx = 0;

        for (auto &e : elements_){
            e.set_sieve_id(sieve_point_idx);
            mesh.set_sieve_cone_size(sieve_point_idx, e.num_nodes());
            ++sieve_point_idx;
        }

        for (auto &n : nodes_){
            n.set_sieve_id(sieve_point_idx);
            ++sieve_point_idx;
        }

        mesh.setup(sieve_point_idx);

        // Set the sieve cone for each entity (only elements by now). 
        // The nodes doesn't cover any entity of lower dimension.
        for (auto &e : elements_){
            std::vector<PetscInt> cone(e.num_nodes());

            for (std::size_t i = 0; i < e.num_nodes(); i++){
                cone[i] = e.node_p(i).sieve_id.value();
            }
            mesh.set_sieve_cone(e.sieve_id.value(), cone.data());
        }
        mesh.symmetrize_sieve();
    }

    template <typename ElementType, typename IntegrationStrategy>
    void make_element(IntegrationStrategy &&integrator, std::size_t tag, PetscInt node_ids[]){
        elements_.emplace_back(
            ElementGeometry<dim>(
                ElementType( // Forward this?
                    std::forward<std::size_t>(tag),
                    std::span<PetscInt>(node_ids, ElementType::num_nodes())),
                std::forward<IntegrationStrategy>(integrator)));
    }
    
    Node<dim> *add_node(Node<dim> &&node){
        nodes_.emplace_back(std::forward<Node<dim>>(node));
        return &nodes_.back();
    };

    // Tol increases capacity by default in 20%.
    void preallocate_node_capacity(std::size_t n, double tol = 1.20)
    { // Use Try and Catch to allow this operation if the container is empty.
        try{
            if (nodes_.empty())
                nodes_.reserve(n * tol);
            else
                throw nodes_.empty();
        }catch (bool NotEmpty){
            std::cout << "Preallocation should be done only before any node definition. Doing nothing." << std::endl;
        }
    };

    // Constructors
    // Copy Constructor
    Domain(const Domain &other) = default;
    // Move Constructor
    Domain(Domain &&other) = default;
    // Copy Assignment
    Domain &operator=(const Domain &other) = default;
    // Move Assignment
    Domain &operator=(Domain &&other) = default;

    Domain() = default;
    ~Domain() = default;
};

#endif


    //template <typename ElementType, typename IntegrationStrategy>
    //void make_element(IntegrationStrategy &&integrator, std::size_t &&tag, std::vector<Node<3> *> nodeAdresses)
    //{
    //    elements_.emplace_back(
    //        ElementGeometry<dim>(
    //            ElementType( // Forward this?
    //                std::forward<std::size_t>(tag),
    //                std::forward<std::vector<Node<3> *>>(nodeAdresses)),
    //            std::forward<IntegrationStrategy>(integrator)));
    //}