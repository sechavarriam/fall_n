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
#include "../elements/Node.hh"

#include "../elements/element_geometry/ElementGeometry.hh"

#include "../mesh/gmsh/ReadGmsh.hh"
//#include "../mesh/gmsh/GmshDomainBuilder.hh"

//TODO: revisar! pyvista!

//namespace domain
//{
    template <std::size_t dim> requires topology::EmbeddableInSpace<dim>
    class Domain
    { // Spacial (Phisical) Domain. Where the simulation takes place

        static constexpr std::size_t dim_ = dim;

        std::size_t num_nodes_{0}   ;
        std::size_t num_elements_{0};

        std::vector<Node<dim>>             nodes_   ;
        std::vector<ElementGeometry<dim>>  elements_;
        
    public:
        std::size_t num_nodes() const { return nodes_.size(); };

        // Getters
        Node<dim> *node_p(std::size_t i) { return &nodes_[i]; };
        // Node<dim>  node  (std::size_t i){return nodes_[i];};
         

        std::span<Node<dim>>            nodes()   { return std::span<Node<dim>>      (nodes_   ); };
        std::span<ElementGeometry<dim>> elements(){ return std::span<ElementGeometry<dim>>(elements_); };

        // ===========================================================================================================

        template <typename ElementType, typename IntegrationStrategy>
        void make_element(IntegrationStrategy&& integrator, std::size_t&& tag, std::vector<Node<3>*> nodeAdresses)
        {
            elements_.emplace_back(
                ElementGeometry<dim>(
                    ElementType(  //Forward this?
                        std::forward<std::size_t>(tag),
                        std::forward<std::vector<Node<3>*>>(nodeAdresses)),
                    std::forward<IntegrationStrategy>(integrator)));
        }   

        Node<dim>* add_node(Node<dim> &&node)
        {
            nodes_.emplace_back(std::forward<Node<dim>>(node));
            ++num_nodes_;
            return &nodes_.back();
        };

        // Tol increases capacity by default in 20%.
        void preallocate_node_capacity(std::size_t n, double tol = 1.20)
        {
            // Use Try and Catch to allow this operation if the container is empty.
            try
            {
                if (nodes_.empty())
                    nodes_.reserve(n * tol);
                else
                    throw nodes_.empty();
            }
            catch (bool NotEmpty)
            {
                std::cout << "Preallocation should be done only before any node definition. Doing nothing." << std::endl;
            }
        };

        // TODO: Preallocated constructor.
        // Domain(uint estimatedNodes, estimatedElements,estimatedDofs){};
        //
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

//} // namespace Domain

#endif
