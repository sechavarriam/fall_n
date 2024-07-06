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
#include <concepts>

#include "../geometry/Topology.hh"
#include "Node.hh"
#include "elements/ElementBase.hh"

#include "../mesh/gmsh/ReadGmsh.hh"
//#include "../mesh/gmsh/GmshDomainBuilder.hh"

#include "MaterialPoint.hh"

//namespace domain
//{
    template <std::size_t dim> requires topology::EmbeddableInSpace<dim>
    class Domain
    { // Spacial (Phisical) Domain. Where the simulation takes place

        static constexpr std::size_t dim_ = dim;

        std::size_t num_nodes_{0};
        std::size_t num_elements_{0};

        std::vector<Node<dim>>          nodes_;
        std::vector<Element>            elements_;
        std::vector<MaterialPoint<dim>> material_points_;

        //std::vector<Material> materials_;

    public:
        std::size_t num_nodes() const { return num_nodes_; };

        // Getters
        Node<dim> *node_p(std::size_t i) { return &nodes_[i]; };
        // Node<dim>  node  (std::size_t i){return nodes_[i];};
        

        std::span<Node<dim>> nodes() { return std::span<Node<dim>>(nodes_); };
        std::span<Element> elements(){ return std::span<Element>(elements_); };

        // ===========================================================================================================

        //template <typename ElementType, typename IntegrationStrategy>
        //void make_element(IntegrationStrategy integrator,std::size_t tag, std::vector<Node<3>*> nodeAdresses)
        //{
        //    elements_.emplace_back(
        //        Element(ElementType(tag,nodeAdresses),integrator));
        //}

        template <typename ElementType, typename IntegrationStrategy>
        void make_element(IntegrationStrategy&& integrator, std::size_t&& tag, std::vector<Node<3>*> nodeAdresses)
        {
            elements_.emplace_back(
                Element(
                    ElementType(
                        std::forward<std::size_t>(tag),
                        std::forward<std::vector<Node<3>*>>(nodeAdresses)),
                    std::forward<IntegrationStrategy>(integrator)));
        }   

        //template <typename ElementType, typename IntegrationStrategy>
        //void make_element(
        //    IntegrationStrategy &&integrator,
        //    std::size_t && tag,
        //    std::ranges::range auto&& nodeAdresses)
        //{
        //    elements_.emplace_back(
        //        Element(
        //            ElementType{
        //                std::move(tag),
        //                std::forward<std::ranges::range auto>(nodeAdresses)
        //                },
        //            std::forward<IntegrationStrategy>(integrator)
        //            ));
        //}

        // template<typename ElementType,typename... Args>
        // void make_element(Args&&... args){
        //     elements_.emplace_back(ElementType{std::forward<Args>(args)...});
        // };


        //Node<dim>* add_node(std::size_t id, double&&... coords)
        //{
        //    nodes_.emplace_back(Node<dim>(id, std::forward<double>(coords)...));
        //    ++num_nodes_;
        //    return std::addressof(nodes_.back());
        //};


        Node<dim>* add_node(Node<dim> &&node)
        {
            nodes_.emplace_back(std::forward<Node<dim>>(node));
            ++num_nodes_;
            return &nodes_.back();
        };

        void set_material_point(std::size_t i, MaterialPoint<dim> &&mp)
        {
            material_points_[i] = std::forward<MaterialPoint<dim>>(mp);
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
