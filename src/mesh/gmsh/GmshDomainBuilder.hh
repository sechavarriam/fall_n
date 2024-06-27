#ifndef GMSH_DOMAIN_BUILDER_HH
#define GMSH_DOMAIN_BUILDER_HH

// Your code goes here
#include <string>
#include <string_view>
#include <memory>
#include <vector>

#include "../../domain/Domain.hh"
#include "../../domain/Node.hh"
#include "../../domain/elements/Element.hh"

#include "../../numerics/numerical_integration/CellQuadrature.hh"

#include "ReadGmsh.hh"
#include "GmshElementTypes.hh"

class GmshDomainBuilder
{
    using Domain3D = Domain<3>;
    //using Node3D = Node<3>;
    using GmshMesh = gmsh::MSHReader;

    Domain3D *domain_;
    GmshMesh mesh_info_;

    std::vector<Node<3> *> node_addresses_;

public:
    void aggregate_nodes()
    {
        node_addresses_.reserve(mesh_info_.nodes_info_.numNodes);
        for (auto &block : mesh_info_.nodes_info_.entityBlocks)
        {
            for (std::size_t node = 0; node < block.numNodesInBlock; node++)
            {
                node_addresses_.push_back(//Stores the address of the node created.
                    domain_->add_node(    //Create Node in the domain
                        Node<3>(
                            std::move(block.nodeTag[node]),
                            std::move(block.coordinates[node][0]),
                            std::move(block.coordinates[node][1]),
                            std::move(block.coordinates[node][2]))));
            }
        }
    };

    void aggregate_elements()
    { // Esto procesa tanto elemento 2d (facets) como 3d  (cells) TODO:FIX.
        auto default_integration_scheme = [](auto const &e) { /**integrate*/ };
        for (auto &block : mesh_info_.elements_info_.entityBlocks)
        {
            if (block.entityDim == 3)
            { // Only 3D elements (by now...)
                for (auto &[element_tag, node_tags] : block.elementTags)
                {
                    std::vector<Node<3>*> element_node_pointers;
                    element_node_pointers.reserve(node_tags.size());

                    for (auto &tag : node_tags)
                    {
                         if (node_addresses_[tag - 1]->id() == tag)
                         {
                             element_node_pointers.push_back(node_addresses_[tag - 1]);
                         }
                         else
                         { // TODO: search for the node with the current tag.
                             throw std::runtime_error("Node tag does not match with node id");
                         }
                     }
                    // Emplace Element
                     switch (block.elementType)
                    {
                     case 5:
                         domain_->make_element<LagrangeElement<2,2,2>, GaussLegendre::CellQuadrature<1,1,1>>(
                             GaussLegendre::CellQuadrature<1,1,1>{},
                             std::forward<std::size_t>(std::remove_const_t<decltype(element_tag)>(element_tag)),
                             [&element_node_pointers]() -> std::array<Node<3>*, 8> {
                                 return {element_node_pointers[0], element_node_pointers[1], element_node_pointers[2], element_node_pointers[3],
                                         element_node_pointers[4], element_node_pointers[5], element_node_pointers[6], element_node_pointers[7]};
                             }());
                    
                     default:
                         throw std::runtime_error("Element type not supported");
                         break;
                     }
                }
            }
        }
    };

    GmshDomainBuilder(std::string_view filename, Domain3D& domain) : mesh_info_(filename), domain_(std::addressof(domain))
    {
        aggregate_nodes();
        aggregate_elements();
    };
};

class GmshDomainBuilder_3D
{
    using Node3D = Node<3>;
    gmsh::MSHReader mesh_info_;

public:
    std::vector<Node3D> nodes_;
    std::vector<Element> elements_;

    void process_nodes()
    {
        for (auto &block : mesh_info_.nodes_info_.entityBlocks)
        {
            for (std::size_t node = 0; node < block.numNodesInBlock; node++)
            {
                nodes_.emplace_back(
                    Node3D(
                        std::move(block.nodeTag[node]),
                        std::move(block.coordinates[node][0]),
                        std::move(block.coordinates[node][1]),
                        std::move(block.coordinates[node][2])));
            }
        }
    };

    void process_elements()
    { // Esto procesa tanto elemento 2d (facets) como 3d  (cells) TODO:FIX.
        elements_.reserve(mesh_info_.elements_info_.numElements);

        auto default_integration_scheme = [](auto const &e) { /**integrate*/ };

        for (auto &block : mesh_info_.elements_info_.entityBlocks)
        {
            if (block.entityDim == 3)
            { // Only 3D elements (by now...)
                for (auto &[element_tag, node_tags] : block.elementTags)
                {
                    std::vector<Node3D *> element_node_pointers;
                    element_node_pointers.reserve(node_tags.size());

                    for (auto &tag : node_tags)
                    {
                        if (nodes_[tag - 1].id() == tag)
                        {
                            element_node_pointers.push_back(&nodes_[tag - 1]);
                        }
                        else
                        { // TODO: search for the node with the current tag.
                            throw std::runtime_error("Node tag does not match with node id");
                        }
                    }
                    // Emplace Element
                    switch (block.elementType)
                    {
                    case 5:
                        elements_.emplace_back(
                            Element(
                                LagrangeElement<2, 2, 2>{
                                    std::move(element_tag),
                                    std::move(element_node_pointers)},
                                default_integration_scheme));
                        break;

                    default:
                        throw std::runtime_error("Element type not supported");
                        break;
                    }
                }
            }
        }
    };

    GmshDomainBuilder_3D(std::string_view filename) : mesh_info_(filename)
    {
        process_nodes();
        process_elements();
    };
};

#endif // GMSH_DOMAIN_BUILDER_HH