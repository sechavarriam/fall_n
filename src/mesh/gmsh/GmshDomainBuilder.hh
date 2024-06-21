#ifndef GMSH_DOMAIN_BUILDER_HH
#define GMSH_DOMAIN_BUILDER_HH

// Your code goes here
#include <string>
#include <string_view>
#include <memory>
#include <vector>

#include "../../domain/Node.hh"
#include "../../domain/elements/Element.hh"

#include "ReadGmsh.hh"
#include "GmshElementTypes.hh"

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