#ifndef GMSH_DOMAIN_BUILDER_HH
#define GMSH_DOMAIN_BUILDER_HH

// Your code goes here
#include <string>
#include <string_view>
#include <memory>
#include <vector>
#include <algorithm>
#include <ranges>

#include "../../domain/Domain.hh"
#include "../../domain/Node.hh"
#include "../../domain/elements/ElementGeometry.hh"

#include "../../numerics/numerical_integration/CellQuadrature.hh"

#include "ReadGmsh.hh"
#include "GmshElementTypes.hh"

class GmshDomainBuilder
{
    using Domain3D = Domain<3>;
    // using Node3D = Node<3>;
    using GmshMesh = gmsh::MSHReader;

    GmshMesh mesh_info_;
    Domain3D *domain_;

    std::vector<Node<3> *> node_addresses_;

public:
    void aggregate_nodes()
    {
        domain_->preallocate_node_capacity(mesh_info_.nodes_info_.numNodes); // IMPORTANT!!!!

        node_addresses_.reserve(mesh_info_.nodes_info_.numNodes);

        for (auto &block : mesh_info_.nodes_info_.entityBlocks)
        {
            for (std::size_t node = 0; node < block.numNodesInBlock; node++)
            {
                node_addresses_.push_back( // Stores the address of the node created.
                    domain_->add_node(     // Create Node in the domain
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
        for (auto &block : mesh_info_.elements_info_.entityBlocks)
        {          
            if (block.entityDim == 3)
            { // Only 3D elements (by now...)

                for (auto &[element_tag, node_tags] : block.elementTags)
                {
                    for (auto &tag : node_tags) std::cout << tag << " ";
                    std::cout << std::endl;

                    std::vector<Node<3> *> elem_nodes;
                    elem_nodes.reserve(node_tags.size());

                    for (auto &tag : node_tags)
                    {
                        if (node_addresses_[tag - 1]->id() == tag)
                        {
                            elem_nodes.push_back(node_addresses_[tag-1]);
                        }
                        else
                        { // TODO: search for the node with the current tag.
                            //throw std::runtime_error("Node tag does not match with node id");
                            // Range search
                            auto it = std::ranges::find_if(domain_->nodes(), [tag](Node<3>  &node) { return node.id() == tag; });
                            if (it != domain_->nodes().end())
                            {
                                elem_nodes.push_back(std::addressof(*it));
                                std::cout << "Node found in the range" << std::endl;
                            }
                            else
                            {
                                std::cout << "Node not found in the range" << std::endl;
                            }
                        }
                    }
                    //std::cout << "°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°" << std::endl;
                    // Emplace Element
                    switch (block.elementType)
                    {
                    case 5:
                        {
                        auto integrator = GaussLegendreCellIntegrator<1,1,1>{};   
                        domain_->make_element<LagrangeElement<2,2,2>, decltype(integrator)>(
                            std::move(integrator),
                            std::size_t(element_tag),
                            {
                                elem_nodes[6], elem_nodes[2], elem_nodes[4], elem_nodes[0],
                                elem_nodes[7], elem_nodes[3], elem_nodes[5], elem_nodes[1]
                            }
                          );
                        break;
                        }
                    default:
                        {
                        //throw std::runtime_error("Element type not supported");
                        std::cout << "Element type "<< block.elementType <<" not supported" << std::endl;
                        break;
                        }
                    }
                }
            }
        }
    };

    GmshDomainBuilder(std::string_view filename, Domain3D &domain) : mesh_info_(filename), domain_(std::addressof(domain))
    {
        aggregate_nodes();
        aggregate_elements();
    };
};



#endif // GMSH_DOMAIN_BUILDER_HH