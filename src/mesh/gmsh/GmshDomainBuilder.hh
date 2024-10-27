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
#include "../../elements/Node.hh"

#include "../../elements/element_geometry/ElementGeometry.hh"

#include "../../numerics/numerical_integration/CellQuadrature.hh"

#include "ReadGmsh.hh"
#include "GmshElementTypes.hh"

class GmshDomainBuilder
{
    using Domain3D = Domain<3>;
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
                    
                    /* // Deprecates adderess version
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
                          // throw std::runtime_error("Node tag does not match with node id");
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
                    */

                    //std::cout << "°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°" << std::endl;
                    // Emplace Element
                    switch (block.elementType)
                    {
                    case 5:
                        {
                        auto integrator = GaussLegendreCellIntegrator<2,2,2>{}; 
                           
                        domain_->make_element<LagrangeElement<2,2,2>, decltype(integrator)>(
                            std::move(integrator),
                            std::size_t(element_tag),
                            std::array{
                                PetscInt(node_tags[0]), PetscInt(node_tags[1]), PetscInt(node_tags[4]), PetscInt(node_tags[5]),
                                PetscInt(node_tags[3]), PetscInt(node_tags[2]), PetscInt(node_tags[7]), PetscInt(node_tags[6])
                            }.data()
                            //{
                            //    elem_nodes[0], elem_nodes[1], elem_nodes[4], elem_nodes[5],
                            //    elem_nodes[3], elem_nodes[2], elem_nodes[7], elem_nodes[6]
                            //}
                          );
                        break;
                        }
                    case 12:
                        {
                        auto integrator = GaussLegendreCellIntegrator<3,3,3>{}; 
                           
                        domain_->make_element<LagrangeElement<3,3,3>, decltype(integrator)>(
                            std::move(integrator),
                            std::size_t(element_tag),
                            //{
                            //    elem_nodes[ 5], elem_nodes[13], elem_nodes[ 4], elem_nodes[14], elem_nodes[21], elem_nodes[12], elem_nodes[ 1], elem_nodes[ 8], elem_nodes[ 0],
                            //    elem_nodes[19], elem_nodes[25], elem_nodes[15], elem_nodes[24], elem_nodes[26], elem_nodes[22], elem_nodes[11], elem_nodes[20], elem_nodes[ 9],
                            //    elem_nodes[ 7], elem_nodes[17], elem_nodes[ 6], elem_nodes[18], elem_nodes[23], elem_nodes[16], elem_nodes[ 3], elem_nodes[10], elem_nodes[ 2]
                            //}
                            std::array{
                                PetscInt(node_tags[ 5]), PetscInt(node_tags[13]), PetscInt(node_tags[ 4]), PetscInt(node_tags[14]), PetscInt(node_tags[21]), PetscInt(node_tags[12]), PetscInt(node_tags[ 1]), PetscInt(node_tags[ 8]), PetscInt(node_tags[ 0]),
                                PetscInt(node_tags[19]), PetscInt(node_tags[25]), PetscInt(node_tags[15]), PetscInt(node_tags[24]), PetscInt(node_tags[26]), PetscInt(node_tags[22]), PetscInt(node_tags[11]), PetscInt(node_tags[20]), PetscInt(node_tags[ 9]),
                                PetscInt(node_tags[ 7]), PetscInt(node_tags[17]), PetscInt(node_tags[ 6]), PetscInt(node_tags[18]), PetscInt(node_tags[23]), PetscInt(node_tags[16]), PetscInt(node_tags[ 3]), PetscInt(node_tags[10]), PetscInt(node_tags[ 2])
                            }.data()
                            //std::array{
                            //    node_tags[ 5], node_tags[13], node_tags[ 4], node_tags[14], node_tags[21], node_tags[12], node_tags[ 1], node_tags[ 8], node_tags[ 0],
                            //    node_tags[19], node_tags[25], node_tags[15], node_tags[24], node_tags[26], node_tags[22], node_tags[11], node_tags[20], node_tags[ 9],
                            //    node_tags[ 7], node_tags[17], node_tags[ 6], node_tags[18], node_tags[23], node_tags[16], node_tags[ 3], node_tags[10], node_tags[ 2]
                            //}.data()
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

        // sort nodes with respect to their id
        // only if node containter is std vector. In is unordered_map, no need to sort.
        std::sort(domain_->nodes().begin(), domain_->nodes().end(), 
        [](Node<3> &a, Node<3> &b) {
             return a.id() < b.id(); 
             }
        );

        domain_->assemble_sieve(); // Assemble sieve (DAG) for the domain (DMPlex)


    };
};



#endif // GMSH_DOMAIN_BUILDER_HH