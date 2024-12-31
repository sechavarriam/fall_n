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
            //std::cout << "Adding nodes from block: " << block.entityDim << std::endl;
            for (std::size_t node = 0; node < block.numNodesInBlock; node++)
            {
                //std::cout << "Node: " << block.nodeTag[node] << " " << block.coordinates[node][0] << " " << block.coordinates[node][1] << " " << block.coordinates[node][2] << std::endl;
                node_addresses_.push_back( // Stores the address of the node created.
                    domain_->add_node(     // Create Node in the domain
                        Node<3>(
                            block.nodeTag[node] -1 , // Node tag starts at 1, but the vector starts at 0.
                            block.coordinates[node][0],
                            block.coordinates[node][1],
                            block.coordinates[node][2])));
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
                    switch (block.elementType)
                    {
                    case 5:
                        {
                        auto integrator = GaussLegendreCellIntegrator<2,2,2>{}; 
                           
                        domain_->make_element<LagrangeElement<2,2,2>, decltype(integrator)>(
                            std::move(integrator),
                            std::size_t(element_tag),
                            std::array{
                                PetscInt(node_tags[0])-1, PetscInt(node_tags[1])-1, 
                                PetscInt(node_tags[4])-1, PetscInt(node_tags[5])-1,  //-----
                                PetscInt(node_tags[3])-1, PetscInt(node_tags[2])-1, 
                                PetscInt(node_tags[7])-1, PetscInt(node_tags[6])-1
                            }.data()
                          );
                        break;
                        }
                    case 12:
                        {
                        auto integrator = GaussLegendreCellIntegrator<3,3,3>{}; 
                           
                        domain_->make_element<LagrangeElement<3,3,3>, decltype(integrator)>(
                            std::move(integrator),
                            std::size_t(element_tag),
                            std::array{ //Esta numeración me garantiza un orden incremental en X,Y y Z para la numeración de los nodos.
                                PetscInt(node_tags[ 2])-1, PetscInt(node_tags[14])-1, PetscInt(node_tags[ 6])-1, 
                                PetscInt(node_tags[13])-1, PetscInt(node_tags[24])-1, PetscInt(node_tags[19])-1, 
                                PetscInt(node_tags[ 3])-1, PetscInt(node_tags[15])-1, PetscInt(node_tags[ 7])-1, //-----
                                PetscInt(node_tags[11])-1, PetscInt(node_tags[23])-1, PetscInt(node_tags[18])-1, 
                                PetscInt(node_tags[20])-1, PetscInt(node_tags[26])-1, PetscInt(node_tags[25])-1, 
                                PetscInt(node_tags[ 9])-1, PetscInt(node_tags[22])-1, PetscInt(node_tags[17])-1, //-----
                                PetscInt(node_tags[ 1])-1, PetscInt(node_tags[12])-1, PetscInt(node_tags[ 5])-1, 
                                PetscInt(node_tags[ 8])-1, PetscInt(node_tags[21])-1, PetscInt(node_tags[16])-1, 
                                PetscInt(node_tags[ 0])-1, PetscInt(node_tags[10])-1, PetscInt(node_tags[ 4])-1
                            }.data()
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