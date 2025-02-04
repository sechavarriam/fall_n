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

    GmshMesh  mesh_info_;
    Domain3D* domain_;

    std::vector<Node<3> *> node_addresses_;

    bool are_elements_aggregated_{false};

public:
    void aggregate_nodes(){
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
        auto index_ordering = [&](auto& tags, std::integral auto... i) {
            return std::array{static_cast<PetscInt>(tags[i] - 1)...};
             };

        for (auto &block : mesh_info_.elements_info_.entityBlocks){          
            if (block.entityDim == 3){ // Only 3D elements (by now...)
                for (auto &[element_tag, node_tags] : block.elementTags){
                    switch (block.elementType){
                    case 5:
                        {
                        auto integrator = GaussLegendreCellIntegrator<2,2,2>{}; 
                           
                        auto elem = domain_->make_element<LagrangeElement<2,2,2>, decltype(integrator)>(
                            std::move(integrator),
                            static_cast<std::size_t>(element_tag),
                            index_ordering(node_tags, 2, 6, 3, 7, 1, 5, 0, 4).data() 
                            //index_ordering(node_tags, 0, 1, 3, 2, 4, 5, 7, 6).data(),
                            //std::array{1,5,3,7,0,4,2,6}.data()
                            //std::array{1,5,3,7,0,4,2,6}.data() // LocalOrdering 
                          );
                        
                        //elem.set_local_index(std::array{1,5,3,7,0,4,2,6}.data());
                        break;
                        }
                    case 12:
                        {
                        auto integrator = GaussLegendreCellIntegrator<3,3,3>{}; 
                           
                        domain_->make_element<LagrangeElement<3,3,3>, decltype(integrator)>(
                            std::move(integrator),
                            static_cast<std::size_t>(element_tag),
                            index_ordering(node_tags, 
                              2,14, 6,13,24,19,3,15, 7,
                             11,23,18,20,26,25,9,22,17,
                              1,12, 5, 8,21,16,0,10, 4
                             ).data()
                            //index_ordering(node_tags, 
                            // 0, 8, 1, 9,20,11, 3,13, 2,
                            //10,21,12,22,26,23,15,24,14,
                            // 4,16, 5,17,25,18, 7,19, 6).data()
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
        are_elements_aggregated_ = true;
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