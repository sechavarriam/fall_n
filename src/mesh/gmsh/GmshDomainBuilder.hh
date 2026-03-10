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
#include "../../numerics/numerical_integration/SimplexQuadrature.hh"

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

    // ── Physical-group support ──────────────────────────────────────────
    // Map: entityDim → (entityTag → physical group name)
    // Built from $PhysicalNames + $Entities sections.
    std::map<int, std::map<int, std::string>> entity_to_physical_name_;

    void build_physical_name_map() {
        if (!mesh_info_.physical_names_info_.has_value()) return;
        if (!mesh_info_.entities_info_.has_value())       return;

        // physical tag → (dimension, name)
        std::map<int, std::pair<int, std::string>> phys_tag_map;
        for (const auto& [pdim, ptag, pname] : mesh_info_.physical_names_info_->physical_entities) {
            phys_tag_map[ptag] = {pdim, pname};
        }

        // Surfaces: entity tag → physical name
        for (const auto& surf : mesh_info_.entities_info_->surfaces) {
            for (int ptag : surf.physical_tags) {
                if (auto it = phys_tag_map.find(ptag); it != phys_tag_map.end()) {
                    entity_to_physical_name_[2][surf.tag] = it->second.second;
                }
            }
        }

        // Volumes: entity tag → physical name
        for (const auto& vol : mesh_info_.entities_info_->volumes) {
            for (int ptag : vol.physical_tags) {
                if (auto it = phys_tag_map.find(ptag); it != phys_tag_map.end()) {
                    entity_to_physical_name_[3][vol.tag] = it->second.second;
                }
            }
        }
    }

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
    { // Processes both 3D (cells) and 2D (boundary facets) element blocks.
        auto index_ordering = [&](auto& tags, std::integral auto... i) {
            return std::array{static_cast<PetscInt>(tags[i] - 1)...};
             };

        for (auto &block : mesh_info_.elements_info_.entityBlocks){          
            if (block.entityDim == 3){ // 3D volume elements
                // Resolve physical group name for this volume entity
                std::string vol_group;
                if (auto it = entity_to_physical_name_[3].find(block.entityTag);
                    it != entity_to_physical_name_[3].end()) {
                    vol_group = it->second;
                }

                for (auto &[element_tag, node_tags] : block.elementTags){
                    switch (block.elementType){
                    case 4: // TET_4 — 4-node linear tetrahedron
                        {
                        auto integrator = SimplexIntegrator<3,1>{};
                        auto& elem = domain_->make_element<SimplexElement<3,3,1>, decltype(integrator)>(
                            std::move(integrator),
                            static_cast<std::size_t>(element_tag),
                            index_ordering(node_tags, 0, 1, 2, 3).data()
                          );
                        if (!vol_group.empty()) elem.set_physical_group(vol_group);
                        break;
                        }
                    case 11: // TET_10 — 10-node quadratic tetrahedron
                        {
                        auto integrator = SimplexIntegrator<3,2>{};
                        // Gmsh edge midpoints: (0,1)=4, (1,2)=5, (0,2)=6, (0,3)=7, (1,3)=8, (2,3)=9
                        // fall_n edge midpoints: (0,1)=4, (0,2)=5, (0,3)=6, (1,2)=7, (1,3)=8, (2,3)=9
                        auto& elem = domain_->make_element<SimplexElement<3,3,2>, decltype(integrator)>(
                            std::move(integrator),
                            static_cast<std::size_t>(element_tag),
                            index_ordering(node_tags, 0, 1, 2, 3, 4, 6, 7, 5, 8, 9).data()
                          );
                        if (!vol_group.empty()) elem.set_physical_group(vol_group);
                        break;
                        }
                    case 5:
                        {
                        auto integrator = GaussLegendreCellIntegrator<1,1,1>{}; 
                           
                        auto& elem = domain_->make_element<LagrangeElement<3,2,2,2>, decltype(integrator)>(
                            std::move(integrator),
                            static_cast<std::size_t>(element_tag),
                            index_ordering(node_tags, 2, 6, 3, 7, 1, 5, 0, 4).data() 
                          );
                        if (!vol_group.empty()) elem.set_physical_group(vol_group);
                        
                        break;
                        }
                    case 12:
                        {
                        auto integrator = GaussLegendreCellIntegrator<3,3,3>{}; 
                           
                        auto& elem = domain_->make_element<LagrangeElement<3,3,3,3>, decltype(integrator)>(
                            std::move(integrator),
                            static_cast<std::size_t>(element_tag),
                            index_ordering(node_tags, 
                              2,14, 6,13,24,19,3,15, 7,
                             11,23,18,20,26,25,9,22,17,
                              1,12, 5, 8,21,16,0,10, 4
                             ).data()
                          );
                        if (!vol_group.empty()) elem.set_physical_group(vol_group);
                        break;
                        }
                    default:
                        {
                        std::cout << "Element type "<< block.elementType <<" not supported" << std::endl;
                        break;
                        }
                    }
                }
            }
            else if (block.entityDim == 2) { // 2D surface elements (boundary facets)
                // Resolve physical group name for this surface entity
                std::string group_name;
                if (auto it = entity_to_physical_name_[2].find(block.entityTag);
                    it != entity_to_physical_name_[2].end()) {
                    group_name = it->second;
                } else {
                    group_name = "__surface_" + std::to_string(block.entityTag);
                }

                for (auto &[element_tag, node_tags] : block.elementTags) {
                    switch (block.elementType) {
                    case 3: // QUA_4 — 4-node quad surface in 3D
                        {
                        auto integrator = GaussLegendreCellIntegrator<2,2>{};
                        domain_->make_boundary_element<LagrangeElement<3,2,2>, decltype(integrator)>(
                            group_name,
                            std::move(integrator),
                            static_cast<std::size_t>(element_tag),
                            index_ordering(node_tags, 0, 1, 2, 3).data()
                        );
                        break;
                        }
                    case 10: // QUA_9 — 9-node quad surface in 3D
                        {
                        auto integrator = GaussLegendreCellIntegrator<3,3>{};
                        domain_->make_boundary_element<LagrangeElement<3,3,3>, decltype(integrator)>(
                            group_name,
                            std::move(integrator),
                            static_cast<std::size_t>(element_tag),
                            index_ordering(node_tags, 0, 4, 1, 7, 8, 5, 3, 6, 2).data()
                        );
                        break;
                        }
                    case 2: // TRI_3 — 3-node triangle surface in 3D
                        {
                        auto integrator = SimplexIntegrator<2,1>{};
                        domain_->make_boundary_element<SimplexElement<3,2,1>, decltype(integrator)>(
                            group_name,
                            std::move(integrator),
                            static_cast<std::size_t>(element_tag),
                            index_ordering(node_tags, 0, 1, 2).data()
                        );
                        break;
                        }
                    case 9: // TRI_6 — 6-node triangle surface in 3D
                        {
                        auto integrator = SimplexIntegrator<2,2>{};
                        // Gmsh edge midpoints: (0,1)=3, (1,2)=4, (0,2)=5
                        // fall_n edge midpoints: (0,1)=3, (0,2)=4, (1,2)=5
                        domain_->make_boundary_element<SimplexElement<3,2,2>, decltype(integrator)>(
                            group_name,
                            std::move(integrator),
                            static_cast<std::size_t>(element_tag),
                            index_ordering(node_tags, 0, 1, 2, 3, 5, 4).data()
                        );
                        break;
                        }
                    default:
                        {
                        std::cout << "Boundary element type " << block.elementType
                                  << " not supported" << std::endl;
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
        build_physical_name_map();   // Resolve entity → physical group name
        aggregate_nodes();
        aggregate_elements();        // Now processes both 3D and 2D blocks

        // sort nodes with respect to their id
        std::sort(domain_->nodes().begin(), domain_->nodes().end(), 
        [](Node<3> &a, Node<3> &b) {
             return a.id() < b.id(); 
             }
        );

        domain_->assemble_sieve(); // Assemble sieve (DAG) for the domain (DMPlex)
    };
};



#endif // GMSH_DOMAIN_BUILDER_HH