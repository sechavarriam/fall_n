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
#include "../../domain/elements/Element.hh"

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

        std::cout << "------------------------------------------------------------------------------------------------" << std::endl;
        std::cout << "--------  Node Aggregation Started -------------------------------------------------------------" << std::endl;
        std::cout << "------------------------------------------------------------------------------------------------" << std::endl;

        
        domain_->preallocate_node_capacity(mesh_info_.nodes_info_.numNodes); // IMPORTANT!!!!
        node_addresses_.reserve(mesh_info_.nodes_info_.numNodes);
        

        std::cout << "Number of nodes: " << mesh_info_.nodes_info_.numNodes << std::endl;
        std::cout << "Node addresses capacity: " << node_addresses_.capacity() << std::endl;     
        std::cout << "------------------------------------------------------------------------------------------------" << std::endl;

        for (auto &block : mesh_info_.nodes_info_.entityBlocks)
        {
            std::cout << "_________________________________________________________________________________" << std::endl;
            std::cout << "Block number of nodes: " << block.numNodesInBlock << std::endl;

            for (std::size_t node = 0; node < block.numNodesInBlock; node++)
            {
                std::cout << "___________________________________________________" << std::endl;
                std::cout << "Node tag: " << block.nodeTag[node] << std::endl;
                std::cout << "Node coordinates: " << block.coordinates[node][0] << " " << block.coordinates[node][1] << " " << block.coordinates[node][2] << std::endl;
                node_addresses_.push_back( // Stores the address of the node created.
                    domain_->add_node(     // Create Node in the domain
                        Node<3>(
                            std::move(block.nodeTag[node]),
                            std::move(block.coordinates[node][0]),
                            std::move(block.coordinates[node][1]),
                            std::move(block.coordinates[node][2]))));

                std::cout << "¬¬¬¬¬¬¬¬ Stored data ¬¬¬¬¬¬¬¬" << std::endl;
                std::cout << "Node address: " << node_addresses_.back()  << std::endl;
                std::cout << "Node id: " << node_addresses_.back()->id() << std::endl;
                std::cout << "Node coordinates: " << node_addresses_.back()->coord(0) << " " << node_addresses_.back()->coord(1) << " " << node_addresses_.back()->coord(2) << std::endl;
            }
            std::cout << "___________________________________________________" << std::endl;
        }

        std::cout << "------------------------------------------------------------------------------------------------" << std::endl;
        std::cout << "--------  End Node Aggregation -----------------------------------------------------------------" << std::endl;
        std::cout << "------------------------------------------------------------------------------------------------" << std::endl;
    };

    void aggregate_elements()
    { // Esto procesa tanto elemento 2d (facets) como 3d  (cells) TODO:FIX.
        std::cout << "------------------------------------------------------------------------------------------------" << std::endl;
        std::cout << "--------  Element Aggregation Started ----------------------------------------------------------" << std::endl;
        std::cout << "------------------------------------------------------------------------------------------------" << std::endl;

        for (auto &block : mesh_info_.elements_info_.entityBlocks)
        {
            std::cout << "_________________________________________________________________________________" << std::endl;
            std::cout << "Block number of elements: " << block.numElementsInBlock << std::endl;
            std::cout << "Block entity dimension: " << block.entityDim << std::endl;
            std::cout << "Block element type: " << block.elementType << std::endl;
            
            if (block.entityDim == 3)
            { // Only 3D elements (by now...)

                for (auto &[element_tag, node_tags] : block.elementTags)
                {
                    std::cout << "___________________________________________________" << std::endl;
                    std::cout << "Element tag: " << element_tag << std::endl;
                    std::cout << "Node tags: ";
                    for (auto &tag : node_tags) std::cout << tag << " ";
                    std::cout << std::endl;

                    std::vector<Node<3> *> element_node_pointers;
                    element_node_pointers.reserve(node_tags.size());

                    for (auto &tag : node_tags)
                    {
                        std::cout << "°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°" << std::endl;
                        std::cout << "Node tag: " << tag << std::endl;
                        std::cout << "Node address: " << node_addresses_[tag - 1] << std::endl;
                        std::cout << "Domain node id: " << node_addresses_[tag - 1]->id() << std::endl;
                        if (node_addresses_[tag - 1]->id() == tag)
                        {
                            element_node_pointers.push_back(node_addresses_[tag-1]);
                        }
                        else
                        { // TODO: search for the node with the current tag.
                            //throw std::runtime_error("Node tag does not match with node id");
                            // Range search
                            auto it = std::ranges::find_if(domain_->nodes(), [tag](Node<3>  &node) { return node.id() == tag; });
                            if (it != domain_->nodes().end())
                            {
                                element_node_pointers.push_back(std::addressof(*it));
                                std::cout << "Node found in the range" << std::endl;
                            }
                            else
                            {
                                std::cout << "Node not found in the range" << std::endl;
                            }
                            
                            //std::cout << " NO SE CUMPLE LA CONDICIÓN "<< std::endl;
                        }
                    }
                    std::cout << "°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°" << std::endl;
                    // Emplace Element
                    switch (block.elementType)
                    {
                    case 5:
                        std::cout << element_tag << std::endl;
                       
                        domain_->make_element<LagrangeElement<2,2,2>,GaussLegendreCellIntegrator<1,1,1>>(
                            GaussLegendreCellIntegrator<1,1,1>{},
                            std::size_t(element_tag),
                            element_node_pointers
                          );

                        break;
                    default:
                        //throw std::runtime_error("Element type not supported");
                        std::cout << "Element type "<< block.elementType <<" not supported" << std::endl;
                        break;
                    }
                }
            }
        }
        std::cout << "------------------------------------------------------------------------------------------------" << std::endl;
        std::cout << "--------  End Element Aggregation --------------------------------------------------------------" << std::endl;
        std::cout << "------------------------------------------------------------------------------------------------" << std::endl;
    };

    GmshDomainBuilder(std::string_view filename, Domain3D &domain) : mesh_info_(filename), domain_(std::addressof(domain))
    {
        aggregate_nodes();
        aggregate_elements();
    };
};

//class GmshDomainBuilder_3D
//{
//    using Node3D = Node<3>;
//    gmsh::MSHReader mesh_info_;
//
//public:
//    std::vector<Node3D> nodes_;
//    std::vector<Element> elements_;
//
//    void process_nodes()
//    {
//        for (auto &block : mesh_info_.nodes_info_.entityBlocks)
//        {
//            for (std::size_t node = 0; node < block.numNodesInBlock; node++)
//            {
//                nodes_.emplace_back(
//                    Node3D(
//                        std::move(block.nodeTag[node]),
//                        std::move(block.coordinates[node][0]),
//                        std::move(block.coordinates[node][1]),
//                        std::move(block.coordinates[node][2])));
//            }
//        }
//    };
//
//    void process_elements()
//    { // Esto procesa tanto elemento 2d (facets) como 3d  (cells) TODO:FIX.
//        elements_.reserve(mesh_info_.elements_info_.numElements);
//
//        auto default_integration_scheme = [](auto const &e) { /**integrate*/ };
//
//        for (auto &block : mesh_info_.elements_info_.entityBlocks)
//        {
//            if (block.entityDim == 3)
//            { // Only 3D elements (by now...)
//                for (auto &[element_tag, node_tags] : block.elementTags)
//                {
//                    std::vector<Node3D *> element_node_pointers;
//                    element_node_pointers.reserve(node_tags.size());
//
//                    for (auto &tag : node_tags)
//                    {
//                        if (nodes_[tag - 1].id() == tag)
//                        {
//                            element_node_pointers.push_back(&nodes_[tag - 1]);
//                        }
//                        else
//                        { // TODO: search for the node with the current tag.
//                            throw std::runtime_error("Node tag does not match with node id");
//                        }
//                    }
//                    // Emplace Element
//                    switch (block.elementType)
//                    {
//                    case 5:
//                        elements_.emplace_back(
//                            Element(
//                                LagrangeElement<2, 2, 2>{
//                                    std::move(element_tag),
//                                    std::move(element_node_pointers)},
//                                default_integration_scheme));
//                        break;
//
//                    default:
//                        throw std::runtime_error("Element type not supported");
//                        break;
//                    }
//                }
//            }
//        }
//    };
//
//    GmshDomainBuilder_3D(std::string_view filename) : mesh_info_(filename)
//    {
//        process_nodes();
//        process_elements();
//    };
//};

#endif // GMSH_DOMAIN_BUILDER_HH