#ifndef GMSH_DOMAIN_BUILDER_HH
#define GMSH_DOMAIN_BUILDER_HH

// Your code goes here
#include <string>
#include <string_view>
#include <memory>
#include <vector>

#include "../../domain/Node.hh"

#include "ReadGmsh.hh"


class GmshDomainBuilder_3D
{
private:
public:

    gmsh::MSHReader mesh_info_;

    //std::unique_ptr<ReadGmsh> reader_;
    std::vector<Node<3>> nodes_;

    void process_nodes(){
        for (auto& block : mesh_info_.nodes_info_.entityBlocks){
            for (std::size_t node = 0;  node < block.numNodesInBlock; node++){
                nodes_.emplace_back(
                    Node<3>(
                        std::move(block.nodeTag[node]),
                        std::move(block.coordinates[node][0]),
                        std::move(block.coordinates[node][1]),
                        std::move(block.coordinates[node][2]) 
                    )
                );
            }
        }
    };



    GmshDomainBuilder_3D(std::string_view filename) : mesh_info_(filename)
    {
        process_nodes();
    };

};




#endif // GMSH_DOMAIN_BUILDER_HH