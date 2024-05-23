#ifndef GMSH_HSHFILE_NODES_INFO
#define GMSH_HSHFILE_NODES_INFO


#include <string>
#include <fstream>
#include <iostream>
#include <string_view>
#include <cstdlib>
#include <charconv>
#include <array>
#include <map>
#include <vector>
#include <tuple>

namespace gmsh
{
    /*
$Nodes
  numEntityBlocks(size_t) numNodes(size_t)
    minNodeTag(size_t) maxNodeTag(size_t)
  entityDim(int) entityTag(int) parametric(int; 0 or 1)
    numNodesInBlock(size_t)
    nodeTag(size_t)
    ...
    x(double) y(double) z(double)
       < u(double; if parametric and entityDim >= 1) >
       < v(double; if parametric and entityDim >= 2) >
       < w(double; if parametric and entityDim == 3) >
    ...
  ...
$EndNodes
*/
    namespace Node{
        struct EntityBlock
        {
            int entityDim, entityTag, parametric;
            std::size_t numNodesInBlock;

            std::vector<int> nodeTag;
            std::vector<std::array<double, 3>> coordinates; // This should be another struct or tuple if parametric is allowed

            void print_raw()
            {
                std::cout << entityDim << " " << entityTag << " " << parametric << " " << numNodesInBlock << " " << std::endl;
                for (auto const &tag : nodeTag){std::cout << tag << std::endl;};
                for (auto const &coord : coordinates){std::cout << coord[0] << " " << coord[1] << " " << coord[2] << std::endl;};
            };
        };
    };

    struct NodesInfo
    {
        std::size_t numEntityBlocks;
        std::size_t numNodes;
        std::size_t minNodeTag;
        std::size_t maxNodeTag;

        std::vector<Node::EntityBlock> entityBlocks;

        //default constructor
        NodesInfo() = default;

        NodesInfo(std::string_view keword_info)
        {
            const auto pos{keword_info.data()};

            std::size_t char_pos{0};
            auto line_limit = keword_info.find_first_of('\n');

            auto get_number = [&](auto &number)
            {
                auto number_limit = keword_info.find_first_of(" \n", char_pos);
                std::from_chars(pos + char_pos, pos + number_limit, number);
                char_pos = number_limit + 1;
                return number;
            };

            get_number(numEntityBlocks);
            get_number(numNodes);
            get_number(minNodeTag);
            get_number(maxNodeTag);
            char_pos = line_limit + 1;

            auto parse_entity_block = [&]()
            {
                int entityDim,entityTag,parametric;
                std::size_t numNodesInBlock;
                std::vector<int> nodeTag;
                std::vector<std::array<double, 3>> coordinates;

                get_number(entityDim);
                get_number(entityTag);
                get_number(parametric);
                get_number(numNodesInBlock);

                int node_tag;
                double x, y, z;

                for (auto j = 0; j < numNodesInBlock; ++j)nodeTag.push_back(get_number(node_tag));
                for (auto j = 0; j < numNodesInBlock; ++j)
                {
                    get_number(x);
                    get_number(y);
                    get_number(z);
                    coordinates.push_back({x, y, z});
                }
                entityBlocks.emplace_back(std::move(Node::EntityBlock{entityDim, entityTag, parametric, numNodesInBlock, nodeTag, coordinates}));
            };
            for (auto i = 0; i < numEntityBlocks; ++i) parse_entity_block();
        };

    };

} // namespace gmsh


#endif // GMSH_HSHFILE_NODES_INFO