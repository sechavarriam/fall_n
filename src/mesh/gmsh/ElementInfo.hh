#ifndef READELEMENTINFO_HH
#define READELEMENTINFO_HH

#include <string>
#include <fstream>
#include <iostream>
#include <string_view>

#include <cstdlib>
#include <charconv>

#include <array>
#include <map>
#include <vector>

namespace gmsh
{
    /*
$Elements
  numEntityBlocks(size_t) numElements(size_t)
    minElementTag(size_t) maxElementTag(size_t)
  entityDim(int) entityTag(int) elementType(int; see below)
    numElementsInBlock(size_t)
    elementTag(size_t) nodeTag(size_t) ...
    ...
  ...
$EndElements
*/

    namespace Element{
        struct EntityBlock
        {
            int entityDim, entityTag, elementType;
            std::size_t numElementsInBlock;

            std::map<std::size_t, std::vector<std::size_t>> elementTags;

            void print_raw()
            {
                std::cout << entityDim << " " << entityTag << " " << elementType << " " << numElementsInBlock << " " << std::endl;
                for (auto const &[e_tag, n_tags] : elementTags)
                {
                    std::cout << e_tag << " ";
                    for (auto const &n_tag : n_tags){std::cout << n_tag << " ";};
                    std::cout << std::endl;
                }
            };
        };
    };

    struct ElementInfo
    {
        std::size_t numEntityBlocks;
        std::size_t numElements;
        std::size_t minElementTag;
        std::size_t maxElementTag;

        std::vector<Element::EntityBlock> entityBlocks;

        //Default constructor
        ElementInfo() = default;

        ElementInfo(std::string_view keword_info)
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
            get_number(numElements);
            get_number(minElementTag);
            get_number(maxElementTag);

            auto parse_entity_block = [&]()
            {
                int entityDim,entityTag,elementType;
                std::size_t numElementsInBlock;
                std::map<std::size_t, std::vector<std::size_t>> elementTags;

                get_number(entityDim);
                get_number(entityTag);
                get_number(elementType);
                get_number(numElementsInBlock);

                std::size_t e_tag; //Element tag
                std::size_t n_tag; //Node tag
                std::vector<std::size_t> node_tags;

                line_limit = keword_info.find_first_of('\n', char_pos);
                for (auto j = 0; j < numElementsInBlock; ++j)
                {
                    get_number(e_tag); 
                    do
                    {
                        node_tags.push_back(get_number(n_tag));
                    } while (char_pos < line_limit);

                    char_pos = line_limit + 1; //next line
                    line_limit = keword_info.find_first_of('\n', char_pos);
                    elementTags.emplace(std::move(e_tag), std::move(node_tags));
                }
                entityBlocks.emplace_back(std::move(Element::EntityBlock{entityDim, entityTag, elementType, numElementsInBlock, elementTags}));
            };
            for (auto i = 0; i < numEntityBlocks; ++i) parse_entity_block();
        };
    };


} // namespace gmsh







#endif // READELEMENTINFO_HH