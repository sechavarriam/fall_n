#ifndef FALL_N_READ_GMSH
#define FALL_N_READ_GMSH

#include <string>
#include <fstream>
#include <iostream>
#include <string_view>

#include <cstdlib>
#include <charconv>

#include <array>
#include <map>
#include <vector>

#include "../Mesh.hh"
#include "../../domain/Domain.hh"

// namespace mesh {

namespace gmsh
{

    class MSHReader
    {

        // https://stackoverflow.com/questions/71882752/c-constexpr-stdarray-of-string-literals
        static constexpr std::array msh_keywords{
            "$MeshFormat",             // 0  //required always
            "$EndMeshFormat",          // 1  //required always
            "$PhysicalNames",          // 2
            "$EndPhysicalNames",       // 3
            "$Entities",               // 4
            "$EndEntities",            // 5
            "$PartitionedEntities",    // 6
            "$EndPartitionedEntities", // 7
            "$Nodes",                  // 8  //required always
            "$EndNodes",               // 9  //required always
            "$Elements",               // 10 //required always
            "$EndElements",            // 11 //required always
            "$Periodic",               // 12
            "$EndPeriodic",            // 13
            "$GhostElements",          // 14
            "$EndGhostElements",       // 15
            "$Parametrizations",       // 16
            "$EndParametrizations",    // 17
            "$NodeData",               // 18
            "$EndNodeData",            // 19
            "$ElementData",            // 20
            "$EndElementData",         // 21
            "$ElementNodeData",        // 22
            "$EndElementNodeData",     // 23
            "$InterpolationScheme",    // 24
            "$EndInterpolationScheme"  // 25
        };

        std::array<std::size_t, msh_keywords.size()> keyword_position_{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        std::string filename_;
        std::string buffer_; // Podría ser un char*

        std::string_view buffer_view_;

    public:
        template <std::size_t N>
        inline constexpr auto view_keyword_info()
        {
            std::string_view keyword = msh_keywords[N];

            std::size_t pos = keyword_position_[N] + keyword.size() + 1; //+1 to skip the newline character
            std::size_t end = keyword_position_[N + 1];

            std::size_t count = end - pos;

            if (end == 0)
            {
                std::cout << "Keyword " << msh_keywords[N] << " not found" << std::endl;
                return std::string_view{};
            }
            return buffer_view_.substr(pos, count);
        };

        inline constexpr auto view_MeshFormat() { return view_keyword_info<0>(); };
        inline constexpr auto view_PhysicalNames() { return view_keyword_info<2>(); };
        inline constexpr auto view_Entities() { return view_keyword_info<4>(); };
        inline constexpr auto view_PartitionedEntities() { return view_keyword_info<6>(); };
        inline constexpr auto view_Nodes() { return view_keyword_info<8>(); };
        inline constexpr auto view_Elements() { return view_keyword_info<10>(); };
        inline constexpr auto view_Periodic() { return view_keyword_info<12>(); };
        inline constexpr auto view_GhostElements() { return view_keyword_info<14>(); };
        inline constexpr auto view_Parametrizations() { return view_keyword_info<16>(); };
        inline constexpr auto view_NodeData() { return view_keyword_info<18>(); };
        inline constexpr auto view_ElementData() { return view_keyword_info<20>(); };
        inline constexpr auto view_ElementNodeData() { return view_keyword_info<22>(); };
        inline constexpr auto view_InterpolationScheme() { return view_keyword_info<24>(); };

        void print_positions()
        {
            for (auto const &pos : keyword_position_)
            {
                std::cout << pos << std::endl;
            }
        };

        MSHReader(std::string_view filename) : filename_(filename)
        {

            // https://stackoverflow.com/questions/18398167/how-to-copy-a-txt-file-to-a-char-array-in-c
            std::ifstream in(filename_);
            if (!in)
            {
                std::cerr << "Cannot open the File : " << filename_ << std::endl;
            }

            buffer_ = std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

            std::size_t i = 0;
            for (auto const &keyword : msh_keywords)
            {                                            // sintetizar en función. set_positions();
                std::size_t pos = buffer_.find(keyword); // Se puede hacer mas efiiente aprovechando el orden de los keywords.
                if (pos != std::string::npos)
                {
                    keyword_position_[i] = pos;
                    std::cout << "Found " << keyword << " at position " << pos << std::endl;
                }
                ++i;
            }
            buffer_view_ = std::string_view(buffer_.data());
        };
    };

    struct MeshFormatInfo
    {
        double version{0.0};
        int file_type{-1};
        int data_size{-1};

        MeshFormatInfo(std::string_view keword_info)
        {
            auto pos{keword_info.data()};
            // https://stackoverflow.com/questions/73333331/convert-stdstring-view-to-float
            // https://lemire.me/blog/2022/07/27/comparing-strtod-with-from_chars-gcc-12/

            auto [ptr, ec] = std::from_chars(pos, pos + 3, version);
            if (ptr == pos)
            {
                std::cerr << "Error parsing MSH file format version.  " << std::endl;
            }

            if (version < 4.1)
            {
                std::cerr << "Error: Gmsh version must be 4.1 or higher.  " << std::endl;
            }

            pos = ptr + 1; // skip the space character
            auto [ptr2, ec2] = std::from_chars(pos, pos + 1, file_type);
            if (ptr2 == pos)
            {
                std::cerr << "Error parsing MSH file format file type.  " << std::endl;
            }

            pos = ptr2 + 1; // skip the space character
            auto [ptr3, ec3] = std::from_chars(pos, pos + 1, data_size);
            if (ptr3 == pos)
            {
                std::cerr << "Error parsing MSH file format data size.  " << std::endl;
            }
        };
    };

    /*
    $PhysicalNames // same as MSH version 2
      numPhysicalNames(ASCII int)
      dimension(ASCII int) physicalTag(ASCII int) "name"(127 characters max)
      ...
    $EndPhysicalNames
    */
    struct PhysicalNamesInfo
    {
        using PhysicalEntity = std::tuple<int, int, std::string>;
        std::vector<PhysicalEntity> physical_entities;

        
    };



    namespace Entity
    {
        struct Point
        {
            int tag;
            double X, Y, Z;
            std::size_t num_physical_tags;
            std::vector<int> physical_tags;

            void print_raw()
            {
                std::cout << tag << " " << X << " " << Y << " " << Z << " " << num_physical_tags << " ";
                for (auto const &tag : physical_tags){std::cout << tag << " ";};
                std::cout << std::endl;
            };
        };

        struct GeneralEntity // Curve, Surface, Volume
        {
            int tag;
            double minX, minY, minZ;
            double maxX, maxY, maxZ;

            std::size_t  num_physical_tags;
            std::vector<int> physical_tags;
            std::size_t  num_bounding_entities; // numBoundingPoints, numBoundingCurves, numBoundingSurfaces
            std::vector<int> bounding_entities;

            void print_raw()
            {
                std::cout << tag << " " << minX << " " << minY << " " << minZ << " " << maxX << " " << maxY << " " << maxZ << " " << num_physical_tags << " " << num_bounding_entities << " ";
                for (auto const &tag : bounding_entities){std::cout << tag << " ";};
                std::cout << std::endl;
            };

            void print()
            {
                std::cout << "tag: " << tag << std::endl;
                std::cout << "minX: " << minX << std::endl;
                std::cout << "minY: " << minY << std::endl;
                std::cout << "minZ: " << minZ << std::endl;
                std::cout << "maxX: " << maxX << std::endl;
                std::cout << "maxY: " << maxY << std::endl;
                std::cout << "maxZ: " << maxZ << std::endl;
                std::cout << "num_physical_tags: " << num_physical_tags << std::endl;
                std::cout << "num_bounding_entities: " << num_bounding_entities << std::endl;
            };
        };
    }; // namespace Entity

    struct EntitiesInfo
    {
        using Point   = Entity::Point;
        using Curve   = Entity::GeneralEntity;
        using Surface = Entity::GeneralEntity;
        using Volume  = Entity::GeneralEntity;

        std::array<std::size_t, 4> num_entities{0, 0, 0, 0}; // numPoints(size_t) numCurves(size_t) numSurfaces(size_t) numVolumes(size_t)

        std::vector<Point>   points;
        std::vector<Curve>   curves;
        std::vector<Surface> surfaces;
        std::vector<Volume>  volumes;

        EntitiesInfo(std::string_view keword_info)
        {
            const auto pos{keword_info.data()};

            std::size_t char_pos{0};
            auto line_limit = keword_info.find_first_of('\n');

            std::size_t N{0};
            for (auto i = char_pos; i < line_limit + 1; ++i)
            { // parse the first line
                if (keword_info[i] == ' ' || keword_info[i] == '\n')
                {
                    std::from_chars(pos + char_pos, pos + i, num_entities[N++]);
                    char_pos = i + 1;
                }
            }

            auto [num_points, num_curves, num_surfaces, num_volumes] = num_entities;

            auto get_number = [&](auto &number)
            {
                auto number_limit = keword_info.find_first_of(" \n", char_pos);
                std::from_chars(pos + char_pos, pos + number_limit, number);
                char_pos = number_limit + 1;
                return number;
            };
            
            auto parse_point = [&]()
            {
                int tag; 
                double X, Y, Z;
                std::size_t  num_physical_tags;
                std::vector<int> physical_tags;

                line_limit = keword_info.find_first_of('\n', char_pos);

                get_number(tag);
                get_number(X);
                get_number(Y);
                get_number(Z);
                get_number(num_physical_tags);
                for (auto j = 0; j < num_physical_tags; ++j)
                {
                    int physical_tag;
                    physical_tags.push_back(get_number(physical_tag));
                }
                points.emplace_back(std::move(Point{tag, X, Y, Z, num_physical_tags, physical_tags}));
                char_pos = line_limit + 1;
            };

            auto parse_entity = [&](std::vector<Entity::GeneralEntity> &container)
            {
                int tag; // Organizar como tupla para recorrer rápido?
                double minX, minY, minZ;
                double maxX, maxY, maxZ;

                std::size_t  num_physical_tags;
                std::vector<int> physical_tags;

                std::size_t  num_bounding_entities;
                std::vector<int> bounding_entities;

                auto line_limit = keword_info.find_first_of('\n', char_pos);

                get_number(tag);
                get_number(minX);
                get_number(minY);
                get_number(minZ);
                get_number(maxX);
                get_number(maxY);
                get_number(maxZ);
                get_number(num_physical_tags);
    
                for (auto j = 0; j < num_physical_tags; ++j)
                {
                    int physical_tag;
                    physical_tags.push_back(get_number(physical_tag));
                }

                get_number(num_bounding_entities);
                for (auto j = 0; j < num_bounding_entities; ++j)
                {
                    int bounding_entity;
                    bounding_entities.push_back(get_number(bounding_entity));
                }

                container.emplace_back(std::move(Entity::GeneralEntity{tag, minX, minY, minZ, maxX, maxY, maxZ, num_physical_tags, physical_tags, num_bounding_entities, bounding_entities}));
                char_pos = line_limit + 1;
            };

            for (auto i = 0; i < num_points;   ++i) parse_point();
            for (auto i = 0; i < num_curves;   ++i) parse_entity(curves);
            for (auto i = 0; i < num_surfaces; ++i) parse_entity(surfaces);
            for (auto i = 0; i < num_volumes;  ++i) parse_entity(volumes);
        };
    };


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

Example
$Nodes
1 6 1 6     1 entity bloc, 6 nodes total, min/max node tags: 1 and 6
2 1 0 6     2D entity (surface) 1, no parametric coordinates, 6 nodes
1             node tag #1
2             node tag #2
3             etc.
4
5
6
0. 0. 0.      node #1 coordinates (0., 0., 0.)
1. 0. 0.      node #2 coordinates (1., 0., 0.)
1. 1. 0.      etc.
0. 1. 0.
2. 0. 0.
2. 1. 0.
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
                        std::cout << n_tag << " ";
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



};// namespace gmsh


 

#endif // FALL_N_MESH_INTERFACE