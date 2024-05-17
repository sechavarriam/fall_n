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

    // $Entities store the boundary representation of the model geometrical entities,

    /*
    $Entities
      numPoints(size_t) numCurves(size_t)
        numSurfaces(size_t) numVolumes(size_t)
      pointTag(int) X(double) Y(double) Z(double)
        numPhysicalTags(size_t) physicalTag(int) ...
      ...
      curveTag(int) minX(double) minY(double) minZ(double)
        maxX(double) maxY(double) maxZ(double)
        numPhysicalTags(size_t) physicalTag(int) ...
        numBoundingPoints(size_t) pointTag(int; sign encodes orientation) ...
      ...
      surfaceTag(int) minX(double) minY(double) minZ(double)
        maxX(double) maxY(double) maxZ(double)
        numPhysicalTags(size_t) physicalTag(int) ...
        numBoundingCurves(size_t) curveTag(int; sign encodes orientation) ...
      ...
      volumeTag(int) minX(double) minY(double) minZ(double)
        maxX(double) maxY(double) maxZ(double
        numPhysicalTags(size_t) physicalTag(int) ...
        numBoundngSurfaces(size_t) surfaceTag(int; sign encodes orientation) ...
      ...
    $EndEntities
    */

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

        // auto parse_point = [&points](std::string_view line_extent){};

        EntitiesInfo(std::string_view keword_info)
        {

            std::size_t char_pos{0};
            auto pos{keword_info.data()};

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


            auto get_number = [&pos, &char_pos, &line_limit, &keword_info](auto &number)
            {
                auto number_limit = keword_info.find_first_of(" \n", char_pos);
                std::from_chars(pos + char_pos, pos + number_limit, number);
                char_pos = number_limit + 1;
                return number;
            };
            
            auto parse_point = [&]()
            {
                int tag; // Organizar como tupla para recorrer rápido?
                double X, Y, Z;

                std::size_t  num_physical_tags;
                std::vector<int> physical_tags;

                //std::size_t char_pos{0};
                //auto pos{keword_info.data()};
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

            for (auto i = 0; i < num_points; ++i)
            {
                parse_point();
            }



            auto parse_entity = [&]()
            {
                int tag; // Organizar como tupla para recorrer rápido?
                double X, Y, Z;

                std::size_t  num_physical_tags;
                std::vector<int> physical_tags;

                auto line_limit = keword_info.find_first_of('\n', char_pos);

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
            };






        };
    };








};// namespace gmsh

// $Nodes and $Elements store mesh data classified on these entities
// $PhysicalNames store the names of the physical entities

 

#endif // FALL_N_MESH_INTERFACE