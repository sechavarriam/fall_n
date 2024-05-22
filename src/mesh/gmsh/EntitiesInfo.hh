#ifndef GMSH_ENTITIES_INFO
#define GMSH_ENTITIES_INFO

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

} // namespace gmsh


#endif // GMSH_ENTITIES_INFO