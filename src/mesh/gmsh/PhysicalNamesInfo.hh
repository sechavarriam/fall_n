#ifndef GMSH_PHYSICAL_NAMES_INFORMATION
#define GMSH_PHYSICAL_NAMES_INFORMATION

namespace gmsh
{
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

        int numPhysicalNames;
        std::vector<PhysicalEntity> physical_entities;

        void print_raw()
        {
            std::cout << numPhysicalNames << std::endl;
            for (auto const &[dim, tag, name] : physical_entities)
            {
                std::cout << dim << " " << tag << " " << name << std::endl;
            }
        };

        PhysicalNamesInfo() = default;
        PhysicalNamesInfo(std::string_view keword_info)
        {
            auto pos{keword_info.data()};
            std::size_t char_pos{0};
            auto get_number = [&](auto &number)
            {
                auto number_limit = keword_info.find_first_of(" \n", char_pos);
                std::from_chars(pos + char_pos, pos + number_limit, number);
                char_pos = number_limit + 1;
                return number;
            };
            
            get_number(numPhysicalNames);

            auto line_limit = keword_info.find_first_of('\n', char_pos);
            
            for (int i = 0; i < numPhysicalNames; i++)
            {
                int dimension, physicalTag;
                std::string name;

                get_number(dimension);
                get_number(physicalTag);
                
                auto name_limit = keword_info.find_first_of("\"", char_pos+1);
                name = std::string(pos + char_pos +1, name_limit - char_pos-1); // +1, -1 to remove the quotes.
                char_pos = line_limit + 1;
                physical_entities.push_back(std::make_tuple(dimension, physicalTag, name));
                line_limit = keword_info.find_first_of('\n', char_pos);
            }
        };
    };

}// namespace gmsh

#endif // GMSH_PHYSICAL_NAMES_INFORMATION