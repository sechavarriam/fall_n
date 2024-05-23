#ifndef GMSH_MSH_FILE_FORMAT_INFO_HH
#define GMSH_MSH_FILE_FORMAT_INFO_HH

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
    struct MeshFormatInfo
    {
        double version{0.0};
        int file_type{-1};
        int data_size{-1};

        //default constructor
        MeshFormatInfo() = default;

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


} // namespace gmsh







#endif // GMSH_MSH_FILE FORMAT_INFO_HH