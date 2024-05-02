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


//namespace mesh {

namespace gmsh{

class MSHReader {

// https://stackoverflow.com/questions/71882752/c-constexpr-stdarray-of-string-literals
static constexpr std::array msh_keywords{
        "$MeshFormat"            ,// 0  //required always
        "$EndMeshFormat"         ,// 1  //required always
        "$PhysicalNames"         ,// 2 
        "$EndPhysicalNames"      ,// 3 
        "$Entities"              ,// 4
        "$EndEntities"           ,// 5 
        "$PartitionedEntities"   ,// 6 
        "$EndPartitionedEntities",// 7 
        "$Nodes"                 ,// 8  //required always
        "$EndNodes"              ,// 9  //required always
        "$Elements"              ,// 10 //required always
        "$EndElements"           ,// 11 //required always
        "$Periodic"              ,// 12 
        "$EndPeriodic"           ,// 13
        "$GhostElements"         ,// 14
        "$EndGhostElements"      ,// 15
        "$Parametrizations"      ,// 16
        "$EndParametrizations"   ,// 17
        "$NodeData"              ,// 18
        "$EndNodeData"           ,// 19
        "$ElementData"           ,// 20
        "$EndElementData"        ,// 21
        "$ElementNodeData"       ,// 22
        "$EndElementNodeData"    ,// 23
        "$InterpolationScheme"   ,// 24
        "$EndInterpolationScheme" // 25
    };

std::array<std::size_t, msh_keywords.size()> keyword_position_{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};


std::string filename_;
std::string buffer_  ; //Podría ser un char*

std::string_view buffer_view_;

public:

template<std::size_t N>
inline constexpr auto view_keyword_info(){
    std::string_view keyword = msh_keywords[N];

    std::size_t pos = keyword_position_[N]+keyword.size();
    std::size_t end = keyword_position_[N+1];

    std::size_t count = end - pos;

    if (end == 0){
        std::cout << "Keyword " << msh_keywords[N] << " not found" << std::endl;
        return std::string_view{};
    }
    return buffer_view_.substr(pos, count);
};

inline constexpr auto view_MeshFormat         (){return view_keyword_info<0 >();};
inline constexpr auto view_PhysicalNames      (){return view_keyword_info<2 >();};
inline constexpr auto view_Entities           (){return view_keyword_info<4 >();};
inline constexpr auto view_PartitionedEntities(){return view_keyword_info<6 >();};
inline constexpr auto view_Nodes              (){return view_keyword_info<8 >();};
inline constexpr auto view_Elements           (){return view_keyword_info<10>();};
inline constexpr auto view_Periodic           (){return view_keyword_info<12>();};
inline constexpr auto view_GhostElements      (){return view_keyword_info<14>();};
inline constexpr auto view_Parametrizations   (){return view_keyword_info<16>();};
inline constexpr auto view_NodeData           (){return view_keyword_info<18>();};
inline constexpr auto view_ElementData        (){return view_keyword_info<20>();};
inline constexpr auto view_ElementNodeData    (){return view_keyword_info<22>();};
inline constexpr auto view_InterpolationScheme(){return view_keyword_info<24>();};


void print_positions(){
    for (auto const& pos : keyword_position_){
        std::cout << pos << std::endl;
    }
};

MSHReader(std::string_view filename): filename_(filename){
    
    //https://stackoverflow.com/questions/18398167/how-to-copy-a-txt-file-to-a-char-array-in-c
    std::ifstream in(filename_);
    if (!in) {std::cerr << "Cannot open the File : " << filename_ << std::endl;}

    buffer_ = std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

    std::size_t i = 0;
    for (auto const& keyword : msh_keywords){ //sintetizar en función. set_positions();
        std::size_t pos = buffer_.find(keyword); //Se puede hacer mas efiiente aprovechando el orden de los keywords.
        if (pos != std::string::npos){
            keyword_position_[i] = pos;
            std::cout << "Found " << keyword << " at position " << pos << std::endl;
        }
        ++i;
    }
    buffer_view_ = std::string_view(buffer_.data());
}; 




};  



struct MeshFormat{
    double  version{0.0};
    int    file_type{-1};
    int    data_size{-1};

    MeshFormat(std::string_view keword_info){
        
        //std::size_t first{0};
        //std::size_t last{0};

        std::size_t pos{0};

        //https://stackoverflow.com/questions/73333331/convert-stdstring-view-to-float
        //https://lemire.me/blog/2022/07/27/comparing-strtod-with-from_chars-gcc-12/

        //for (auto c : keword_info){
        //    
        //}
        |
        auto [ptr, ec] = std::from_chars( keword_info.data(), keword_info.data()+3, version);
        if (ptr == keword_info.data()){
            std::cerr << "Error parsing version " << std::endl; //you have errors!
        //you have errors!
        }
        

        //pos = keword_info.find_first_of(" \t", pos);
        //pos = keword_info.find_first_not_of(" \t", pos);
        //auto [p1, ec1] = std::from_chars(keword_info.data()+pos, keword_info.data()+keword_info.size(), file_type);
        //pos = keword_info.find_first_of(" \t", pos);
        //pos = keword_info.find_first_not_of(" \t", pos);
        //auto [p2, ec2] = std::from_chars(keword_info.data()+pos, keword_info.data()+keword_info.size(), data_size);
    
    };
};

} // namespace gmsh


#endif // FALL_N_MESH_INTERFACE