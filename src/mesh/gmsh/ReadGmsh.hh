#ifndef FALL_N_READ_GMSH
#define FALL_N_READ_GMSH

#include <string>
#include <fstream>
#include <iostream>

#include <array>
#include <map>
#include <vector>


#include "../Mesh.hh"
#include "../../domain/Domain.hh"


//namespace mesh {


class MSHReader {

// https://stackoverflow.com/questions/71882752/c-constexpr-stdarray-of-string-literals
static constexpr std::array msh_keywords{
        "$MeshFormat"            , //required always
        "$EndMeshFormat"         , //required always
        "$PhysicalNames"         ,
        "$EndPhysicalNames"      ,
        "$Entities"              , 
        "$PartitionedEntities"   , 
        "$EndPartitionedEntities",
        "$Nodes"                 , //required always
        "$EndNodes"              , //required always
        "$Elements"              , //required always
        "$EndElements"           , //required always
        "$Periodic"              ,
        "$EndPeriodic"           ,
        "$GhostElements"         ,
        "$EndGhostElements"      ,
        "$Parametrizations"      ,
        "$EndParametrizations"   ,
        "$NodeData"              ,
        "$EndNodeData"           ,
        "$ElementData"           ,
        "$EndElementData"        ,
        "$ElementNodeData"       ,
        "$EndElementNodeData"    ,
        "$InterpolationScheme"   ,
        "$EndInterpolationScheme" 
    };

std::ifstream file_;

std::string filename_;
std::string buffer_  ;



public:

MSHReader(std::string filename): filename_(filename){
    file_.open(filename_);
    if(!file_.is_open()){
        std::cerr << "Error opening file: " << filename_ << std::endl;
    }
    file_.seekg(0, std::ios::end);
    buffer_.resize(file_.tellg());
    file_.seekg(0, std::ios::beg);
    file_.read(buffer_.data(), buffer_.size());
    file_.close();
    
    for (auto const& keyword : msh_keywords){
        std::size_t pos = buffer_.find(keyword);
        if (pos != std::string::npos){
            std::cout << "Found " << keyword << " at position " << pos << std::endl;
        }
    }

    
}; 


};  

namespace gmsh{

struct MeshFormat{
    double version;
    int    file_type;
    int    data_size;
};

}


#endif // FALL_N_MESH_INTERFACE