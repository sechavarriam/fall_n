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
#include <optional>

#include "../Mesh.hh"
#include "../../domain/Domain.hh"

#include "GmshElementTypes.hh"
#include "MeshFormatInfo.hh"
#include "PhysicalNamesInfo.hh"
#include "EntitiesInfo.hh"
#include "PartitionedEntitiesInfo.hh"
#include "NodesInfo.hh"
#include "ElementInfo.hh"
#include "PeriodicInfo.hh"
#include "GhostElementsInfo.hh"
#include "ParametrizationsInfo.hh"
#include "NodeDataInfo.hh"
#include "ElementDataInfo.hh"
#include "ElementNodeDataInfo.hh"
#include "InterpolationSchemeInfo.hh"

// namespace mesh {
namespace gmsh
{
    class MSHReader
    {
        using MeshFormat          = MeshFormatInfo;
        using PhysicalNames       = std::optional<PhysicalNamesInfo>;
        using Entities            = std::optional<EntitiesInfo>;
        using PartitionedEntities = std::optional<PartitionedEntitiesInfo>;
        using Nodes               = NodesInfo;
        using Elements            = ElementInfo;
        using Periodic            = std::optional<PeriodicInfo>;
        using GhostElements       = std::optional<GhostElementsInfo>;
        using Parametrizations    = std::optional<ParametrizationsInfo>;
        using NodeData            = std::optional<NodeDataInfo>;
        using ElementData         = std::optional<ElementDataInfo>;
        using ElementNodeData     = std::optional<ElementNodeDataInfo>;
        using InterpolationScheme = std::optional<InterpolationSchemeInfo>;

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

    public:
        MeshFormat          mesh_format_info_;
        PhysicalNames       physical_names_info_;
        Entities            entities_info_;
        PartitionedEntities partitioned_entities_info_;
        Nodes               nodes_info_;
        Elements            elements_info_;
        Periodic            periodic_info_;
        GhostElements       ghost_elements_info_;
        Parametrizations    parametrizations_info_;
        NodeData            node_data_info_;
        ElementData         element_data_info_;
        ElementNodeData     element_node_data_info_;
        InterpolationScheme interpolation_scheme_info_;
    

        std::array<std::size_t, msh_keywords.size()> keyword_position_{
            [](){ // Inmediately invoked lambda expression to initialize the array with zeros 
                std::array<std::size_t, msh_keywords.size()> arr{};
                arr.fill(0);
                return arr;
                }()};

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
        }

        inline constexpr auto view_MeshFormat()          { return view_keyword_info<0>(); };
        inline constexpr auto view_PhysicalNames()       { return view_keyword_info<2>(); };
        inline constexpr auto view_Entities()            { return view_keyword_info<4>(); };
        inline constexpr auto view_PartitionedEntities() { return view_keyword_info<6>(); };
        inline constexpr auto view_Nodes()               { return view_keyword_info<8>(); };
        inline constexpr auto view_Elements()            { return view_keyword_info<10>(); };
        inline constexpr auto view_Periodic()            { return view_keyword_info<12>(); };
        inline constexpr auto view_GhostElements()       { return view_keyword_info<14>(); };
        inline constexpr auto view_Parametrizations()    { return view_keyword_info<16>(); };
        inline constexpr auto view_NodeData()            { return view_keyword_info<18>(); };
        inline constexpr auto view_ElementData()         { return view_keyword_info<20>(); };
        inline constexpr auto view_ElementNodeData()     { return view_keyword_info<22>(); };
        inline constexpr auto view_InterpolationScheme() { return view_keyword_info<24>(); };

        MSHReader(std::string_view filename) : filename_(filename)
        {
            // https://stackoverflow.com/questions/18398167/how-to-copy-a-txt-file-to-a-char-array-in-c
            std::ifstream in(filename_);
            if (!in){std::cerr << "Cannot open the File : " << filename_ << std::endl;}

            buffer_ = std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

            std::size_t i = 0;
            for (auto const &keyword : msh_keywords)
            {                                            // sintetizar en función. set_positions();
                std::size_t pos = buffer_.find(keyword); // Se puede hacer mas efiiente aprovechando el orden de los keywords.
                if (pos != std::string::npos) keyword_position_[i] = pos;
                ++i;
            }
            buffer_view_ = std::string_view(buffer_.data());

            mesh_format_info_          = MeshFormat         (view_MeshFormat());
            physical_names_info_       = PhysicalNames      (view_PhysicalNames());
            entities_info_             = Entities           (view_Entities());
            //partitioned_entities_info_ = PartitionedEntities(view_PartitionedEntities());
            nodes_info_                = Nodes              (view_Nodes());
            elements_info_             = Elements           (view_Elements());
            //periodic_info_             = Periodic           (view_Periodic());
            //ghost_elements_info_       = GhostElements      (view_GhostElements());
            //parametrizations_info_     = Parametrizations   (view_Parametrizations());
            //node_data_info_            = NodeData           (view_NodeData());
            //element_data_info_         = ElementData        (view_ElementData());
            //element_node_data_info_    = ElementNodeData    (view_ElementNodeData());
            //interpolation_scheme_info_ = InterpolationScheme(view_InterpolationScheme());

        };
    };

};// namespace gmsh

#endif // FALL_N_MESH_INTERFACE


/*
//RAW TO TEST IN MAIN
    std::cout <<  reader.view_MeshFormat() << std::endl;

    std::cout << "=======================================================================================" << std::endl;
    gmsh::MeshFormatInfo mesh_format(reader.view_MeshFormat());
    std::cout << "Version:   " << mesh_format.version   << " data size: " << sizeof(mesh_format.version  ) << std::endl;
    std::cout << "File Type: " << mesh_format.file_type << " data size: " << sizeof(mesh_format.file_type) << std::endl;
    std::cout << "Data Size: " << mesh_format.data_size << " data size: " << sizeof(mesh_format.data_size) << std::endl;    

    std::cout << "=======================================================================================" << std::endl;
    gmsh::PhysicalNamesInfo physical_names(reader.view_PhysicalNames());
    std::cout << "Number of Physical Names: " << physical_names.numPhysicalNames << std::endl;
    physical_names.print_raw();

    std::cout << "=======================================================================================" << std::endl;

    gmsh::EntitiesInfo entities(reader.view_Entities());
    std::cout << "Number of Entities per type : " << std::endl;
    for (auto const& e : entities.num_entities) std::cout << e << " "; std::cout << std::endl;

    std::cout << "_______________________________________________________________________________________" << std::endl;
    for (auto p : entities.points) p.print_raw();
    std::cout << "_______________________________________________________________________________________" << std::endl;
    for (auto c : entities.curves) c.print_raw();
    std::cout << "_______________________________________________________________________________________" << std::endl;
    for (auto s : entities.surfaces) s.print_raw();
    std::cout << "_______________________________________________________________________________________" << std::endl;
    for (auto v : entities.volumes) v.print_raw();

    std::cout << "=======================================================================================" << std::endl;
    
    gmsh::NodesInfo nodes(reader.view_Nodes());
    std::cout << "numEntityBlocks: " << nodes.numEntityBlocks << std::endl;
    std::cout << "numNodes:        " << nodes.numNodes << std::endl;
    std::cout << "minNodeTag:      " << nodes.minNodeTag << std::endl;
    std::cout << "maxNodeTag:      " << nodes.maxNodeTag << std::endl;
    std::cout << "_______________________________________________________________________________________" << std::endl;

    for (auto block : nodes.entityBlocks){
        block.print_raw();
        std::cout << "-----------------------------------------------------" << std::endl;
    }
    
    std::cout << "=======================================================================================" << std::endl;
    gmsh::ElementInfo elements(reader.view_Elements());
    std::cout << "numEntityBlocks: " << elements.numEntityBlocks << std::endl;
    std::cout << "numElements:     " << elements.numElements << std::endl;
    std::cout << "minElementTag:   " << elements.minElementTag << std::endl;
    std::cout << "maxElementTag:   " << elements.maxElementTag << std::endl;
    std::cout << "_______________________________________________________________________________________" << std::endl;

    for (auto block : elements.entityBlocks){
        block.print_raw();
        std::cout << "-----------------------------------------------------" << std::endl;
    }
    std::cout << "=======================================================================================" << std::endl;
*/