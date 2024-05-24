#ifndef GMSH_MSHFILE_ELEMENT_NODE_DATA_INFORMATION
#define GMSH_MSHFILE_ELEMENT_NODE_DATA_INFORMATION

/*
$ElementNodeData
  numStringTags(ASCII int)
  stringTag(string) ...
  numRealTags(ASCII int)
  realTag(ASCII double) ...
  numIntegerTags(ASCII int)
  integerTag(ASCII int) ...
  elementTag(int) numNodesPerElement(int) value(double) ...
  ...
$EndElementNodeData
*/

namespace gmsh
{
    struct ElementNodeDataInfo
    {
        /* data */
    };
    

}// namespace gmsh

#endif // GMSH_MSHFILE_ELEMENT_NODE_DATA_INFORMATION