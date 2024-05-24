#ifndef GMSH_MSHFILE_NODE_DATA_INFORMATION
#define GMSH_MSHFILE_NODE_DATA_INFORMATION


/*
$NodeData
  numStringTags(ASCII int)
  stringTag(string) ...
  numRealTags(ASCII int)
  realTag(ASCII double) ...
  numIntegerTags(ASCII int)
  integerTag(ASCII int) ...
  nodeTag(int) value(double) ...
  ...
$EndNodeData
*/

namespace gmsh
{
    struct NodeDataInfo
    {
        /* data */
    };
    

}// namespace gmsh

#endif // GMSH_MSHFILE_NODE_DATA_INFORMATION