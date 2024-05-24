#ifndef GMSH_MSHFILE_GHOST_ELEMENTS_INFORMATION
#define GMSH_MSHFILE_GHOST_ELEMENTS_INFORMATION


/*
$GhostElements
  numGhostElements(size_t)
  elementTag(size_t) partitionTag(int)
    numGhostPartitions(size_t) ghostPartitionTag(int) ...
  ...
$EndGhostElements
*/

namespace gmsh
{
    struct GhostElementsInfo
    {
        /* data */
    };
    

}// namespace gmsh

#endif // GMSH_MSHFILE_GHOST_ELEMENTS_INFORMATION