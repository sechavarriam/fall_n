#ifndef GMSH_PERIODIC_INFORMATION
#define GMSH_PERIODIC_INFORMATION

/*
$Periodic
  numPeriodicLinks(size_t)
  entityDim(int) entityTag(int) entityTagMaster(int)
  numAffine(size_t) value(double) ...
  numCorrespondingNodes(size_t)
    nodeTag(size_t) nodeTagMaster(size_t)
    ...
  ...
$EndPeriodic
*/

namespace gmsh
{
    struct PeriodicInfo
    {
        /* data */
    };
    

}// namespace gmsh

#endif // GMSH_PERIODIC_INFORMATION