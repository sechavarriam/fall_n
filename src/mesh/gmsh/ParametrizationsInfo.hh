#ifndef GMSH_MSHFILE_PARAMETRIZATIONS_INFORMATION
#define GMSH_MSHFILE_PARAMETRIZATIONS_INFORMATION



/*
$Parametrizations
  numCurveParam(size_t) numSurfaceParam(size_t)
  curveTag(int) numNodes(size_t)
    nodeX(double) nodeY(double) nodeZ(double) nodeU(double)
    ...
  ...
  surfaceTag(int) numNodes(size_t) numTriangles(size_t)
    nodeX(double) nodeY(double) nodeZ(double)
      nodeU(double) nodeV(double)
      curvMaxX(double) curvMaxY(double) curvMaxZ(double)
      curvMinX(double) curvMinY(double) curvMinZ(double)
    ...
    nodeIndex1(int) nodeIndex2(int) nodeIndex3(int)
    ...
  ...
$EndParametrizations
*/

namespace gmsh
{
    struct ParametrizationsInfo
    {
        /* data */
    };
    

}// namespace gmsh

#endif // GMSH_MSHFILE_PARAMETRIZATIONS_INFORMATION