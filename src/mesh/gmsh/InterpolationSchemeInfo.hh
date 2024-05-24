#ifndef GMSH_MSHFILE_INTERPOLATION_SCHEME_INFORMATION
#define GMSH_MSHFILE_INTERPOLATION_SCHEME_INFORMATION


/*
$InterpolationScheme
  name(string)
  numElementTopologies(ASCII int)
  elementTopology
  numInterpolationMatrices(ASCII int)
  numRows(ASCII int) numColumns(ASCII int) value(ASCII double) ...
$EndInterpolationScheme
*/
namespace gmsh
{
    struct InterpolationSchemeInfo
    {
        /* data */
    };
    

}// namespace gmsh

#endif // GMSH_MSHFILE_INTERPOLATION_SCHEME_INFORMATION