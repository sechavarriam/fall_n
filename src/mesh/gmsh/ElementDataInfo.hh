#ifndef GMSH_MSHFILE_ELEMENT_DATA_INFORMATION
#define GMSH_MSHFILE_ELEMENT_DATA_INFORMATION


/*
$ElementData
  numStringTags(ASCII int)
  stringTag(string) ...
  numRealTags(ASCII int)
  realTag(ASCII double) ...
  numIntegerTags(ASCII int)
  integerTag(ASCII int) ...
  elementTag(int) value(double) ...
  ...
$EndElementData
*/

namespace gmsh
{
    struct ElementDataInfo
    {
        /* data */
    };
    

}// namespace gmsh

#endif // GMSH_MSHFILE_ELEMENT_DATA_INFORMATION