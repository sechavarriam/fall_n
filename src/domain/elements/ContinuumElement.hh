#ifndef FN_CONTINUUM_ELEMENT
#define FN_CONTINUUM_ELEMENT


#include <memory>

#include "ElementGeometry.hh"
#include "../../materials/Material.hh"

template <typename MaterialPolicy>
class ContinuumElement{

    std::unique_ptr<ElementGeometry> element_;



}; // Forward declaration



#endif