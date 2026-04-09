#ifndef FALL_N_SRC_RECONSTRUCTION_MATERIAL_FACTORY_HH
#define FALL_N_SRC_RECONSTRUCTION_MATERIAL_FACTORY_HH

// =============================================================================
//  MaterialFactory -- compatibility shim
// =============================================================================
//
//  The normative contracts and their current default implementations now live
//  in the materials module:
//    - src/materials/SubmodelMaterialFactory.hh
//    - src/materials/SubmodelMaterialFactoryDefaults.hh
//
//  This forwarding header is kept for source compatibility with the legacy
//  reconstruction include path and with header_files.hh.
//
// =============================================================================

#include "../materials/SubmodelMaterialFactory.hh"
#include "../materials/SubmodelMaterialFactoryDefaults.hh"

#endif // FALL_N_SRC_RECONSTRUCTION_MATERIAL_FACTORY_HH
