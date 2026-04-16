#ifndef FALL_N_BEAM_VALIDATION_SUPPORT_HH
#define FALL_N_BEAM_VALIDATION_SUPPORT_HH

// =============================================================================
//  BeamValidationSupport.hh
// =============================================================================
//
//  Narrow validation/testing umbrella for the beam slice.
//
//  Purpose:
//    - replace ad-hoc dependence on the repository-wide header_files.hh for
//      beam-centric validation surfaces,
//    - keep compile-time coupling local to the beam validation campaign,
//    - provide a reusable migration path toward focused module umbrellas
//      instead of one giant project umbrella.
//
//  This header is intentionally scoped to the current reduced-column /
//  beam-validation campaign. It is not a general public API.
//
// =============================================================================

#include "../elements/BeamElement.hh"
#include "../elements/Node.hh"
#include "../elements/TimoshenkoBeamN.hh"
#include "../elements/element_geometry/ElementGeometry.hh"
#include "../elements/element_geometry/LagrangeElement.hh"
#include "../geometry/Vertex.hh"
#include "../materials/LinealElasticMaterial.hh"
#include "../numerics/numerical_integration/BeamAxisQuadrature.hh"
#include "../numerics/numerical_integration/GaussLegendreCellIntegrator.hh"
#include "../numerics/numerical_integration/GaussLobattoCellIntegrator.hh"
#include "../numerics/numerical_integration/GaussRadauCellIntegrator.hh"

#endif // FALL_N_BEAM_VALIDATION_SUPPORT_HH
