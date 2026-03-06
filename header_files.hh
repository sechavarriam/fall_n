#include <Eigen/Dense>
#include <petsc.h>

#include <cstddef>
#include <array>
#include <concepts>
#include <functional>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <ranges>
#include <tuple>
#include <utility>

#include <charconv>
#include <string>
#include <string_view>
#include <fstream>
#include <filesystem>
    
//================================================================================
// Geometry Module
//================================================================================

#include "src/geometry/geometry.hh"
#include "src/geometry/Topology.hh"
#include "src/geometry/Cell.hh"
#include "src/geometry/Point.hh"

//================================================================================
// Elements Module
//================================================================================

#include "src/elements/Node.hh"
#include "src/elements/Section.hh"
//#include "src/elements/NodalSection.hh"

#include "src/elements/FEM_Element.hh"
#include "src/elements/ContinuumElement.hh"
#include "src/elements/StructuralElement.hh"

#include "src/elements/element_geometry/ElementGeometry.hh"
#include "src/elements/element_geometry/LagrangeElement.hh"


// =================================================================================
// Numerics Module
// =================================================================================

// #include "src/numerics/Polynomial.hh"
// #include "src/numerics/Tensor.hh"

#include "src/numerics/Interpolation/GenericInterpolant.hh"
#include "src/numerics/Interpolation/LagrangeInterpolation.hh"

#include "src/numerics/numerical_integration/Quadrature.hh"
#include "src/numerics/numerical_integration/GaussLegendreNodes.hh"
#include "src/numerics/numerical_integration/GaussLegendreWeights.hh"

#include "src/numerics/linear_algebra/Matrix.hh"
#include "src/numerics/linear_algebra/Vector.hh"
#include "src/numerics/linear_algebra/LinalgOperations.hh"

// =================================================================================
// Material Module
// =================================================================================

// --- Low-level types (Voigt notation, beam measures) ---
#include "src/materials/VoigtVector.hh"
#include "src/materials/Strain.hh"
#include "src/materials/Stress.hh"

#include "src/materials/beam/BeamGeneralizedStrain.hh"
#include "src/materials/beam/BeamSectionForces.hh"

// --- Policy / State infrastructure ---
#include "src/materials/MaterialPolicy.hh"
#include "src/materials/MaterialStatePolicy.hh"
#include "src/materials/MaterialState.hh"

// --- Concept hierarchy ---
#include "src/materials/ConstitutiveRelation.hh"

// --- Concrete constitutive relations ---
#include "src/materials/constitutive_models/lineal/ElasticRelation.hh"
#include "src/materials/constitutive_models/lineal/IsotropicRelation.hh"
#include "src/materials/constitutive_models/lineal/TimoshenkoBeamSection.hh"
#include "src/materials/constitutive_models/non_lineal/InelasticRelation.hh"

// --- Material instance (MaterialInstance<R, StatePolicy>) + aliases ---
#include "src/materials/LinealElasticMaterial.hh"

// --- Integration strategies (ElasticUpdate, InelasticUpdate) ---
#include "src/materials/update_strategy/IntegrationStrategy.hh"

// --- Type-erased wrapper (Material<Policy>) ---
#include "src/materials/Material.hh"


// =================================================================================
// Model Module
// =================================================================================

#include "src/model/DoF.hh"
#include "src/model/Model.hh"
#include "src/model/ModelBuilder.hh"

// =================================================================================
// Analysis Module
// =================================================================================

#include "src/analysis/Analysis.hh"

// =================================================================================
// Mesh Module
// =================================================================================
#include "src/mesh/Mesh.hh"

#include "src/mesh/gmsh/ReadGmsh.hh"
#include "src/mesh/gmsh/GmshDomainBuilder.hh"

//#include "src/graph/AdjacencyList.hh"
//#include "src/graph/AdjacencyMatrix.hh"

// =================================================================================
// Post-processing Module
// =================================================================================

#include "src/post-processing/VTK/VTKheaders.hh"
#include "src/post-processing/VTK/VTKwriter.hh"