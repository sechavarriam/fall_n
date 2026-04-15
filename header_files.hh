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
#include "src/geometry/SerendipityCell.hh"
#include "src/geometry/SimplexCell.hh"
#include "src/geometry/Point.hh"
#include "src/geometry/Vertex.hh"

//================================================================================
// Elements Module
//================================================================================

#include "src/elements/Node.hh"
#include "src/elements/Section.hh"
//#include "src/elements/NodalSection.hh"

#include "src/elements/FEM_Element.hh"
#include "src/elements/ContinuumElement.hh"
#include "src/elements/StructuralElement.hh"
#include "src/elements/BeamElement.hh"
#include "src/elements/TimoshenkoBeamN.hh"
#include "src/elements/ShellElement.hh"
#include "src/elements/MITCShellPolicy.hh"
#include "src/elements/ShellKinematicPolicy.hh"
#include "src/elements/MITCShellElement.hh"
#include "src/elements/ElementPolicy.hh"

#include "src/elements/element_geometry/ElementGeometry.hh"
#include "src/elements/element_geometry/LagrangeElement.hh"
#include "src/elements/element_geometry/SerendipityElement.hh"
#include "src/elements/element_geometry/SerendipitySimplexElement.hh"
#include "src/elements/element_geometry/SimplexElement.hh"


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
#include "src/numerics/numerical_integration/GaussLobattoNodes.hh"
#include "src/numerics/numerical_integration/GaussLobattoWeights.hh"
#include "src/numerics/numerical_integration/GaussLobattoCellIntegrator.hh"
#include "src/numerics/numerical_integration/GaussRadauNodes.hh"
#include "src/numerics/numerical_integration/GaussRadauWeights.hh"
#include "src/numerics/numerical_integration/GaussRadauCellIntegrator.hh"
#include "src/numerics/numerical_integration/BeamAxisQuadrature.hh"
#include "src/numerics/numerical_integration/SimplexQuadrature.hh"
#include "src/numerics/numerical_integration/StroudConicalProduct.hh"

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

#include "src/materials/shell/ShellGeneralizedStrain.hh"
#include "src/materials/shell/ShellResultants.hh"

// --- Policy / State infrastructure ---
#include "src/materials/MaterialPolicy.hh"
#include "src/materials/MaterialStatePolicy.hh"
#include "src/materials/MaterialState.hh"
#include "src/materials/ConstitutiveState.hh"
#include "src/materials/ConstitutiveIntegrator.hh"
#include "src/materials/ConstitutiveProtocol.hh"
#include "src/materials/local_problem/LocalLinearSolver.hh"
#include "src/materials/local_problem/LocalStepControl.hh"
#include "src/materials/local_problem/LocalNonlinearProblem.hh"
#include "src/materials/local_problem/ContinuumLocalProblem.hh"
#include "src/materials/local_problem/NewtonLocalSolver.hh"
#include "src/materials/SectionConstitutiveSnapshot.hh"

// --- Concept hierarchy ---
#include "src/materials/ConstitutiveRelation.hh"

// --- Concrete constitutive relations (elastic) ---
#include "src/materials/constitutive_models/lineal/ElasticRelation.hh"
#include "src/materials/constitutive_models/lineal/IsotropicRelation.hh"
#include "src/materials/constitutive_models/lineal/TimoshenkoBeamSection.hh"
#include "src/materials/constitutive_models/lineal/MindlinShellSection.hh"

// --- Plasticity building blocks ---
#include "src/materials/constitutive_models/non_lineal/plasticity/PlasticityConcepts.hh"
#include "src/materials/constitutive_models/non_lineal/plasticity/VonMises.hh"
#include "src/materials/constitutive_models/non_lineal/plasticity/IsotropicHardening.hh"
#include "src/materials/constitutive_models/non_lineal/plasticity/AssociatedFlow.hh"
#include "src/materials/constitutive_models/non_lineal/plasticity/YieldFunction.hh"
#include "src/materials/constitutive_models/non_lineal/plasticity/ConsistencyFunction.hh"
#include "src/materials/constitutive_models/non_lineal/plasticity/ScalarConsistencyProblem.hh"
#include "src/materials/constitutive_models/non_lineal/plasticity/ReturnAlgorithm.hh"

// --- Composed plasticity relation + backward-compat aliases ---
#include "src/materials/constitutive_models/non_lineal/PlasticityRelation.hh"
#include "src/materials/constitutive_models/non_lineal/InelasticRelation.hh"

// --- Uniaxial cyclic materials (Phase 2) ---
#include "src/materials/constitutive_models/non_lineal/MenegottoPintoSteel.hh"
#include "src/materials/constitutive_models/non_lineal/KentParkConcrete.hh"

// --- Fiber section (Phase 3) ---
#include "src/materials/constitutive_models/non_lineal/FiberSection.hh"

// --- Material instance (MaterialInstance<R, StatePolicy>) + aliases ---
#include "src/materials/LinealElasticMaterial.hh"

// --- Integration strategies (ElasticUpdate, InelasticUpdate) ---
#include "src/materials/update_strategy/IntegrationStrategy.hh"

// --- Type-erased wrapper (Material<Policy>) ---
#include "src/materials/Material.hh"

// --- Arena allocator for bulk Material creation (Phase 6) ---
#include "src/materials/ArenaAllocator.hh"

// --- Section property utilities ---
#include "src/utils/SectionProperties.hh"

// --- Fiber section factory (patch/rebar helpers + uniaxial material factories) ---
#include "src/materials/FiberSectionFactory.hh"

// --- RC section builders (column/beam) ---
#include "src/materials/RCSectionBuilder.hh"


// =================================================================================
// Model Module
// =================================================================================

#include "src/continuum/Continuum.hh"

#include "src/model/DoFStorage.hh"
#include "src/model/DoF.hh"       // Legacy — kept for transition (see DoFStorage.hh)
#include "src/model/NodeSelector.hh"
#include "src/model/ModelCheckpoint.hh"
#include "src/model/ModelState.hh"
#include "src/model/Model.hh"
#include "src/model/ModelBuilder.hh"
#include "src/model/BoundaryCondition.hh"

// --- Load application utilities ---
#include "src/model/LoadUtilities.hh"

// --- Building domain + structural model builders (Phase 3) ---
#include "src/model/BuildingDomainBuilder.hh"
#include "src/model/StructuralModelBuilder.hh"

// --- Prismatic hex mesh builder (Phase 3) ---
#include "src/model/PrismaticDomainBuilder.hh"

// --- Ground motion record parser (seismic analysis) ---
#include "src/model/GroundMotionRecord.hh"

// =================================================================================
// Reconstruction Module
// =================================================================================

#include "src/reconstruction/SectionProfile.hh"
#include "src/reconstruction/StructuralFieldReconstruction.hh"
#include "src/reconstruction/FieldTransfer.hh"
#include "src/reconstruction/SubModelSolver.hh"
#include "src/reconstruction/SubModelEvolver.hh"
#include "src/reconstruction/NonlinearSubModelEvolver.hh"
#include "src/reconstruction/HomogenizedSection.hh"
#include "src/reconstruction/LocalModelAdapter.hh"
#include "src/reconstruction/MaterialFactory.hh"
#include "src/reconstruction/HomogenisationStrategy.hh"

// =================================================================================
// Utils Module
// =================================================================================

#include "src/utils/Benchmark.hh"
#include "src/utils/CyclicProtocol.hh"

// =================================================================================
// Analysis Module
// =================================================================================

#include "src/analysis/Analysis.hh"

// --- Multiscale coordinator (Phase 5) ---
#include "src/analysis/MultiscaleCoordinator.hh"
#include "src/analysis/MultiscaleTypes.hh"
#include "src/analysis/MicroSolveExecutor.hh"
#include "src/analysis/BeamMacroBridge.hh"
#include "src/analysis/CouplingStrategy.hh"
#include "src/analysis/MultiscaleModel.hh"
#include "src/analysis/MultiscaleAnalysis.hh"

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

#include "src/post-processing/VTK/VTKModelExporter.hh"
#include "src/post-processing/VTK/VTKConstitutiveCurveWriter.hh"
#include "src/post-processing/VTK/StructuralVTMExporter.hh"
#include "src/post-processing/VTK/PVDWriter.hh"
#include "src/post-processing/StateQuery.hh"
