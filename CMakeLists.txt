cmake_minimum_required(VERSION 3.21.0)
project(fall_n VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent) #
include(CTest)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Werror")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Werror -fsanitize=address ")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Werror -fsanitize=leak -fsanitize=address ")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Werror -fsanitize=leak -fsanitize=address -static-libasan ")


#if     (CMAKE_CXX_COMPILER_ID STREQUAL "Clang") # using Clang
#elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU"  ) # using GCC
#elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel") # using Intel C++
#elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" ) # using Visual Studio C++
#endif()

add_executable(fall_n main.cpp)
enable_testing()


set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# ===============================================================================
# ============== MPI ============================================================
# ===============================================================================

find_package(MPI REQUIRED) # https://stackoverflow.com/questions/23163075/how-to-compile-an-mpi-included-c-program-using-cmake
# add_definitions(-DOMPI_SKIP_MPICXX) # add this line only when you are using openmpi which has a different c++ bindings
target_link_libraries(fall_n PUBLIC MPI::MPI_CXX)


# ===============================================================================
# ============== PETSc ==========================================================
# ===============================================================================

set(PETSC $ENV{PETSC_DIR}/$ENV{PETSC_ARCH}) # set root of location to find PETSc's pkg-config
set(ENV{PKG_CONFIG_PATH} ${PETSC}/lib/pkgconfig)
  
find_package(PkgConfig REQUIRED)
pkg_search_module(PETSC REQUIRED IMPORTED_TARGET PETSc)
target_link_libraries(fall_n PUBLIC PkgConfig::PETSC)

# ===============================================================================
# ============== Eigen ==========================================================
# ===============================================================================

find_package(Eigen3 REQUIRED)
target_link_libraries(fall_n PUBLIC Eigen3::Eigen)


# ===============================================================================
# ============== VTK ============================================================
# ===============================================================================
find_package(VTK COMPONENTS 
  CommonCore
  CommonDataModel
  IOXML
)

set(VTK_USE_MPI ON)

if (NOT VTK_FOUND)
  message(FATAL_ERROR "CylinderExample: Unable to find the VTK build folder.")
endif()

target_link_libraries(fall_n PUBLIC ${VTK_LIBRARIES})

vtk_module_autoinit(
  TARGETS fall_n
  MODULES ${VTK_LIBRARIES}
)

# ==============================================================================
# ============== Tests ==========================================================
# ==============================================================================

add_executable(fall_n_tests
  tests/test_element_geometry.cpp
)

target_include_directories(fall_n_tests PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(fall_n_tests PRIVATE MPI::MPI_CXX)
target_link_libraries(fall_n_tests PRIVATE PkgConfig::PETSC)
target_link_libraries(fall_n_tests PRIVATE Eigen3::Eigen)
target_link_libraries(fall_n_tests PRIVATE ${VTK_LIBRARIES})

vtk_module_autoinit(
  TARGETS fall_n_tests
  MODULES ${VTK_LIBRARIES}
)

add_test(NAME element_geometry COMMAND fall_n_tests)

# --- Simplex (Tetrahedral) Element Family test ---
add_executable(fall_n_simplex_test
  tests/test_simplex.cpp
)

target_include_directories(fall_n_simplex_test PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(fall_n_simplex_test PRIVATE MPI::MPI_CXX)
target_link_libraries(fall_n_simplex_test PRIVATE PkgConfig::PETSC)
target_link_libraries(fall_n_simplex_test PRIVATE Eigen3::Eigen)
target_link_libraries(fall_n_simplex_test PRIVATE ${VTK_LIBRARIES})

vtk_module_autoinit(
  TARGETS fall_n_simplex_test
  MODULES ${VTK_LIBRARIES}
)

add_test(NAME simplex_elements COMMAND fall_n_simplex_test)

# --- VTK tensor-field derivative test ---
add_executable(fall_n_vtk_tensor_fields_test
  tests/test_vtk_tensor_field_derivatives.cpp
)

target_include_directories(fall_n_vtk_tensor_fields_test PRIVATE ${CMAKE_SOURCE_DIR})

add_test(NAME vtk_tensor_field_derivatives COMMAND fall_n_vtk_tensor_fields_test)

# --- VTK Gauss-cloud contract test ---
add_executable(fall_n_vtk_gauss_cloud_test
  tests/test_vtk_gauss_cloud_contract.cpp
)

target_link_libraries(fall_n_vtk_gauss_cloud_test PRIVATE ${VTK_LIBRARIES})

vtk_module_autoinit(
  TARGETS fall_n_vtk_gauss_cloud_test
  MODULES ${VTK_LIBRARIES}
)

add_test(NAME vtk_gauss_cloud_contract COMMAND fall_n_vtk_gauss_cloud_test)

# --- Material constitutive curves VTK test ---
add_executable(fall_n_material_curves_test
  tests/test_material_curves_vtk.cpp
)

target_include_directories(fall_n_material_curves_test PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(fall_n_material_curves_test PRIVATE MPI::MPI_CXX)
target_link_libraries(fall_n_material_curves_test PRIVATE PkgConfig::PETSC)
target_link_libraries(fall_n_material_curves_test PRIVATE Eigen3::Eigen)
target_link_libraries(fall_n_material_curves_test PRIVATE ${VTK_LIBRARIES})

vtk_module_autoinit(
  TARGETS fall_n_material_curves_test
  MODULES ${VTK_LIBRARIES}
)

add_test(NAME material_curves_vtk COMMAND fall_n_material_curves_test)

# --- Structural reconstruction test ---
add_executable(fall_n_structural_reconstruction_test
  tests/test_structural_reconstruction.cpp
)

target_include_directories(fall_n_structural_reconstruction_test PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(fall_n_structural_reconstruction_test PRIVATE MPI::MPI_CXX)
target_link_libraries(fall_n_structural_reconstruction_test PRIVATE PkgConfig::PETSC)
target_link_libraries(fall_n_structural_reconstruction_test PRIVATE Eigen3::Eigen)
target_link_libraries(fall_n_structural_reconstruction_test PRIVATE ${VTK_LIBRARIES})

vtk_module_autoinit(
  TARGETS fall_n_structural_reconstruction_test
  MODULES ${VTK_LIBRARIES}
)

add_test(NAME structural_reconstruction COMMAND fall_n_structural_reconstruction_test)

# --- Structural VTM export test ---
add_executable(fall_n_structural_vtm_test
  tests/test_structural_vtm_export.cpp
)

target_include_directories(fall_n_structural_vtm_test PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(fall_n_structural_vtm_test PRIVATE MPI::MPI_CXX)
target_link_libraries(fall_n_structural_vtm_test PRIVATE PkgConfig::PETSC)
target_link_libraries(fall_n_structural_vtm_test PRIVATE Eigen3::Eigen)
target_link_libraries(fall_n_structural_vtm_test PRIVATE ${VTK_LIBRARIES})

vtk_module_autoinit(
  TARGETS fall_n_structural_vtm_test
  MODULES ${VTK_LIBRARIES}
)

add_test(NAME structural_vtm_export COMMAND fall_n_structural_vtm_test)

# --- Mixed structural VTM export test ---
add_executable(fall_n_structural_mixed_vtm_test
  tests/test_structural_mixed_vtm_export.cpp
)

target_include_directories(fall_n_structural_mixed_vtm_test PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(fall_n_structural_mixed_vtm_test PRIVATE MPI::MPI_CXX)
target_link_libraries(fall_n_structural_mixed_vtm_test PRIVATE PkgConfig::PETSC)
target_link_libraries(fall_n_structural_mixed_vtm_test PRIVATE Eigen3::Eigen)
target_link_libraries(fall_n_structural_mixed_vtm_test PRIVATE ${VTK_LIBRARIES})

vtk_module_autoinit(
  TARGETS fall_n_structural_mixed_vtm_test
  MODULES ${VTK_LIBRARIES}
)

add_test(NAME structural_mixed_vtm_export COMMAND fall_n_structural_mixed_vtm_test)

# --- Serendipity element family test ---
add_executable(fall_n_serendipity_test
  tests/test_serendipity.cpp
)

target_include_directories(fall_n_serendipity_test PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(fall_n_serendipity_test PRIVATE MPI::MPI_CXX)
target_link_libraries(fall_n_serendipity_test PRIVATE PkgConfig::PETSC)
target_link_libraries(fall_n_serendipity_test PRIVATE Eigen3::Eigen)
target_link_libraries(fall_n_serendipity_test PRIVATE ${VTK_LIBRARIES})

vtk_module_autoinit(
  TARGETS fall_n_serendipity_test
  MODULES ${VTK_LIBRARIES}
)

add_test(NAME serendipity_family COMMAND fall_n_serendipity_test)

# --- Material Strategy test ---
add_executable(fall_n_strategy_test
  tests/test_material_strategy.cpp
)

target_include_directories(fall_n_strategy_test PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(fall_n_strategy_test PRIVATE MPI::MPI_CXX)
target_link_libraries(fall_n_strategy_test PRIVATE PkgConfig::PETSC)
target_link_libraries(fall_n_strategy_test PRIVATE Eigen3::Eigen)
target_link_libraries(fall_n_strategy_test PRIVATE ${VTK_LIBRARIES})

vtk_module_autoinit(
  TARGETS fall_n_strategy_test
  MODULES ${VTK_LIBRARIES}
)

add_test(NAME material_strategy COMMAND fall_n_strategy_test)

# --- Plasticity composition test ---
add_executable(fall_n_plasticity_test
  tests/test_plasticity_composition.cpp
)

target_include_directories(fall_n_plasticity_test PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(fall_n_plasticity_test PRIVATE MPI::MPI_CXX)
target_link_libraries(fall_n_plasticity_test PRIVATE PkgConfig::PETSC)
target_link_libraries(fall_n_plasticity_test PRIVATE Eigen3::Eigen)
target_link_libraries(fall_n_plasticity_test PRIVATE ${VTK_LIBRARIES})

vtk_module_autoinit(
  TARGETS fall_n_plasticity_test
  MODULES ${VTK_LIBRARIES}
)

add_test(NAME plasticity_composition COMMAND fall_n_plasticity_test)

# --- Point / geometry test ---
add_executable(fall_n_point_test
  tests/test_point.cpp
)

target_include_directories(fall_n_point_test PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(fall_n_point_test PRIVATE MPI::MPI_CXX)
target_link_libraries(fall_n_point_test PRIVATE PkgConfig::PETSC)
target_link_libraries(fall_n_point_test PRIVATE Eigen3::Eigen)

add_test(NAME point_geometry COMMAND fall_n_point_test)

# ── Node + DoF Storage Policy tests ──────────────────────────────────────────
add_executable(fall_n_node_test
  tests/test_node_dof.cpp
)

target_include_directories(fall_n_node_test PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(fall_n_node_test PRIVATE MPI::MPI_CXX)
target_link_libraries(fall_n_node_test PRIVATE PkgConfig::PETSC)
target_link_libraries(fall_n_node_test PRIVATE Eigen3::Eigen)

add_test(NAME node_dof_storage COMMAND fall_n_node_test)

# ── Domain / Vertex refactor tests ───────────────────────────────────────────
add_executable(fall_n_domain_test
  tests/test_domain_vertex.cpp
)

target_include_directories(fall_n_domain_test PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(fall_n_domain_test PRIVATE MPI::MPI_CXX)
target_link_libraries(fall_n_domain_test PRIVATE PkgConfig::PETSC)
target_link_libraries(fall_n_domain_test PRIVATE Eigen3::Eigen)
target_link_libraries(fall_n_domain_test PRIVATE ${VTK_LIBRARIES})

vtk_module_autoinit(
  TARGETS fall_n_domain_test
  MODULES ${VTK_LIBRARIES}
)

add_test(NAME domain_vertex_refactor COMMAND fall_n_domain_test)

# ── DMPlex mixed-topology tests ───────────────────────────────────────────────
add_executable(fall_n_dmplex_mixed_test
  tests/test_dmplex_mixed_topology.cpp
)

target_include_directories(fall_n_dmplex_mixed_test PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(fall_n_dmplex_mixed_test PRIVATE MPI::MPI_CXX)
target_link_libraries(fall_n_dmplex_mixed_test PRIVATE PkgConfig::PETSC)
target_link_libraries(fall_n_dmplex_mixed_test PRIVATE Eigen3::Eigen)
target_link_libraries(fall_n_dmplex_mixed_test PRIVATE ${VTK_LIBRARIES})

vtk_module_autoinit(
  TARGETS fall_n_dmplex_mixed_test
  MODULES ${VTK_LIBRARIES}
)

add_test(NAME dmplex_mixed_topology COMMAND fall_n_dmplex_mixed_test)

# ── Static Condensation tests ─────────────────────────────────────────────────
add_executable(fall_n_condensation_test
  tests/test_condensation.cpp
)

target_include_directories(fall_n_condensation_test PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(fall_n_condensation_test PRIVATE Eigen3::Eigen)

add_test(NAME static_condensation COMMAND fall_n_condensation_test)

# ── Assembly Policy tests ──────────────────────────────────────────────────────
add_executable(fall_n_assembly_test
  tests/test_assembly_policy.cpp
)

target_include_directories(fall_n_assembly_test PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(fall_n_assembly_test PRIVATE Eigen3::Eigen)

add_test(NAME assembly_policy COMMAND fall_n_assembly_test)

# ── Section tests (SectionGeometry + NodeSection + MaterialSection) ────────
add_executable(fall_n_section_test
  tests/test_section.cpp
)

target_include_directories(fall_n_section_test PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(fall_n_section_test PRIVATE MPI::MPI_CXX)
target_link_libraries(fall_n_section_test PRIVATE PkgConfig::PETSC)
target_link_libraries(fall_n_section_test PRIVATE Eigen3::Eigen)
target_link_libraries(fall_n_section_test PRIVATE ${VTK_LIBRARIES})

vtk_module_autoinit(
  TARGETS fall_n_section_test
  MODULES ${VTK_LIBRARIES}
)

add_test(NAME section COMMAND fall_n_section_test)

# ── Integration site ontology tests ─────────────────────────────────────────
add_executable(fall_n_integration_sites_test
  tests/test_integration_sites.cpp
)

target_include_directories(fall_n_integration_sites_test PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(fall_n_integration_sites_test PRIVATE MPI::MPI_CXX)
target_link_libraries(fall_n_integration_sites_test PRIVATE PkgConfig::PETSC)
target_link_libraries(fall_n_integration_sites_test PRIVATE Eigen3::Eigen)
target_link_libraries(fall_n_integration_sites_test PRIVATE ${VTK_LIBRARIES})

vtk_module_autoinit(
  TARGETS fall_n_integration_sites_test
  MODULES ${VTK_LIBRARIES}
)

add_test(NAME integration_sites COMMAND fall_n_integration_sites_test)

# ── FiniteElement concept + FEM_Element + ElementPolicy tests ──────────────
add_executable(fall_n_element_concept_test
  tests/test_finite_element_concept.cpp
)

target_include_directories(fall_n_element_concept_test PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(fall_n_element_concept_test PRIVATE MPI::MPI_CXX)
target_link_libraries(fall_n_element_concept_test PRIVATE PkgConfig::PETSC)
target_link_libraries(fall_n_element_concept_test PRIVATE Eigen3::Eigen)
target_link_libraries(fall_n_element_concept_test PRIVATE ${VTK_LIBRARIES})

vtk_module_autoinit(
  TARGETS fall_n_element_concept_test
  MODULES ${VTK_LIBRARIES}
)

add_test(NAME element_concept COMMAND fall_n_element_concept_test)

# ── BeamElement tests ──────────────────────────────────────────────────────
add_executable(fall_n_beam_test
  tests/test_beam_element.cpp
)

target_include_directories(fall_n_beam_test PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(fall_n_beam_test PRIVATE MPI::MPI_CXX)
target_link_libraries(fall_n_beam_test PRIVATE PkgConfig::PETSC)
target_link_libraries(fall_n_beam_test PRIVATE Eigen3::Eigen)
target_link_libraries(fall_n_beam_test PRIVATE ${VTK_LIBRARIES})

vtk_module_autoinit(
  TARGETS fall_n_beam_test
  MODULES ${VTK_LIBRARIES}
)

add_test(NAME beam_element COMMAND fall_n_beam_test)

# ── TimoshenkoBeamN tests ──────────────────────────────────────────────────
add_executable(fall_n_beam_n_test
  tests/test_timoshenko_beam_n.cpp
)

target_include_directories(fall_n_beam_n_test PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(fall_n_beam_n_test PRIVATE MPI::MPI_CXX)
target_link_libraries(fall_n_beam_n_test PRIVATE PkgConfig::PETSC)
target_link_libraries(fall_n_beam_n_test PRIVATE Eigen3::Eigen)
target_link_libraries(fall_n_beam_n_test PRIVATE ${VTK_LIBRARIES})

vtk_module_autoinit(
  TARGETS fall_n_beam_n_test
  MODULES ${VTK_LIBRARIES}
)

add_test(NAME beam_n_element COMMAND fall_n_beam_n_test)

# ── Nonlinear RC fiber-frame regression ────────────────────────────────────
add_executable(fall_n_rc_fiber_frame_test
  tests/test_rc_fiber_frame_nonlinear.cpp
)

target_include_directories(fall_n_rc_fiber_frame_test PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(fall_n_rc_fiber_frame_test PRIVATE MPI::MPI_CXX)
target_link_libraries(fall_n_rc_fiber_frame_test PRIVATE PkgConfig::PETSC)
target_link_libraries(fall_n_rc_fiber_frame_test PRIVATE Eigen3::Eigen)
target_link_libraries(fall_n_rc_fiber_frame_test PRIVATE ${VTK_LIBRARIES})

vtk_module_autoinit(
  TARGETS fall_n_rc_fiber_frame_test
  MODULES ${VTK_LIBRARIES}
)

add_test(NAME rc_fiber_frame_nonlinear COMMAND fall_n_rc_fiber_frame_test)

# ── Timoshenko cantilever benchmark (continuum vs beam vs analytical) ─────────
add_executable(fall_n_cantilever_benchmark
  tests/test_timoshenko_cantilever_benchmark.cpp
)

target_include_directories(fall_n_cantilever_benchmark PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(fall_n_cantilever_benchmark PRIVATE MPI::MPI_CXX)
target_link_libraries(fall_n_cantilever_benchmark PRIVATE PkgConfig::PETSC)
target_link_libraries(fall_n_cantilever_benchmark PRIVATE Eigen3::Eigen)
target_link_libraries(fall_n_cantilever_benchmark PRIVATE ${VTK_LIBRARIES})

vtk_module_autoinit(
  TARGETS fall_n_cantilever_benchmark
  MODULES ${VTK_LIBRARIES}
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(fall_n_cantilever_benchmark PRIVATE OpenMP::OpenMP_CXX)
endif()

add_test(NAME cantilever_benchmark COMMAND fall_n_cantilever_benchmark)

# ── Continuum tensor module tests ─────────────────────────────────────────────
add_executable(fall_n_continuum_test
  tests/test_continuum_tensors.cpp
)

target_include_directories(fall_n_continuum_test PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(fall_n_continuum_test PRIVATE MPI::MPI_CXX)
target_link_libraries(fall_n_continuum_test PRIVATE PkgConfig::PETSC)
target_link_libraries(fall_n_continuum_test PRIVATE Eigen3::Eigen)
target_link_libraries(fall_n_continuum_test PRIVATE ${VTK_LIBRARIES})

vtk_module_autoinit(
  TARGETS fall_n_continuum_test
  MODULES ${VTK_LIBRARIES}
)

add_test(NAME continuum_tensors COMMAND fall_n_continuum_test)

# ── KinematicPolicy tests (Phase 2) ──────────────────────────────────────────
add_executable(fall_n_kinematic_test
  tests/test_kinematic_policy.cpp
)

target_include_directories(fall_n_kinematic_test PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(fall_n_kinematic_test PRIVATE MPI::MPI_CXX)
target_link_libraries(fall_n_kinematic_test PRIVATE PkgConfig::PETSC)
target_link_libraries(fall_n_kinematic_test PRIVATE Eigen3::Eigen)
target_link_libraries(fall_n_kinematic_test PRIVATE ${VTK_LIBRARIES})

vtk_module_autoinit(
  TARGETS fall_n_kinematic_test
  MODULES ${VTK_LIBRARIES}
)

add_test(NAME kinematic_policy COMMAND fall_n_kinematic_test)

# ── Hyperelastic model tests (Phase 3) ───────────────────────────────────────
add_executable(fall_n_hyperelastic_test
  tests/test_hyperelastic.cpp
)

target_include_directories(fall_n_hyperelastic_test PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(fall_n_hyperelastic_test PRIVATE MPI::MPI_CXX)
target_link_libraries(fall_n_hyperelastic_test PRIVATE PkgConfig::PETSC)
target_link_libraries(fall_n_hyperelastic_test PRIVATE Eigen3::Eigen)
target_link_libraries(fall_n_hyperelastic_test PRIVATE ${VTK_LIBRARIES})

vtk_module_autoinit(
  TARGETS fall_n_hyperelastic_test
  MODULES ${VTK_LIBRARIES}
)

add_test(NAME hyperelastic COMMAND fall_n_hyperelastic_test)

# ── Continuum constitutive-interface tests ───────────────────────────────────
add_executable(fall_n_continuum_interface_test
  tests/test_continuum_constitutive_interface.cpp
)

target_include_directories(fall_n_continuum_interface_test PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(fall_n_continuum_interface_test PRIVATE MPI::MPI_CXX)
target_link_libraries(fall_n_continuum_interface_test PRIVATE PkgConfig::PETSC)
target_link_libraries(fall_n_continuum_interface_test PRIVATE Eigen3::Eigen)

add_test(NAME continuum_constitutive_interface COMMAND fall_n_continuum_interface_test)

# ── Continuum local-problem tests ────────────────────────────────────────────
add_executable(fall_n_continuum_local_problem_test
  tests/test_continuum_local_problem.cpp
)

target_include_directories(fall_n_continuum_local_problem_test PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(fall_n_continuum_local_problem_test PRIVATE MPI::MPI_CXX)
target_link_libraries(fall_n_continuum_local_problem_test PRIVATE PkgConfig::PETSC)
target_link_libraries(fall_n_continuum_local_problem_test PRIVATE Eigen3::Eigen)
target_link_libraries(fall_n_continuum_local_problem_test PRIVATE ${VTK_LIBRARIES})

vtk_module_autoinit(
  TARGETS fall_n_continuum_local_problem_test
  MODULES ${VTK_LIBRARIES}
)

add_test(NAME continuum_local_problem COMMAND fall_n_continuum_local_problem_test)

# ── Finite-strain damage relation tests ──────────────────────────────────────
add_executable(fall_n_finite_strain_damage_test
  tests/test_finite_strain_damage_relation.cpp
)

target_include_directories(fall_n_finite_strain_damage_test PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(fall_n_finite_strain_damage_test PRIVATE MPI::MPI_CXX)
target_link_libraries(fall_n_finite_strain_damage_test PRIVATE PkgConfig::PETSC)
target_link_libraries(fall_n_finite_strain_damage_test PRIVATE Eigen3::Eigen)
target_link_libraries(fall_n_finite_strain_damage_test PRIVATE ${VTK_LIBRARIES})

vtk_module_autoinit(
  TARGETS fall_n_finite_strain_damage_test
  MODULES ${VTK_LIBRARIES}
)

add_test(NAME finite_strain_damage COMMAND fall_n_finite_strain_damage_test)

# ── TotalLagrangian integration tests (Phase 4) ─────────────────────────────
add_executable(fall_n_tl_integration_test
  tests/test_tl_integration.cpp
)

target_include_directories(fall_n_tl_integration_test PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(fall_n_tl_integration_test PRIVATE MPI::MPI_CXX)
target_link_libraries(fall_n_tl_integration_test PRIVATE PkgConfig::PETSC)
target_link_libraries(fall_n_tl_integration_test PRIVATE Eigen3::Eigen)
target_link_libraries(fall_n_tl_integration_test PRIVATE ${VTK_LIBRARIES})

vtk_module_autoinit(
  TARGETS fall_n_tl_integration_test
  MODULES ${VTK_LIBRARIES}
)

add_test(NAME tl_integration COMMAND fall_n_tl_integration_test)

# ── Newton-Raphson element tests (Phase 5) ───────────────────────────────────
add_executable(fall_n_newton_test
  tests/test_newton_element.cpp
)

target_include_directories(fall_n_newton_test PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(fall_n_newton_test PRIVATE MPI::MPI_CXX)
target_link_libraries(fall_n_newton_test PRIVATE PkgConfig::PETSC)
target_link_libraries(fall_n_newton_test PRIVATE Eigen3::Eigen)
target_link_libraries(fall_n_newton_test PRIVATE ${VTK_LIBRARIES})

vtk_module_autoinit(
  TARGETS fall_n_newton_test
  MODULES ${VTK_LIBRARIES}
)

add_test(NAME newton_element COMMAND fall_n_newton_test)

# ── Updated Lagrangian tests (Phase 6) ───────────────────────────────────────
add_executable(fall_n_ul_test
  tests/test_updated_lagrangian.cpp
)

target_include_directories(fall_n_ul_test PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(fall_n_ul_test PRIVATE MPI::MPI_CXX)
target_link_libraries(fall_n_ul_test PRIVATE PkgConfig::PETSC)
target_link_libraries(fall_n_ul_test PRIVATE Eigen3::Eigen)
target_link_libraries(fall_n_ul_test PRIVATE ${VTK_LIBRARIES})

vtk_module_autoinit(
  TARGETS fall_n_ul_test
  MODULES ${VTK_LIBRARIES}
)

add_test(NAME updated_lagrangian COMMAND fall_n_ul_test)

# ── SNES + Hyperelastic end-to-end tests (Phase 7) ──────────────────────────
add_executable(fall_n_snes_test
  tests/test_snes_hyperelastic.cpp
)

target_include_directories(fall_n_snes_test PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(fall_n_snes_test PRIVATE MPI::MPI_CXX)
target_link_libraries(fall_n_snes_test PRIVATE PkgConfig::PETSC)
target_link_libraries(fall_n_snes_test PRIVATE Eigen3::Eigen)
target_link_libraries(fall_n_snes_test PRIVATE ${VTK_LIBRARIES})

vtk_module_autoinit(
  TARGETS fall_n_snes_test
  MODULES ${VTK_LIBRARIES}
)

# Optional: enable OpenMP for parallel element assembly
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  target_link_libraries(fall_n_snes_test PRIVATE OpenMP::OpenMP_CXX)
  target_link_libraries(fall_n         PUBLIC  OpenMP::OpenMP_CXX)
endif()

add_test(NAME snes_hyperelastic COMMAND fall_n_snes_test)

# ── SNES + Gmsh pipeline tests (Phase 7B: multi-element, UL, plasticity) ────
add_executable(fall_n_snes_gmsh_test
  tests/test_snes_gmsh_pipeline.cpp
)

target_include_directories(fall_n_snes_gmsh_test PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(fall_n_snes_gmsh_test PRIVATE MPI::MPI_CXX)
target_link_libraries(fall_n_snes_gmsh_test PRIVATE PkgConfig::PETSC)
target_link_libraries(fall_n_snes_gmsh_test PRIVATE Eigen3::Eigen)
target_link_libraries(fall_n_snes_gmsh_test PRIVATE ${VTK_LIBRARIES})

vtk_module_autoinit(
  TARGETS fall_n_snes_gmsh_test
  MODULES ${VTK_LIBRARIES}
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(fall_n_snes_gmsh_test PRIVATE OpenMP::OpenMP_CXX)
endif()

add_test(NAME snes_gmsh_pipeline COMMAND fall_n_snes_gmsh_test)

# ── Nonlinear Validation Suite (small mesh, 10 systematic tests) ─────────────
add_executable(fall_n_nl_validation_test
  tests/test_nonlinear_validation.cpp
)

target_include_directories(fall_n_nl_validation_test PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(fall_n_nl_validation_test PRIVATE MPI::MPI_CXX)
target_link_libraries(fall_n_nl_validation_test PRIVATE PkgConfig::PETSC)
target_link_libraries(fall_n_nl_validation_test PRIVATE Eigen3::Eigen)
target_link_libraries(fall_n_nl_validation_test PRIVATE ${VTK_LIBRARIES})

vtk_module_autoinit(
  TARGETS fall_n_nl_validation_test
  MODULES ${VTK_LIBRARIES}
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(fall_n_nl_validation_test PRIVATE OpenMP::OpenMP_CXX)
endif()

add_test(NAME nonlinear_validation COMMAND fall_n_nl_validation_test)

# ── Subentity topology compile-time tests (Phase 10) ──────────────────────
add_executable(fall_n_subentity_test
  tests/test_subentity_topology.cpp
)

target_include_directories(fall_n_subentity_test PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(fall_n_subentity_test PRIVATE MPI::MPI_CXX)
target_link_libraries(fall_n_subentity_test PRIVATE PkgConfig::PETSC)
target_link_libraries(fall_n_subentity_test PRIVATE Eigen3::Eigen)
target_link_libraries(fall_n_subentity_test PRIVATE ${VTK_LIBRARIES})

vtk_module_autoinit(
  TARGETS fall_n_subentity_test
  MODULES ${VTK_LIBRARIES}
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(fall_n_subentity_test PRIVATE OpenMP::OpenMP_CXX)
endif()

add_test(NAME subentity_topology COMMAND fall_n_subentity_test)

# ── Uniaxial Materials + Fiber Section tests (Phase 2–3) ────────────────────
add_executable(fall_n_uniaxial_fiber_test
  tests/test_uniaxial_fiber.cpp
)

target_include_directories(fall_n_uniaxial_fiber_test PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(fall_n_uniaxial_fiber_test PRIVATE MPI::MPI_CXX)
target_link_libraries(fall_n_uniaxial_fiber_test PRIVATE PkgConfig::PETSC)
target_link_libraries(fall_n_uniaxial_fiber_test PRIVATE Eigen3::Eigen)
target_link_libraries(fall_n_uniaxial_fiber_test PRIVATE ${VTK_LIBRARIES})

vtk_module_autoinit(
  TARGETS fall_n_uniaxial_fiber_test
  MODULES ${VTK_LIBRARIES}
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(fall_n_uniaxial_fiber_test PRIVATE OpenMP::OpenMP_CXX)
endif()

add_test(NAME uniaxial_fiber COMMAND fall_n_uniaxial_fiber_test)

# ── Dynamic Analysis tests (Phase 4: PETSc TS) ─────────────────────────────
add_executable(fall_n_dynamic_test
  tests/test_dynamic_analysis.cpp
)

target_include_directories(fall_n_dynamic_test PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(fall_n_dynamic_test PRIVATE MPI::MPI_CXX)
target_link_libraries(fall_n_dynamic_test PRIVATE PkgConfig::PETSC)
target_link_libraries(fall_n_dynamic_test PRIVATE Eigen3::Eigen)
target_link_libraries(fall_n_dynamic_test PRIVATE ${VTK_LIBRARIES})

vtk_module_autoinit(
  TARGETS fall_n_dynamic_test
  MODULES ${VTK_LIBRARIES}
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(fall_n_dynamic_test PRIVATE OpenMP::OpenMP_CXX)
endif()

add_test(NAME dynamic_analysis COMMAND fall_n_dynamic_test)

# ── Performance & Scalability tests (Phase 5: Benchmark + SolverConfig) ──────
add_executable(fall_n_benchmark_test
  tests/test_benchmark.cpp
)

target_include_directories(fall_n_benchmark_test PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(fall_n_benchmark_test PRIVATE MPI::MPI_CXX)
target_link_libraries(fall_n_benchmark_test PRIVATE PkgConfig::PETSC)
target_link_libraries(fall_n_benchmark_test PRIVATE Eigen3::Eigen)
target_link_libraries(fall_n_benchmark_test PRIVATE ${VTK_LIBRARIES})

vtk_module_autoinit(
  TARGETS fall_n_benchmark_test
  MODULES ${VTK_LIBRARIES}
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(fall_n_benchmark_test PRIVATE OpenMP::OpenMP_CXX)
endif()

add_test(NAME benchmark COMMAND fall_n_benchmark_test)

# ── Boundary Condition / Initial Condition tests ──────────────────────────────
add_executable(fall_n_bc_test
  tests/test_boundary_conditions.cpp
)

target_include_directories(fall_n_bc_test PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(fall_n_bc_test PRIVATE MPI::MPI_CXX)
target_link_libraries(fall_n_bc_test PRIVATE PkgConfig::PETSC)
target_link_libraries(fall_n_bc_test PRIVATE Eigen3::Eigen)
target_link_libraries(fall_n_bc_test PRIVATE ${VTK_LIBRARIES})

vtk_module_autoinit(
  TARGETS fall_n_bc_test
  MODULES ${VTK_LIBRARIES}
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(fall_n_bc_test PRIVATE OpenMP::OpenMP_CXX)
endif()

add_test(NAME boundary_conditions COMMAND fall_n_bc_test)

# ── Phase 1 verification: constitutive state protocol commit chain ───────────
add_executable(fall_n_commit_chain_test
  tests/test_constitutive_commit_chain.cpp
)

target_include_directories(fall_n_commit_chain_test PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(fall_n_commit_chain_test PRIVATE MPI::MPI_CXX)
target_link_libraries(fall_n_commit_chain_test PRIVATE PkgConfig::PETSC)
target_link_libraries(fall_n_commit_chain_test PRIVATE Eigen3::Eigen)
target_link_libraries(fall_n_commit_chain_test PRIVATE ${VTK_LIBRARIES})

vtk_module_autoinit(
  TARGETS fall_n_commit_chain_test
  MODULES ${VTK_LIBRARIES}
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(fall_n_commit_chain_test PRIVATE OpenMP::OpenMP_CXX)
endif()

add_test(NAME constitutive_commit_chain COMMAND fall_n_commit_chain_test)

# ── Phase 2 verification: element contract & heterogeneous containers ────────
add_executable(fall_n_heterogeneity_test
  tests/test_element_heterogeneity.cpp
)

target_include_directories(fall_n_heterogeneity_test PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(fall_n_heterogeneity_test PRIVATE MPI::MPI_CXX)
target_link_libraries(fall_n_heterogeneity_test PRIVATE PkgConfig::PETSC)
target_link_libraries(fall_n_heterogeneity_test PRIVATE Eigen3::Eigen)
target_link_libraries(fall_n_heterogeneity_test PRIVATE ${VTK_LIBRARIES})

vtk_module_autoinit(
  TARGETS fall_n_heterogeneity_test
  MODULES ${VTK_LIBRARIES}
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(fall_n_heterogeneity_test PRIVATE OpenMP::OpenMP_CXX)
endif()

add_test(NAME element_heterogeneity COMMAND fall_n_heterogeneity_test)

# ── Phase 3 vertical-slice gate test ──────────────────────────────────────────
add_executable(fall_n_vertical_slice_test
  tests/test_vertical_slice_gate.cpp
)

target_include_directories(fall_n_vertical_slice_test PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(fall_n_vertical_slice_test PRIVATE MPI::MPI_CXX)
target_link_libraries(fall_n_vertical_slice_test PRIVATE PkgConfig::PETSC)
target_link_libraries(fall_n_vertical_slice_test PRIVATE Eigen3::Eigen)
target_link_libraries(fall_n_vertical_slice_test PRIVATE ${VTK_LIBRARIES})

vtk_module_autoinit(
  TARGETS fall_n_vertical_slice_test
  MODULES ${VTK_LIBRARIES}
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(fall_n_vertical_slice_test PRIVATE OpenMP::OpenMP_CXX)
endif()

add_test(NAME vertical_slice_gate COMMAND fall_n_vertical_slice_test)

# --- Phase 4: Hysteretic cycling test ---
add_executable(fall_n_phase4_hysteretic_test
  tests/test_phase4_hysteretic_cycles.cpp
)

target_include_directories(fall_n_phase4_hysteretic_test PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(fall_n_phase4_hysteretic_test PRIVATE MPI::MPI_CXX)
target_link_libraries(fall_n_phase4_hysteretic_test PRIVATE PkgConfig::PETSC)
target_link_libraries(fall_n_phase4_hysteretic_test PRIVATE Eigen3::Eigen)
target_link_libraries(fall_n_phase4_hysteretic_test PRIVATE ${VTK_LIBRARIES})

vtk_module_autoinit(
  TARGETS fall_n_phase4_hysteretic_test
  MODULES ${VTK_LIBRARIES}
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(fall_n_phase4_hysteretic_test PRIVATE OpenMP::OpenMP_CXX)
endif()

add_test(NAME phase4_hysteretic_cycles COMMAND fall_n_phase4_hysteretic_test)

# --- Phase 5: Dynamic & Seismic verification test ---
add_executable(fall_n_phase5_dynamic_test
  tests/test_phase5_dynamic_verification.cpp
)

target_include_directories(fall_n_phase5_dynamic_test PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(fall_n_phase5_dynamic_test PRIVATE MPI::MPI_CXX)
target_link_libraries(fall_n_phase5_dynamic_test PRIVATE PkgConfig::PETSC)
target_link_libraries(fall_n_phase5_dynamic_test PRIVATE Eigen3::Eigen)
target_link_libraries(fall_n_phase5_dynamic_test PRIVATE ${VTK_LIBRARIES})

vtk_module_autoinit(
  TARGETS fall_n_phase5_dynamic_test
  MODULES ${VTK_LIBRARIES}
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(fall_n_phase5_dynamic_test PRIVATE OpenMP::OpenMP_CXX)
endif()

add_test(NAME phase5_dynamic_verification COMMAND fall_n_phase5_dynamic_test)

# ── Phase 6: MPI scalability audit test ─────────────────────────────────
add_executable(fall_n_phase6_mpi_test
  tests/test_phase6_mpi_scalability.cpp
)

target_include_directories(fall_n_phase6_mpi_test PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(fall_n_phase6_mpi_test PRIVATE MPI::MPI_CXX)
target_link_libraries(fall_n_phase6_mpi_test PRIVATE PkgConfig::PETSC)
target_link_libraries(fall_n_phase6_mpi_test PRIVATE Eigen3::Eigen)
target_link_libraries(fall_n_phase6_mpi_test PRIVATE ${VTK_LIBRARIES})

vtk_module_autoinit(
  TARGETS fall_n_phase6_mpi_test
  MODULES ${VTK_LIBRARIES}
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(fall_n_phase6_mpi_test PRIVATE OpenMP::OpenMP_CXX)
endif()

add_test(NAME phase6_mpi_scalability COMMAND fall_n_phase6_mpi_test)

# ===============================================================================
# ============== Matplot++  =====================================================
# ===============================================================================

#if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang") #Not workking on GNU.  
  set(FETCHCONTENT_UPDATES_DISCONNECTED_MATPLOTPLUSPLUS ON)
  FetchContent_Declare(
      matplotplusplus
      GIT_REPOSITORY https://github.com/alandefreitas/matplotplusplus
      GIT_TAG origin/master) # or whatever tag you want

  FetchContent_GetProperties(matplotplusplus)
  if(NOT matplotplusplus_POPULATED)
      FetchContent_Populate(matplotplusplus)
      add_subdirectory(${matplotplusplus_SOURCE_DIR} ${matplotplusplus_BINARY_DIR} EXCLUDE_FROM_ALL)
  endif()
#endif()

target_link_libraries(fall_n PUBLIC matplot)

# ===============================================================================

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
