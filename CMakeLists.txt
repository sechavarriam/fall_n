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
# ============== OpenMP ========================================================
# ==============================================================================

find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  target_link_libraries(fall_n PUBLIC OpenMP::OpenMP_CXX)
endif()


# ==============================================================================
# ============== Compiler cache (ccache) =======================================
# ==============================================================================

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
  message(STATUS "ccache found: ${CCACHE_PROGRAM}")
endif()


# ==============================================================================
# ============== Test helper function ==========================================
# ==============================================================================
#
#  fall_n_add_test(
#    TARGET  <executable-name>         # CMake target (keeps backward compat)
#    NAME    <ctest-name>              # CTest test name
#    SOURCE  <path/to/source.cpp>      # Source file
#    LIBS    FULL | CORE | EIGEN | VTK | NONE
#    LABEL   unit | integration | slow
#  )
#
#  Library tiers:
#    FULL  = MPI + PETSc + Eigen + VTK (+ autoinit)
#    CORE  = MPI + PETSc + Eigen
#    EIGEN = Eigen only
#    VTK   = VTK only (+ autoinit)
#    NONE  = header-only (include dir only)
#
#  OpenMP is linked automatically on all tiers (except NONE) when available.
#
function(fall_n_add_test)
    cmake_parse_arguments(ARG "" "TARGET;NAME;SOURCE;LIBS;LABEL" "" ${ARGN})

    add_executable(${ARG_TARGET} ${ARG_SOURCE})
    target_include_directories(${ARG_TARGET} PRIVATE ${CMAKE_SOURCE_DIR})

    # ── Link libraries by tier ──────────────────────────────────────────
    if(ARG_LIBS STREQUAL "FULL" OR ARG_LIBS STREQUAL "CORE")
        target_link_libraries(${ARG_TARGET} PRIVATE
            MPI::MPI_CXX  PkgConfig::PETSC  Eigen3::Eigen)
    elseif(ARG_LIBS STREQUAL "EIGEN")
        target_link_libraries(${ARG_TARGET} PRIVATE Eigen3::Eigen)
    endif()

    if(ARG_LIBS STREQUAL "FULL" OR ARG_LIBS STREQUAL "VTK")
        target_link_libraries(${ARG_TARGET} PRIVATE ${VTK_LIBRARIES})
        vtk_module_autoinit(TARGETS ${ARG_TARGET} MODULES ${VTK_LIBRARIES})
    endif()

    # OpenMP is harmless if unused; link everywhere if available
    if(OpenMP_CXX_FOUND AND NOT ARG_LIBS STREQUAL "NONE")
        target_link_libraries(${ARG_TARGET} PRIVATE OpenMP::OpenMP_CXX)
    endif()

    # ── Register CTest ──────────────────────────────────────────────────
    add_test(NAME ${ARG_NAME} COMMAND ${ARG_TARGET})
    if(ARG_LABEL)
        set_tests_properties(${ARG_NAME} PROPERTIES LABELS "${ARG_LABEL}")
    endif()
endfunction()


# ==============================================================================
# ============== Tests =========================================================
# ==============================================================================
#
#  Labels:   unit         (<3 s)   — fast feedback during development
#            integration  (3–30 s) — solver-level / MPI tests
#            slow         (>30 s)  — full nonlinear / dynamic analyses
#
#  Usage:    ctest -L unit                   # ~5 s   (35 tests)
#            ctest -L 'unit|integration'     # ~15 s  (38 tests)
#            ctest                           # ~3 min (45 tests, all)
# ==============================================================================

# ── unit tests ───────────────────────────────────────────────────────────────

fall_n_add_test(TARGET fall_n_tests
    NAME element_geometry  SOURCE tests/test_element_geometry.cpp
    LIBS FULL  LABEL unit)

fall_n_add_test(TARGET fall_n_simplex_test
    NAME simplex_elements  SOURCE tests/test_simplex.cpp
    LIBS FULL  LABEL unit)

fall_n_add_test(TARGET fall_n_vtk_tensor_fields_test
    NAME vtk_tensor_field_derivatives  SOURCE tests/test_vtk_tensor_field_derivatives.cpp
    LIBS NONE  LABEL unit)

fall_n_add_test(TARGET fall_n_vtk_gauss_cloud_test
    NAME vtk_gauss_cloud_contract  SOURCE tests/test_vtk_gauss_cloud_contract.cpp
    LIBS VTK  LABEL unit)

fall_n_add_test(TARGET fall_n_material_curves_test
    NAME material_curves_vtk  SOURCE tests/test_material_curves_vtk.cpp
    LIBS FULL  LABEL unit)

fall_n_add_test(TARGET fall_n_structural_reconstruction_test
    NAME structural_reconstruction  SOURCE tests/test_structural_reconstruction.cpp
    LIBS FULL  LABEL unit)

fall_n_add_test(TARGET fall_n_structural_vtm_test
    NAME structural_vtm_export  SOURCE tests/test_structural_vtm_export.cpp
    LIBS FULL  LABEL unit)

fall_n_add_test(TARGET fall_n_structural_mixed_vtm_test
    NAME structural_mixed_vtm_export  SOURCE tests/test_structural_mixed_vtm_export.cpp
    LIBS FULL  LABEL unit)

fall_n_add_test(TARGET fall_n_strategy_test
    NAME material_strategy  SOURCE tests/test_material_strategy.cpp
    LIBS FULL  LABEL unit)

fall_n_add_test(TARGET fall_n_plasticity_test
    NAME plasticity_composition  SOURCE tests/test_plasticity_composition.cpp
    LIBS FULL  LABEL unit)

fall_n_add_test(TARGET fall_n_point_test
    NAME point_geometry  SOURCE tests/test_point.cpp
    LIBS CORE  LABEL unit)

fall_n_add_test(TARGET fall_n_node_test
    NAME node_dof_storage  SOURCE tests/test_node_dof.cpp
    LIBS CORE  LABEL unit)

fall_n_add_test(TARGET fall_n_domain_test
    NAME domain_vertex_refactor  SOURCE tests/test_domain_vertex.cpp
    LIBS FULL  LABEL unit)

fall_n_add_test(TARGET fall_n_dmplex_mixed_test
    NAME dmplex_mixed_topology  SOURCE tests/test_dmplex_mixed_topology.cpp
    LIBS FULL  LABEL unit)

fall_n_add_test(TARGET fall_n_condensation_test
    NAME static_condensation  SOURCE tests/test_condensation.cpp
    LIBS EIGEN  LABEL unit)

fall_n_add_test(TARGET fall_n_assembly_test
    NAME assembly_policy  SOURCE tests/test_assembly_policy.cpp
    LIBS EIGEN  LABEL unit)

fall_n_add_test(TARGET fall_n_section_test
    NAME section  SOURCE tests/test_section.cpp
    LIBS FULL  LABEL unit)

fall_n_add_test(TARGET fall_n_integration_sites_test
    NAME integration_sites  SOURCE tests/test_integration_sites.cpp
    LIBS FULL  LABEL unit)

fall_n_add_test(TARGET fall_n_element_concept_test
    NAME element_concept  SOURCE tests/test_finite_element_concept.cpp
    LIBS FULL  LABEL unit)

fall_n_add_test(TARGET fall_n_beam_test
    NAME beam_element  SOURCE tests/test_beam_element.cpp
    LIBS FULL  LABEL unit)

fall_n_add_test(TARGET fall_n_beam_n_test
    NAME beam_n_element  SOURCE tests/test_timoshenko_beam_n.cpp
    LIBS FULL  LABEL unit)

fall_n_add_test(TARGET fall_n_rc_fiber_frame_test
    NAME rc_fiber_frame_nonlinear  SOURCE tests/test_rc_fiber_frame_nonlinear.cpp
    LIBS FULL  LABEL unit)

fall_n_add_test(TARGET fall_n_continuum_test
    NAME continuum_tensors  SOURCE tests/test_continuum_tensors.cpp
    LIBS FULL  LABEL unit)

fall_n_add_test(TARGET fall_n_kinematic_test
    NAME kinematic_policy  SOURCE tests/test_kinematic_policy.cpp
    LIBS FULL  LABEL unit)

fall_n_add_test(TARGET fall_n_hyperelastic_test
    NAME hyperelastic  SOURCE tests/test_hyperelastic.cpp
    LIBS FULL  LABEL unit)

fall_n_add_test(TARGET fall_n_continuum_interface_test
    NAME continuum_constitutive_interface  SOURCE tests/test_continuum_constitutive_interface.cpp
    LIBS CORE  LABEL unit)

fall_n_add_test(TARGET fall_n_continuum_local_problem_test
    NAME continuum_local_problem  SOURCE tests/test_continuum_local_problem.cpp
    LIBS FULL  LABEL unit)

fall_n_add_test(TARGET fall_n_finite_strain_damage_test
    NAME finite_strain_damage  SOURCE tests/test_finite_strain_damage_relation.cpp
    LIBS FULL  LABEL unit)

fall_n_add_test(TARGET fall_n_tl_integration_test
    NAME tl_integration  SOURCE tests/test_tl_integration.cpp
    LIBS FULL  LABEL unit)

fall_n_add_test(TARGET fall_n_newton_test
    NAME newton_element  SOURCE tests/test_newton_element.cpp
    LIBS FULL  LABEL unit)

fall_n_add_test(TARGET fall_n_ul_test
    NAME updated_lagrangian  SOURCE tests/test_updated_lagrangian.cpp
    LIBS FULL  LABEL unit)

fall_n_add_test(TARGET fall_n_subentity_test
    NAME subentity_topology  SOURCE tests/test_subentity_topology.cpp
    LIBS FULL  LABEL unit)

fall_n_add_test(TARGET fall_n_uniaxial_fiber_test
    NAME uniaxial_fiber  SOURCE tests/test_uniaxial_fiber.cpp
    LIBS FULL  LABEL unit)

fall_n_add_test(TARGET fall_n_commit_chain_test
    NAME constitutive_commit_chain  SOURCE tests/test_constitutive_commit_chain.cpp
    LIBS FULL  LABEL unit)

fall_n_add_test(TARGET fall_n_heterogeneity_test
    NAME element_heterogeneity  SOURCE tests/test_element_heterogeneity.cpp
    LIBS FULL  LABEL unit)

fall_n_add_test(TARGET fall_n_phase4_hysteretic_test
    NAME phase4_hysteretic_cycles  SOURCE tests/test_phase4_hysteretic_cycles.cpp
    LIBS FULL  LABEL unit)

# ── integration tests (3–30 s) ──────────────────────────────────────────────

fall_n_add_test(TARGET fall_n_snes_test
    NAME snes_hyperelastic  SOURCE tests/test_snes_hyperelastic.cpp
    LIBS FULL  LABEL integration)

fall_n_add_test(TARGET fall_n_benchmark_test
    NAME benchmark  SOURCE tests/test_benchmark.cpp
    LIBS FULL  LABEL integration)

fall_n_add_test(TARGET fall_n_phase6_mpi_test
    NAME phase6_mpi_scalability  SOURCE tests/test_phase6_mpi_scalability.cpp
    LIBS FULL  LABEL integration)

# ── slow tests (>30 s) ──────────────────────────────────────────────────────

fall_n_add_test(TARGET fall_n_serendipity_test
    NAME serendipity_family  SOURCE tests/test_serendipity.cpp
    LIBS FULL  LABEL slow)

fall_n_add_test(TARGET fall_n_dynamic_test
    NAME dynamic_analysis  SOURCE tests/test_dynamic_analysis.cpp
    LIBS FULL  LABEL slow)

fall_n_add_test(TARGET fall_n_nl_validation_test
    NAME nonlinear_validation  SOURCE tests/test_nonlinear_validation.cpp
    LIBS FULL  LABEL slow)

fall_n_add_test(TARGET fall_n_phase5_dynamic_test
    NAME phase5_dynamic_verification  SOURCE tests/test_phase5_dynamic_verification.cpp
    LIBS FULL  LABEL slow)

fall_n_add_test(TARGET fall_n_vertical_slice_test
    NAME vertical_slice_gate  SOURCE tests/test_vertical_slice_gate.cpp
    LIBS FULL  LABEL slow)

fall_n_add_test(TARGET fall_n_bc_test
    NAME boundary_conditions  SOURCE tests/test_boundary_conditions.cpp
    LIBS FULL  LABEL slow)

# ── Timeout guards for slow tests ───────────────────────────────────────────
set_tests_properties(
    serendipity_family  dynamic_analysis  nonlinear_validation
    phase5_dynamic_verification  vertical_slice_gate  boundary_conditions
    PROPERTIES TIMEOUT 600)


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
