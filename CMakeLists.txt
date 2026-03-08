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
  CommonColor
  CommonCore
  CommonDataModel
  IOXML
  FiltersGeneral
  FiltersSources
  InteractionStyle
  RenderingContextOpenGL2
  RenderingCore
  RenderingFreeType
  RenderingGL2PSOpenGL2
  RenderingOpenGL2
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

# ── Continuum tensor module tests ─────────────────────────────────────────────
add_executable(fall_n_continuum_test
  tests/test_continuum_tensors.cpp
)

target_include_directories(fall_n_continuum_test PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(fall_n_continuum_test PRIVATE Eigen3::Eigen)

add_test(NAME continuum_tensors COMMAND fall_n_continuum_test)

# ── KinematicPolicy tests (Phase 2) ──────────────────────────────────────────
add_executable(fall_n_kinematic_test
  tests/test_kinematic_policy.cpp
)

target_include_directories(fall_n_kinematic_test PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(fall_n_kinematic_test PRIVATE Eigen3::Eigen)

add_test(NAME kinematic_policy COMMAND fall_n_kinematic_test)

# ── Hyperelastic model tests (Phase 3) ───────────────────────────────────────
add_executable(fall_n_hyperelastic_test
  tests/test_hyperelastic.cpp
)

target_include_directories(fall_n_hyperelastic_test PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(fall_n_hyperelastic_test PRIVATE Eigen3::Eigen)

add_test(NAME hyperelastic COMMAND fall_n_hyperelastic_test)

# ── TotalLagrangian integration tests (Phase 4) ─────────────────────────────
add_executable(fall_n_tl_integration_test
  tests/test_tl_integration.cpp
)

target_include_directories(fall_n_tl_integration_test PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(fall_n_tl_integration_test PRIVATE Eigen3::Eigen)

add_test(NAME tl_integration COMMAND fall_n_tl_integration_test)

# ── Newton-Raphson element tests (Phase 5) ───────────────────────────────────
add_executable(fall_n_newton_test
  tests/test_newton_element.cpp
)

target_include_directories(fall_n_newton_test PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(fall_n_newton_test PRIVATE Eigen3::Eigen)

add_test(NAME newton_element COMMAND fall_n_newton_test)

# ── Updated Lagrangian tests (Phase 6) ───────────────────────────────────────
add_executable(fall_n_ul_test
  tests/test_updated_lagrangian.cpp
)

target_include_directories(fall_n_ul_test PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(fall_n_ul_test PRIVATE Eigen3::Eigen)

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

# ===============================================================================
# ============== Matplot++  =====================================================
# ===============================================================================

#if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang") #Not workking on GNU.  
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
