cmake_minimum_required(VERSION 3.21.0)
project(fall_n VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CTest)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /WX")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Werror")
endif()
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Werror -fsanitize=address ")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Werror -fsanitize=leak -fsanitize=address ")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Werror -fsanitize=leak -fsanitize=address -static-libasan ")

# On MinGW-w64 with GCC, link -lstdc++exp for C++23 <print> support
# and enable big-obj for PE/COFF (large template-heavy translation units)
if(MINGW AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  link_libraries(stdc++exp)
  add_compile_options(-Wa,-mbig-obj)

  # MinGW VTK DLLs don't export inline template member functions (like SetValue).
  # Predefine the extern-template guards so the compiler instantiates them locally.
  add_compile_definitions(
    VTK_AOS_DATA_ARRAY_TEMPLATE_EXTERN
    VTK_GDA_TEMPLATE_EXTERN
  )
endif()

# Expose source directory to C++ so files can locate data/ at runtime
add_compile_definitions(FALL_N_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

add_executable(fall_n main.cpp)
add_executable(fall_n_seismic main_seismic.cpp)
add_executable(fall_n_multiscale main_multiscale.cpp)
add_executable(fall_n_multiscale_seismic main_multiscale_seismic.cpp)
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

if(DEFINED ENV{PETSC_DIR} AND NOT "$ENV{PETSC_DIR}" STREQUAL "")
  set(PETSC $ENV{PETSC_DIR}/$ENV{PETSC_ARCH})
  set(ENV{PKG_CONFIG_PATH} ${PETSC}/lib/pkgconfig)
endif()

find_package(PkgConfig REQUIRED)
# Try canonical name first, then MSYS2 variant (double-precision, MPI, optimised)
pkg_search_module(PETSC REQUIRED IMPORTED_TARGET PETSc petsc petsc-dmo petsc-dso)
target_link_libraries(fall_n PUBLIC PkgConfig::PETSC)

# MSYS2 PETSc may lack C++ macros (built with PETSC_CLANGUAGE_C) — provide fallbacks
include(CheckSymbolExists)
set(CMAKE_REQUIRED_INCLUDES ${PETSC_INCLUDE_DIRS})
check_symbol_exists(PETSC_FUNCTION_NAME_CXX "petscconf.h" _HAVE_PETSC_FN_CXX)
if(NOT _HAVE_PETSC_FN_CXX)
  add_compile_definitions(PETSC_FUNCTION_NAME_CXX=__PRETTY_FUNCTION__)
endif()
check_symbol_exists(PETSC_CXX_RESTRICT "petscconf.h" _HAVE_PETSC_CXX_RESTRICT)
if(NOT _HAVE_PETSC_CXX_RESTRICT)
  add_compile_definitions(PETSC_CXX_RESTRICT=__restrict__)
endif()

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
  CommonExecutionModel
  IOCore
  IOXML
  IOXMLParser
  IOLegacy
  FiltersCore
  FiltersGeneral
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
  target_link_libraries(fall_n_seismic PUBLIC OpenMP::OpenMP_CXX)
endif()

# ==============================================================================
# ============== Precompiled headers ===========================================
# ==============================================================================

option(FALL_N_ENABLE_COMMON_PCH
  "Enable shared precompiled headers for heavy fall_n targets" ON)
option(FALL_N_ENABLE_HEADER_FILES_DIAGNOSTICS
  "Emit compile-time messages for legacy header_files.hh includes" OFF)

if(FALL_N_ENABLE_HEADER_FILES_DIAGNOSTICS)
  add_compile_definitions(FALL_N_ENABLE_HEADER_FILES_DIAGNOSTICS=1)
endif()

function(fall_n_enable_common_pch target_name)
  if(NOT FALL_N_ENABLE_COMMON_PCH)
    return()
  endif()

  target_precompile_headers(${target_name} PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:${CMAKE_SOURCE_DIR}/src/fall_n_common_pch.hh>")
endfunction()

function(fall_n_enable_cyclic_validation_pch target_name)
  if(NOT FALL_N_ENABLE_COMMON_PCH)
    return()
  endif()

  target_precompile_headers(${target_name} PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:${CMAKE_SOURCE_DIR}/src/validation/TableCyclicValidationPch.hh>")
endfunction()

function(fall_n_reduce_benchmark_debug_info target_name)
  if(MINGW AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(${target_name} PRIVATE
      $<$<CONFIG:Debug>:-g0>
      $<$<CONFIG:Debug>:-fno-var-tracking-assignments>)
  endif()
endfunction()

# ==============================================================================
# ============== Public Multiscale API target ==================================
# ==============================================================================

add_library(fall_n_multiscale_api INTERFACE)
add_library(fall_n::multiscale_api ALIAS fall_n_multiscale_api)

target_include_directories(fall_n_multiscale_api INTERFACE ${CMAKE_SOURCE_DIR})
target_compile_definitions(fall_n_multiscale_api INTERFACE
  FALL_N_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_link_libraries(fall_n_multiscale_api INTERFACE
  MPI::MPI_CXX
  PkgConfig::PETSC
  Eigen3::Eigen
  ${VTK_LIBRARIES})
if(OpenMP_CXX_FOUND)
  target_link_libraries(fall_n_multiscale_api INTERFACE OpenMP::OpenMP_CXX)
endif()

function(fall_n_use_multiscale_api target_name)
  target_link_libraries(${target_name} PUBLIC fall_n::multiscale_api)
  target_include_directories(${target_name} PRIVATE ${CMAKE_SOURCE_DIR})
  vtk_module_autoinit(TARGETS ${target_name} MODULES ${VTK_LIBRARIES})
  fall_n_enable_common_pch(${target_name})
endfunction()

# ── Seismic executable links (same as fall_n) ────────────────────────────────
target_link_libraries(fall_n_seismic PUBLIC MPI::MPI_CXX PkgConfig::PETSC Eigen3::Eigen ${VTK_LIBRARIES})
vtk_module_autoinit(TARGETS fall_n_seismic MODULES ${VTK_LIBRARIES})
target_include_directories(fall_n_seismic PRIVATE ${CMAKE_SOURCE_DIR})
fall_n_enable_common_pch(fall_n)
fall_n_enable_common_pch(fall_n_seismic)

# ── Multiscale example (same library tier as fall_n_seismic) ─────────────────
fall_n_use_multiscale_api(fall_n_multiscale)

# ── Seismic multiscale example (fiber sections + Japan 2011) ─────────────────
fall_n_use_multiscale_api(fall_n_multiscale_seismic)

# ── L-shaped building multiscale seismic (3-component, Hex20, nonlinear) ─────
add_executable(fall_n_lshaped_multiscale main_lshaped_multiscale.cpp)
fall_n_use_multiscale_api(fall_n_lshaped_multiscale)

# ── L-shaped 16-story building, height-irregular, multiscale FE² ──────────────
add_executable(fall_n_lshaped_multiscale_16 main_lshaped_multiscale_16storey.cpp)
fall_n_use_multiscale_api(fall_n_lshaped_multiscale_16)

# ── L-shaped RC building, bidirectional seismic, reinforced multiscale ────────
add_executable(fall_n_lshaped_rc main_lshaped_rc.cpp)
target_link_libraries(fall_n_lshaped_rc PUBLIC MPI::MPI_CXX PkgConfig::PETSC Eigen3::Eigen ${VTK_LIBRARIES})
vtk_module_autoinit(TARGETS fall_n_lshaped_rc MODULES ${VTK_LIBRARIES})
target_include_directories(fall_n_lshaped_rc PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_definitions(fall_n_lshaped_rc PRIVATE
        FALL_N_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
if(OpenMP_CXX_FOUND)
    target_link_libraries(fall_n_lshaped_rc PUBLIC OpenMP::OpenMP_CXX)
endif()
fall_n_enable_common_pch(fall_n_lshaped_rc)

# ── RC beam 3D validation (Ko-Bathe + rebar, imposed rotation) ────────────────
add_executable(fall_n_rc_beam_validation main_rc_beam_validation.cpp)
target_link_libraries(fall_n_rc_beam_validation PUBLIC MPI::MPI_CXX PkgConfig::PETSC Eigen3::Eigen ${VTK_LIBRARIES})
vtk_module_autoinit(TARGETS fall_n_rc_beam_validation MODULES ${VTK_LIBRARIES})
target_include_directories(fall_n_rc_beam_validation PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_definitions(fall_n_rc_beam_validation PRIVATE
        FALL_N_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
if(OpenMP_CXX_FOUND)
    target_link_libraries(fall_n_rc_beam_validation PUBLIC OpenMP::OpenMP_CXX)
endif()
fall_n_enable_common_pch(fall_n_rc_beam_validation)

# ── 4-legged table, MITC16 shell slab, multiscale FE² + embedded rebar ───────
add_executable(fall_n_table_multiscale main_table_multiscale.cpp)
fall_n_use_multiscale_api(fall_n_table_multiscale)

# ── Cyclic validation of FE² pipeline (progressive 5-case suite) ─────────────
add_executable(fall_n_table_cyclic_validation
    main_table_cyclic_validation.cpp
    src/validation/ReducedRCColumnStructuralBaseline.cpp
    src/validation/ReducedRCColumnSectionBaseline.cpp
    src/validation/ReducedRCColumnMomentCurvatureClosure.cpp
    src/validation/TableCyclicValidationStructural.cpp
    src/validation/TableCyclicValidationFE2Recorders.cpp
    src/validation/TableCyclicValidationFE2Setup.cpp
    src/validation/TableCyclicValidationFE2Restart.cpp
    src/validation/TableCyclicValidationFE2StepPostprocess.cpp
    src/validation/TableCyclicValidationRuntimeIO.cpp
    src/validation/TableCyclicValidationSubmodelFactories.cpp
    src/validation/TableCyclicValidationFE2.cpp
    src/validation/ComprehensiveCyclicValidation.cpp)
fall_n_use_multiscale_api(fall_n_table_cyclic_validation)
# Use a validation-specific PCH instead of the global umbrella so edits to the
# thin CLI or one implementation TU do not keep invalidating an oversized
# project-wide precompiled header.
fall_n_enable_cyclic_validation_pch(fall_n_table_cyclic_validation)
set_source_files_properties(main_table_cyclic_validation.cpp PROPERTIES
  SKIP_PRECOMPILE_HEADERS ON)
set_source_files_properties(
    src/validation/TableCyclicValidationStructural.cpp
    src/validation/TableCyclicValidationFE2Recorders.cpp
    src/validation/TableCyclicValidationFE2.cpp
    src/validation/TableCyclicValidationFE2Setup.cpp
    src/validation/TableCyclicValidationFE2Restart.cpp
    src/validation/TableCyclicValidationFE2StepPostprocess.cpp
    src/validation/ComprehensiveCyclicValidation.cpp
    PROPERTIES
    SKIP_PRECOMPILE_HEADERS ON)

# -- Reduced RC-column benchmark runner (fall_n vs external bridge)
add_executable(
    fall_n_reduced_rc_column_reference_benchmark
    main_reduced_rc_column_reference_benchmark.cpp
    src/validation/ReducedRCColumnStructuralBaseline.cpp)
fall_n_reduce_benchmark_debug_info(
    fall_n_reduced_rc_column_reference_benchmark)
target_include_directories(
    fall_n_reduced_rc_column_reference_benchmark PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(
    fall_n_reduced_rc_column_reference_benchmark PUBLIC
    MPI::MPI_CXX
    PkgConfig::PETSC
    Eigen3::Eigen
    ${VTK_LIBRARIES})
vtk_module_autoinit(
  TARGETS fall_n_reduced_rc_column_reference_benchmark
  MODULES ${VTK_LIBRARIES})
if(OpenMP_CXX_FOUND)
  target_link_libraries(
      fall_n_reduced_rc_column_reference_benchmark PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(
    fall_n_reduced_rc_column_continuum_reference_benchmark
    main_reduced_rc_column_continuum_reference_benchmark.cpp
    src/validation/ReducedRCColumnContinuumBaseline.cpp)
fall_n_reduce_benchmark_debug_info(
    fall_n_reduced_rc_column_continuum_reference_benchmark)
target_include_directories(
    fall_n_reduced_rc_column_continuum_reference_benchmark PRIVATE
    ${CMAKE_SOURCE_DIR})
target_link_libraries(
    fall_n_reduced_rc_column_continuum_reference_benchmark PUBLIC
    MPI::MPI_CXX
    PkgConfig::PETSC
    Eigen3::Eigen
    ${VTK_LIBRARIES})
vtk_module_autoinit(
  TARGETS fall_n_reduced_rc_column_continuum_reference_benchmark
  MODULES ${VTK_LIBRARIES})
if(OpenMP_CXX_FOUND)
  target_link_libraries(
      fall_n_reduced_rc_column_continuum_reference_benchmark PUBLIC
      OpenMP::OpenMP_CXX)
endif()

add_executable(
    fall_n_reduced_rc_column_section_reference_benchmark
    main_reduced_rc_column_section_reference_benchmark.cpp
    src/validation/ReducedRCColumnSectionBaseline.cpp)
fall_n_reduce_benchmark_debug_info(
    fall_n_reduced_rc_column_section_reference_benchmark)
target_include_directories(
    fall_n_reduced_rc_column_section_reference_benchmark PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(
    fall_n_reduced_rc_column_section_reference_benchmark PUBLIC
    PkgConfig::PETSC
    Eigen3::Eigen)

add_executable(
    fall_n_reduced_rc_material_reference_benchmark
    main_reduced_rc_material_reference_benchmark.cpp
    src/validation/ReducedRCColumnMaterialBaseline.cpp)
fall_n_reduce_benchmark_debug_info(
    fall_n_reduced_rc_material_reference_benchmark)
target_include_directories(
    fall_n_reduced_rc_material_reference_benchmark PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(
    fall_n_reduced_rc_material_reference_benchmark PUBLIC
    MPI::MPI_CXX
    PkgConfig::PETSC
    Eigen3::Eigen)

add_executable(
    fall_n_reduced_rc_truss_reference_benchmark
    main_reduced_rc_truss_reference_benchmark.cpp
    src/validation/ReducedRCColumnMaterialBaseline.cpp
    src/validation/ReducedRCColumnTrussBaseline.cpp)
fall_n_reduce_benchmark_debug_info(
    fall_n_reduced_rc_truss_reference_benchmark)
target_include_directories(
    fall_n_reduced_rc_truss_reference_benchmark PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(
    fall_n_reduced_rc_truss_reference_benchmark PUBLIC
    MPI::MPI_CXX
    PkgConfig::PETSC
    Eigen3::Eigen
    ${VTK_LIBRARIES})
vtk_module_autoinit(
  TARGETS fall_n_reduced_rc_truss_reference_benchmark
  MODULES ${VTK_LIBRARIES})
if(OpenMP_CXX_FOUND)
  target_link_libraries(
      fall_n_reduced_rc_truss_reference_benchmark PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(
    fall_n_reduced_rc_xfem_reference_benchmark
    main_reduced_rc_xfem_reference_benchmark.cpp)
fall_n_reduce_benchmark_debug_info(
    fall_n_reduced_rc_xfem_reference_benchmark)
target_include_directories(
    fall_n_reduced_rc_xfem_reference_benchmark PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(
    fall_n_reduced_rc_xfem_reference_benchmark PUBLIC
    MPI::MPI_CXX
    PkgConfig::PETSC
    Eigen3::Eigen
    ${VTK_LIBRARIES})
vtk_module_autoinit(
  TARGETS fall_n_reduced_rc_xfem_reference_benchmark
  MODULES ${VTK_LIBRARIES})

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

    if(ARG_LIBS STREQUAL "FULL" OR ARG_LIBS STREQUAL "CORE")
        fall_n_enable_common_pch(${ARG_TARGET})
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

fall_n_add_test(TARGET fall_n_mitc_shell_test
    NAME mitc_shell_elements  SOURCE tests/test_mitc_shell_elements.cpp
    LIBS FULL  LABEL unit)

fall_n_add_test(TARGET fall_n_seismic_infra_test
    NAME seismic_infrastructure  SOURCE tests/test_seismic_infrastructure.cpp
    LIBS FULL  LABEL unit)

fall_n_add_test(TARGET fall_n_ground_motion_parsers_test
    NAME ground_motion_parsers  SOURCE tests/test_ground_motion_parsers.cpp
    LIBS FULL  LABEL unit)

fall_n_add_test(TARGET fall_n_l_shaped_building_test
    NAME l_shaped_building  SOURCE tests/test_l_shaped_building.cpp
    LIBS FULL  LABEL unit)

fall_n_add_test(TARGET fall_n_prismatic_mesh_test
    NAME prismatic_mesh  SOURCE tests/test_prismatic_mesh.cpp
    LIBS FULL  LABEL unit)

fall_n_add_test(TARGET fall_n_linear_nl_transition_test
    NAME linear_nl_transition  SOURCE tests/test_linear_nl_transition.cpp
    LIBS FULL  LABEL unit)

fall_n_add_test(TARGET fall_n_field_transfer_test
    NAME field_transfer  SOURCE tests/test_field_transfer.cpp
    LIBS FULL  LABEL unit)

fall_n_add_test(TARGET fall_n_multiscale_coordinator_test
    NAME multiscale_coordinator  SOURCE tests/test_multiscale_coordinator.cpp
    LIBS FULL  LABEL unit)

fall_n_add_test(TARGET fall_n_multiscale_api_test
    NAME multiscale_analysis_api  SOURCE tests/test_multiscale_analysis_api.cpp
    LIBS FULL  LABEL unit)

fall_n_add_test(TARGET fall_n_affine_section_subscale_model_test
    NAME affine_section_subscale_model
    SOURCE tests/test_affine_section_subscale_model.cpp
    LIBS FULL LABEL unit)
set_target_properties(fall_n_affine_section_subscale_model_test PROPERTIES
    DISABLE_PRECOMPILE_HEADERS ON)

fall_n_add_test(TARGET fall_n_micro_solve_executor_test
    NAME micro_solve_executor  SOURCE tests/test_micro_solve_executor.cpp
    LIBS FULL  LABEL unit)

fall_n_add_test(TARGET fall_n_local_subproblem_runtime_test
    NAME local_subproblem_runtime
    SOURCE tests/test_local_subproblem_runtime.cpp
    LIBS FULL LABEL unit)

fall_n_add_test(TARGET fall_n_strategy_test
    NAME material_strategy  SOURCE tests/test_material_strategy.cpp
    LIBS FULL  LABEL unit)

fall_n_add_test(TARGET fall_n_plasticity_test
    NAME plasticity_composition  SOURCE tests/test_plasticity_composition.cpp
    LIBS FULL  LABEL unit)

fall_n_add_test(TARGET fall_n_ko_bathe_concrete_test
    NAME ko_bathe_concrete  SOURCE tests/test_ko_bathe_concrete.cpp
    LIBS FULL  LABEL unit)

fall_n_add_test(TARGET fall_n_ko_bathe_concrete_3d_test
    NAME ko_bathe_concrete_3d  SOURCE tests/test_ko_bathe_concrete_3d.cpp
    LIBS FULL  LABEL unit)

fall_n_add_test(TARGET fall_n_tensile_crack_band_damage_concrete_proxy_3d_test
    NAME tensile_crack_band_damage_concrete_proxy_3d
    SOURCE tests/test_tensile_crack_band_damage_concrete_proxy_3d.cpp
    LIBS FULL LABEL unit)

fall_n_add_test(TARGET fall_n_cyclic_crack_band_concrete_3d_test
    NAME cyclic_crack_band_concrete_3d
    SOURCE tests/test_cyclic_crack_band_concrete_3d.cpp
    LIBS FULL LABEL unit)

fall_n_add_test(TARGET fall_n_fixed_crack_band_concrete_3d_test
    NAME fixed_crack_band_concrete_3d
    SOURCE tests/test_fixed_crack_band_concrete_3d.cpp
    LIBS FULL LABEL unit)

fall_n_add_test(TARGET fall_n_crack_shear_transfer_law_test
    NAME crack_shear_transfer_law
    SOURCE tests/test_crack_shear_transfer_law.cpp
    LIBS NONE LABEL unit)

fall_n_add_test(TARGET fall_n_componentwise_kent_park_concrete_3d_test
    NAME componentwise_kent_park_concrete_3d
    SOURCE tests/test_componentwise_kent_park_concrete_3d.cpp
    LIBS FULL LABEL unit)

fall_n_add_test(TARGET fall_n_state_injection_test
    NAME state_injection  SOURCE tests/test_state_injection.cpp
    LIBS FULL  LABEL unit)

fall_n_add_test(TARGET fall_n_truss_element_test
    NAME truss_element  SOURCE tests/test_truss_element.cpp
    LIBS FULL  LABEL unit)

fall_n_add_test(TARGET fall_n_reinforced_submodel_test
    NAME reinforced_submodel  SOURCE tests/test_reinforced_submodel.cpp
    LIBS FULL  LABEL integration)

fall_n_add_test(TARGET fall_n_evolver_advanced_test
    NAME evolver_advanced  SOURCE tests/test_evolver_advanced.cpp
    LIBS FULL  LABEL integration)

fall_n_add_test(TARGET fall_n_tangent_validation_benchmark_test
    NAME tangent_validation_benchmark
    SOURCE tests/test_tangent_validation_benchmark.cpp
    LIBS FULL  LABEL integration)

fall_n_add_test(TARGET fall_n_coupling_residual_benchmark_test
    NAME coupling_residual_benchmark
    SOURCE tests/test_coupling_residual_benchmark.cpp
    LIBS FULL  LABEL integration)

fall_n_add_test(TARGET fall_n_point_test
    NAME point_geometry  SOURCE tests/test_point.cpp
    LIBS CORE  LABEL unit)

fall_n_add_test(TARGET fall_n_cyclic_protocol_test
    NAME cyclic_protocol  SOURCE tests/test_cyclic_protocol.cpp
    LIBS NONE  LABEL unit)

fall_n_add_test(TARGET fall_n_cyclic_validation_api_test
    NAME cyclic_validation_api  SOURCE tests/test_cyclic_validation_api.cpp
    LIBS FULL  LABEL unit)
target_sources(fall_n_cyclic_validation_api_test PRIVATE
    src/validation/TableCyclicValidationFE2Setup.cpp
    src/validation/TableCyclicValidationSubmodelFactories.cpp)

fall_n_add_test(TARGET fall_n_case5_frontier_probe_test
    NAME case5_frontier_probe  SOURCE tests/test_case5_frontier_probe.cpp
    LIBS FULL  LABEL slow)
set_target_properties(fall_n_case5_frontier_probe_test PROPERTIES
    DISABLE_PRECOMPILE_HEADERS ON)
target_sources(fall_n_case5_frontier_probe_test PRIVATE
    src/validation/TableCyclicValidationSupport.cpp
    src/validation/TableCyclicValidationFE2Setup.cpp
    src/validation/TableCyclicValidationSubmodelFactories.cpp)

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

fall_n_add_test(TARGET fall_n_beam_axis_quadrature_test
    NAME beam_axis_quadrature  SOURCE tests/test_beam_axis_quadrature.cpp
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
    LIBS FULL  LABEL unit)

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

fall_n_add_test(TARGET fall_n_incremental_control_test
    NAME incremental_control  SOURCE tests/test_incremental_control.cpp
    LIBS FULL  LABEL unit)

fall_n_add_test(TARGET fall_n_steppable_dynamic_test
    NAME steppable_dynamic  SOURCE tests/test_steppable_dynamic.cpp
    LIBS FULL  LABEL integration)

fall_n_add_test(TARGET fall_n_steppable_solver_test
    NAME steppable_solver  SOURCE tests/test_steppable_solver.cpp
    LIBS FULL  LABEL unit)

fall_n_add_test(TARGET fall_n_computational_variational_slice_catalog_test
    NAME computational_variational_slice_catalog
    SOURCE tests/test_computational_variational_slice_catalog.cpp
    LIBS FULL  LABEL unit)

fall_n_add_test(TARGET fall_n_computational_claim_trace_catalog_test
    NAME computational_claim_trace_catalog
    SOURCE tests/test_computational_claim_trace_catalog.cpp
    LIBS FULL  LABEL unit)

fall_n_add_test(TARGET fall_n_computational_validation_readiness_catalog_test
    NAME computational_validation_readiness_catalog
    SOURCE tests/test_computational_validation_readiness_catalog.cpp
    LIBS FULL  LABEL unit)

fall_n_add_test(TARGET fall_n_validation_campaign_catalog_test
    NAME validation_campaign_catalog
    SOURCE tests/test_validation_campaign_catalog.cpp
    LIBS NONE  LABEL unit)

fall_n_add_test(TARGET fall_n_reduced_rc_column_structural_matrix_test
    NAME reduced_rc_column_structural_matrix
    SOURCE tests/test_reduced_rc_column_structural_matrix.cpp
    LIBS FULL  LABEL integration)
set_target_properties(fall_n_reduced_rc_column_structural_matrix_test PROPERTIES
    DISABLE_PRECOMPILE_HEADERS ON)
target_sources(fall_n_reduced_rc_column_structural_matrix_test PRIVATE
    src/validation/ReducedRCColumnStructuralBaseline.cpp)

fall_n_add_test(TARGET fall_n_reduced_rc_column_section_baseline_test
    NAME reduced_rc_column_section_baseline
    SOURCE tests/test_reduced_rc_column_section_baseline.cpp
    LIBS FULL  LABEL integration)
set_target_properties(fall_n_reduced_rc_column_section_baseline_test PROPERTIES
    DISABLE_PRECOMPILE_HEADERS ON)
target_sources(fall_n_reduced_rc_column_section_baseline_test PRIVATE
    src/validation/ReducedRCColumnSectionBaseline.cpp)

fall_n_add_test(TARGET fall_n_reduced_rc_column_material_baseline_test
    NAME reduced_rc_column_material_baseline
    SOURCE tests/test_reduced_rc_column_material_baseline.cpp
    LIBS FULL  LABEL integration)
set_target_properties(fall_n_reduced_rc_column_material_baseline_test PROPERTIES
    DISABLE_PRECOMPILE_HEADERS ON)
target_sources(fall_n_reduced_rc_column_material_baseline_test PRIVATE
    src/validation/ReducedRCColumnMaterialBaseline.cpp)

fall_n_add_test(TARGET fall_n_reduced_rc_column_continuum_baseline_test
    NAME reduced_rc_column_continuum_baseline
    SOURCE tests/test_reduced_rc_column_continuum_baseline.cpp
    LIBS FULL  LABEL integration)
set_target_properties(fall_n_reduced_rc_column_continuum_baseline_test PROPERTIES
    DISABLE_PRECOMPILE_HEADERS ON)
target_sources(fall_n_reduced_rc_column_continuum_baseline_test PRIVATE
    src/validation/ReducedRCColumnContinuumBaseline.cpp)

fall_n_add_test(TARGET fall_n_reduced_rc_column_moment_curvature_closure_test
    NAME reduced_rc_column_moment_curvature_closure
    SOURCE tests/test_reduced_rc_column_moment_curvature_closure.cpp
    LIBS FULL  LABEL integration)
set_target_properties(fall_n_reduced_rc_column_moment_curvature_closure_test PROPERTIES
    DISABLE_PRECOMPILE_HEADERS ON)
target_sources(fall_n_reduced_rc_column_moment_curvature_closure_test PRIVATE
    src/validation/ReducedRCColumnStructuralBaseline.cpp
    src/validation/ReducedRCColumnSectionBaseline.cpp
    src/validation/ReducedRCColumnMomentCurvatureClosure.cpp)

fall_n_add_test(TARGET fall_n_reduced_rc_column_moment_curvature_closure_matrix_test
    NAME reduced_rc_column_moment_curvature_closure_matrix
    SOURCE tests/test_reduced_rc_column_moment_curvature_closure_matrix.cpp
    LIBS FULL  LABEL integration)
set_target_properties(fall_n_reduced_rc_column_moment_curvature_closure_matrix_test PROPERTIES
    DISABLE_PRECOMPILE_HEADERS ON)
target_sources(fall_n_reduced_rc_column_moment_curvature_closure_matrix_test PRIVATE
    src/validation/ReducedRCColumnStructuralBaseline.cpp
    src/validation/ReducedRCColumnSectionBaseline.cpp
    src/validation/ReducedRCColumnMomentCurvatureClosure.cpp
    src/validation/ReducedRCColumnMomentCurvatureClosureMatrix.cpp)

fall_n_add_test(TARGET fall_n_reduced_rc_column_node_refinement_study_test
    NAME reduced_rc_column_node_refinement_study
    SOURCE tests/test_reduced_rc_column_node_refinement_study.cpp
    LIBS FULL  LABEL integration)
set_target_properties(fall_n_reduced_rc_column_node_refinement_study_test PROPERTIES
    DISABLE_PRECOMPILE_HEADERS ON)
target_sources(fall_n_reduced_rc_column_node_refinement_study_test PRIVATE
    src/validation/ReducedRCColumnStructuralBaseline.cpp
    src/validation/ReducedRCColumnSectionBaseline.cpp
    src/validation/ReducedRCColumnMomentCurvatureClosure.cpp
    src/validation/ReducedRCColumnNodeRefinementStudy.cpp)

fall_n_add_test(TARGET fall_n_reduced_rc_column_cyclic_node_refinement_study_test
    NAME reduced_rc_column_cyclic_node_refinement_study
    SOURCE tests/test_reduced_rc_column_cyclic_node_refinement_study.cpp
    LIBS FULL  LABEL integration)
set_target_properties(fall_n_reduced_rc_column_cyclic_node_refinement_study_test PROPERTIES
    DISABLE_PRECOMPILE_HEADERS ON)
target_sources(fall_n_reduced_rc_column_cyclic_node_refinement_study_test PRIVATE
    src/validation/ReducedRCColumnStructuralBaseline.cpp
    src/validation/ReducedRCColumnCyclicNodeRefinementStudy.cpp)

fall_n_add_test(TARGET fall_n_reduced_rc_column_cyclic_continuation_sensitivity_study_test
    NAME reduced_rc_column_cyclic_continuation_sensitivity_study
    SOURCE tests/test_reduced_rc_column_cyclic_continuation_sensitivity_study.cpp
    LIBS FULL  LABEL integration)
set_target_properties(fall_n_reduced_rc_column_cyclic_continuation_sensitivity_study_test PROPERTIES
    DISABLE_PRECOMPILE_HEADERS ON)
target_sources(fall_n_reduced_rc_column_cyclic_continuation_sensitivity_study_test PRIVATE
    src/validation/ReducedRCColumnStructuralBaseline.cpp
    src/validation/ReducedRCColumnCyclicNodeRefinementStudy.cpp
    src/validation/ReducedRCColumnCyclicContinuationSensitivityStudy.cpp)

fall_n_add_test(TARGET fall_n_reduced_rc_column_quadrature_sensitivity_study_test
    NAME reduced_rc_column_quadrature_sensitivity_study
    SOURCE tests/test_reduced_rc_column_quadrature_sensitivity_study.cpp
    LIBS FULL  LABEL integration)
set_target_properties(fall_n_reduced_rc_column_quadrature_sensitivity_study_test PROPERTIES
    DISABLE_PRECOMPILE_HEADERS ON)
target_sources(fall_n_reduced_rc_column_quadrature_sensitivity_study_test PRIVATE
    src/validation/ReducedRCColumnStructuralBaseline.cpp
    src/validation/ReducedRCColumnSectionBaseline.cpp
    src/validation/ReducedRCColumnMomentCurvatureClosure.cpp
    src/validation/ReducedRCColumnQuadratureSensitivityStudy.cpp)

fall_n_add_test(TARGET fall_n_reduced_rc_column_cyclic_quadrature_sensitivity_study_test
    NAME reduced_rc_column_cyclic_quadrature_sensitivity_study
    SOURCE tests/test_reduced_rc_column_cyclic_quadrature_sensitivity_study.cpp
    LIBS FULL  LABEL integration)
set_target_properties(fall_n_reduced_rc_column_cyclic_quadrature_sensitivity_study_test PROPERTIES
    DISABLE_PRECOMPILE_HEADERS ON)
target_sources(fall_n_reduced_rc_column_cyclic_quadrature_sensitivity_study_test PRIVATE
    src/validation/ReducedRCColumnStructuralBaseline.cpp
    src/validation/ReducedRCColumnCyclicQuadratureSensitivityStudy.cpp)

fall_n_add_test(TARGET fall_n_reduced_rc_column_validation_claim_catalog_test
    NAME reduced_rc_column_validation_claim_catalog
    SOURCE tests/test_reduced_rc_column_validation_claim_catalog.cpp
    LIBS FULL  LABEL unit)
set_target_properties(fall_n_reduced_rc_column_validation_claim_catalog_test PROPERTIES
    DISABLE_PRECOMPILE_HEADERS ON)

fall_n_add_test(TARGET fall_n_reduced_rc_column_benchmark_trace_catalog_test
    NAME reduced_rc_column_benchmark_trace_catalog
    SOURCE tests/test_reduced_rc_column_benchmark_trace_catalog.cpp
    LIBS EIGEN  LABEL unit)

fall_n_add_test(TARGET fall_n_reduced_rc_column_evidence_closure_catalog_test
    NAME reduced_rc_column_evidence_closure_catalog
    SOURCE tests/test_reduced_rc_column_evidence_closure_catalog.cpp
    LIBS EIGEN  LABEL unit)

fall_n_add_test(TARGET fall_n_reduced_rc_local_model_promotion_catalog_test
    NAME reduced_rc_local_model_promotion_catalog
    SOURCE tests/test_reduced_rc_local_model_promotion_catalog.cpp
    LIBS EIGEN  LABEL unit)

fall_n_add_test(TARGET fall_n_reduced_rc_local_mesh_scale_audit_test
    NAME reduced_rc_local_mesh_scale_audit
    SOURCE tests/test_reduced_rc_local_mesh_scale_audit.cpp
    LIBS EIGEN  LABEL unit)

fall_n_add_test(TARGET fall_n_reduced_rc_petsc_matrix_storage_audit_test
    NAME reduced_rc_petsc_matrix_storage_audit
    SOURCE tests/test_reduced_rc_petsc_matrix_storage_audit.cpp
    LIBS EIGEN  LABEL unit)

fall_n_add_test(TARGET fall_n_reduced_rc_multiscale_validation_start_catalog_test
    NAME reduced_rc_multiscale_validation_start_catalog
    SOURCE tests/test_reduced_rc_multiscale_validation_start_catalog.cpp
    LIBS EIGEN  LABEL unit)

fall_n_add_test(TARGET fall_n_reduced_rc_multiscale_replay_plan_test
    NAME reduced_rc_multiscale_replay_plan
    SOURCE tests/test_reduced_rc_multiscale_replay_plan.cpp
    LIBS EIGEN  LABEL unit)

fall_n_add_test(TARGET fall_n_reduced_rc_multiscale_runtime_policy_test
    NAME reduced_rc_multiscale_runtime_policy
    SOURCE tests/test_reduced_rc_multiscale_runtime_policy.cpp
    LIBS EIGEN  LABEL unit)

fall_n_add_test(TARGET fall_n_reduced_rc_local_site_batch_plan_test
    NAME reduced_rc_local_site_batch_plan
    SOURCE tests/test_reduced_rc_local_site_batch_plan.cpp
    LIBS EIGEN  LABEL unit)

fall_n_add_test(TARGET fall_n_reduced_rc_local_site_replay_runner_test
    NAME reduced_rc_local_site_replay_runner
    SOURCE tests/test_reduced_rc_local_site_replay_runner.cpp
    LIBS EIGEN  LABEL unit)

fall_n_add_test(TARGET fall_n_reduced_rc_multiscale_readiness_gate_test
    NAME reduced_rc_multiscale_readiness_gate
    SOURCE tests/test_reduced_rc_multiscale_readiness_gate.cpp
    LIBS EIGEN  LABEL unit)

fall_n_add_test(TARGET fall_n_nonlinear_solve_policy_test
    NAME nonlinear_solve_policy
    SOURCE tests/test_nonlinear_solve_policy.cpp
    LIBS CORE  LABEL unit)

fall_n_add_test(TARGET fall_n_reduced_rc_benchmark_surface_test
    NAME reduced_rc_benchmark_surface
    SOURCE tests/test_reduced_rc_benchmark_surface.cpp
    LIBS FULL  LABEL unit)
set_target_properties(fall_n_reduced_rc_benchmark_surface_test PROPERTIES
    DISABLE_PRECOMPILE_HEADERS ON)

fall_n_add_test(TARGET fall_n_mixed_control_arc_length_test
    NAME mixed_control_arc_length
    SOURCE tests/test_mixed_control_arc_length.cpp
    LIBS FULL  LABEL unit)

fall_n_add_test(TARGET fall_n_bordered_mixed_control_newton_test
    NAME bordered_mixed_control_newton
    SOURCE tests/test_bordered_mixed_control_newton.cpp
    LIBS EIGEN LABEL unit)

fall_n_add_test(TARGET fall_n_petsc_bordered_mixed_control_newton_test
    NAME petsc_bordered_mixed_control_newton
    SOURCE tests/test_petsc_bordered_mixed_control_newton.cpp
    LIBS CORE LABEL unit)

fall_n_add_test(TARGET fall_n_petsc_nonlinear_analysis_bordered_adapter_test
    NAME petsc_nonlinear_analysis_bordered_adapter
    SOURCE tests/test_petsc_nonlinear_analysis_bordered_adapter.cpp
    LIBS FULL LABEL unit)

fall_n_add_test(TARGET fall_n_reduced_rc_column_truss_baseline_test
    NAME reduced_rc_column_truss_baseline
    SOURCE tests/test_reduced_rc_column_truss_baseline.cpp
    LIBS FULL  LABEL integration)
set_target_properties(fall_n_reduced_rc_column_truss_baseline_test PROPERTIES
    DISABLE_PRECOMPILE_HEADERS ON)
target_sources(fall_n_reduced_rc_column_truss_baseline_test PRIVATE
    src/validation/ReducedRCColumnMaterialBaseline.cpp
    src/validation/ReducedRCColumnTrussBaseline.cpp)

fall_n_add_test(TARGET fall_n_rc_section_layout_test
    NAME rc_section_layout
    SOURCE tests/test_rc_section_layout.cpp
    LIBS FULL  LABEL unit)

fall_n_add_test(TARGET fall_n_xfem_enrichment_test
    NAME xfem_enrichment
    SOURCE tests/test_xfem_enrichment.cpp
    LIBS FULL LABEL unit)

fall_n_add_test(TARGET fall_n_shifted_heaviside_kinematic_policy_test
    NAME shifted_heaviside_kinematic_policy
    SOURCE tests/test_shifted_heaviside_kinematic_policy.cpp
    LIBS EIGEN LABEL unit)

fall_n_add_test(TARGET fall_n_analysis_director_test
    NAME analysis_director  SOURCE tests/test_analysis_director.cpp
    LIBS FULL  LABEL unit)

fall_n_add_test(TARGET fall_n_phase4_hysteretic_test
    NAME phase4_hysteretic_cycles  SOURCE tests/test_phase4_hysteretic_cycles.cpp
    LIBS FULL  LABEL unit)

fall_n_add_test(TARGET fall_n_sub_model_solver_test
    NAME sub_model_solver  SOURCE tests/test_sub_model_solver.cpp
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
    case5_frontier_probe
    PROPERTIES TIMEOUT 600)

# ── Long-running validation_reboot studies (Phase 3 sweeps) ──────────────────
# These are intentionally exhaustive (36-case structural matrix × cyclic +
# monotonic protocols × quadrature/node sweeps). They legitimately need >180 s
# headroom and are skipped from the default `unit` lane.
set_tests_properties(
    reduced_rc_column_moment_curvature_closure_matrix
    reduced_rc_column_node_refinement_study
    reduced_rc_column_cyclic_node_refinement_study
    reduced_rc_column_cyclic_continuation_sensitivity_study
    reduced_rc_column_quadrature_sensitivity_study
    reduced_rc_column_cyclic_quadrature_sensitivity_study
    PROPERTIES TIMEOUT 1800)

# ── Plan-v2 phase / workstream labels (additive, do not displace `unit`/`integration`/`slow`) ──
# These reflect the canonical workstream table in
# `src/validation/ValidationCampaignCatalog.hh` and the matrix axis frozen in
# `src/validation/ReducedRCColumnStructuralMatrixCatalog.hh`. They allow:
#   ctest -L validation_reboot          → full reboot suite
#   ctest -L phase3_structural_matrix   → 36-case 𝔅_phase3 sweep + matrix
#   ctest -L phaseN                     → tests gating workstream phase N

# validation_reboot : the whole reboot harness (catalogs + baselines + closures + studies)
set(_FN_VALIDATION_REBOOT_TESTS
    validation_campaign_catalog
    reduced_rc_column_structural_matrix
    reduced_rc_column_section_baseline
    reduced_rc_column_material_baseline
    reduced_rc_column_continuum_baseline
    reduced_rc_column_moment_curvature_closure
    reduced_rc_column_moment_curvature_closure_matrix
    reduced_rc_column_node_refinement_study
    reduced_rc_column_cyclic_node_refinement_study
    reduced_rc_column_cyclic_continuation_sensitivity_study
    reduced_rc_column_quadrature_sensitivity_study
    reduced_rc_column_cyclic_quadrature_sensitivity_study
    reduced_rc_column_validation_claim_catalog
    reduced_rc_column_benchmark_trace_catalog
    reduced_rc_column_evidence_closure_catalog
    reduced_rc_local_model_promotion_catalog
    reduced_rc_local_mesh_scale_audit
    reduced_rc_petsc_matrix_storage_audit
    reduced_rc_multiscale_validation_start_catalog
    reduced_rc_multiscale_replay_plan
    reduced_rc_multiscale_runtime_policy
    reduced_rc_local_site_batch_plan
    reduced_rc_local_site_replay_runner
    reduced_rc_multiscale_readiness_gate
    reduced_rc_benchmark_surface
    reduced_rc_column_truss_baseline)
foreach(_t IN LISTS _FN_VALIDATION_REBOOT_TESTS)
    set_property(TEST ${_t} APPEND PROPERTY LABELS validation_reboot)
endforeach()

# phase3_structural_matrix : 36-case 𝔅_phase3 + sweep studies
set(_FN_PHASE3_MATRIX_TESTS
    reduced_rc_column_structural_matrix
    reduced_rc_column_moment_curvature_closure_matrix
    reduced_rc_column_node_refinement_study
    reduced_rc_column_cyclic_node_refinement_study
    reduced_rc_column_cyclic_continuation_sensitivity_study
    reduced_rc_column_quadrature_sensitivity_study
    reduced_rc_column_cyclic_quadrature_sensitivity_study)
foreach(_t IN LISTS _FN_PHASE3_MATRIX_TESTS)
    set_property(TEST ${_t} APPEND PROPERTY LABELS phase3_structural_matrix)
endforeach()

# Per-phase labels (canonical workstream gating, plan v2 §Fase 0)
# Phase semantics follow `ValidationRebootPhaseKind` in
# `src/validation/ValidationCampaignCatalog.hh`:
#   phase0 = governance_and_legacy_quarantine
#   phase1 = formulation_and_solver_audit
#   phase2 = material_and_section_baseline
#   phase3 = reduced_order_rc_column (matrix + sweeps)
#   phase4 = continuum_rc_column
#   phase5 = cross_model_equivalence
#   phase6 = full_structure_escalation
set_property(TEST validation_campaign_catalog           APPEND PROPERTY LABELS phase0)
set_property(TEST reduced_rc_column_material_baseline   APPEND PROPERTY LABELS phase2)
set_property(TEST reduced_rc_column_section_baseline    APPEND PROPERTY LABELS phase2)
foreach(_t IN LISTS _FN_PHASE3_MATRIX_TESTS)
    set_property(TEST ${_t} APPEND PROPERTY LABELS phase3)
endforeach()
set_property(TEST reduced_rc_column_continuum_baseline  APPEND PROPERTY LABELS phase4)
set_property(TEST reduced_rc_column_truss_baseline      APPEND PROPERTY LABELS phase4)
set_property(TEST phase4_hysteretic_cycles              APPEND PROPERTY LABELS phase4)
set_property(TEST reduced_rc_column_moment_curvature_closure APPEND PROPERTY LABELS phase5)
set_property(TEST phase5_dynamic_verification           APPEND PROPERTY LABELS phase5)
set_property(TEST phase6_mpi_scalability                APPEND PROPERTY LABELS phase6)


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
