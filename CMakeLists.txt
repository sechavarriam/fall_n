cmake_minimum_required(VERSION 3.21.0)
project(fall_n VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent) #
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

# ── Seismic executable links (same as fall_n) ────────────────────────────────
target_link_libraries(fall_n_seismic PUBLIC MPI::MPI_CXX PkgConfig::PETSC Eigen3::Eigen ${VTK_LIBRARIES})
vtk_module_autoinit(TARGETS fall_n_seismic MODULES ${VTK_LIBRARIES})
target_include_directories(fall_n_seismic PRIVATE ${CMAKE_SOURCE_DIR})

# ── Multiscale example (same library tier as fall_n_seismic) ─────────────────
target_link_libraries(fall_n_multiscale PUBLIC MPI::MPI_CXX PkgConfig::PETSC Eigen3::Eigen ${VTK_LIBRARIES})
vtk_module_autoinit(TARGETS fall_n_multiscale MODULES ${VTK_LIBRARIES})
target_include_directories(fall_n_multiscale PRIVATE ${CMAKE_SOURCE_DIR})
if(OpenMP_CXX_FOUND)
  target_link_libraries(fall_n_multiscale PUBLIC OpenMP::OpenMP_CXX)
endif()

# ── Seismic multiscale example (fiber sections + Japan 2011) ─────────────────
target_link_libraries(fall_n_multiscale_seismic PUBLIC MPI::MPI_CXX PkgConfig::PETSC Eigen3::Eigen ${VTK_LIBRARIES})
vtk_module_autoinit(TARGETS fall_n_multiscale_seismic MODULES ${VTK_LIBRARIES})
target_include_directories(fall_n_multiscale_seismic PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_definitions(fall_n_multiscale_seismic PRIVATE
    FALL_N_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
if(OpenMP_CXX_FOUND)
  target_link_libraries(fall_n_multiscale_seismic PUBLIC OpenMP::OpenMP_CXX)
endif()

# ── L-shaped building multiscale seismic (3-component, Hex20, nonlinear) ─────
add_executable(fall_n_lshaped_multiscale main_lshaped_multiscale.cpp)
target_link_libraries(fall_n_lshaped_multiscale PUBLIC MPI::MPI_CXX PkgConfig::PETSC Eigen3::Eigen ${VTK_LIBRARIES})
vtk_module_autoinit(TARGETS fall_n_lshaped_multiscale MODULES ${VTK_LIBRARIES})
target_include_directories(fall_n_lshaped_multiscale PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_definitions(fall_n_lshaped_multiscale PRIVATE
    FALL_N_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
if(OpenMP_CXX_FOUND)
  target_link_libraries(fall_n_lshaped_multiscale PUBLIC OpenMP::OpenMP_CXX)
endif()

# ── Seismic multiscale v2 (fixed DT, explicit time step) ─────────────────────
add_executable(fall_n_seismic_v2 main_multiscale_seismic.cpp)
target_link_libraries(fall_n_seismic_v2 PUBLIC MPI::MPI_CXX PkgConfig::PETSC Eigen3::Eigen ${VTK_LIBRARIES})
vtk_module_autoinit(TARGETS fall_n_seismic_v2 MODULES ${VTK_LIBRARIES})
target_include_directories(fall_n_seismic_v2 PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_definitions(fall_n_seismic_v2 PRIVATE
    FALL_N_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
if(OpenMP_CXX_FOUND)
  target_link_libraries(fall_n_seismic_v2 PUBLIC OpenMP::OpenMP_CXX)
endif()

# ── Seismic multiscale v3 (DT=0.02s, scale=5, robust SNES) ──────────────────
add_executable(fall_n_seismic_v3 main_multiscale_seismic.cpp)
target_link_libraries(fall_n_seismic_v3 PUBLIC MPI::MPI_CXX PkgConfig::PETSC Eigen3::Eigen ${VTK_LIBRARIES})
vtk_module_autoinit(TARGETS fall_n_seismic_v3 MODULES ${VTK_LIBRARIES})
target_include_directories(fall_n_seismic_v3 PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_definitions(fall_n_seismic_v3 PRIVATE
    FALL_N_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
if(OpenMP_CXX_FOUND)
  target_link_libraries(fall_n_seismic_v3 PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(fall_n_seismic_v4 main_multiscale_seismic.cpp)
target_link_libraries(fall_n_seismic_v4 PUBLIC MPI::MPI_CXX PkgConfig::PETSC Eigen3::Eigen ${VTK_LIBRARIES})
vtk_module_autoinit(TARGETS fall_n_seismic_v4 MODULES ${VTK_LIBRARIES})
target_include_directories(fall_n_seismic_v4 PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_definitions(fall_n_seismic_v4 PRIVATE
    FALL_N_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
if(OpenMP_CXX_FOUND)
  target_link_libraries(fall_n_seismic_v4 PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(fall_n_seismic_v5 main_multiscale_seismic.cpp)
target_link_libraries(fall_n_seismic_v5 PUBLIC MPI::MPI_CXX PkgConfig::PETSC Eigen3::Eigen ${VTK_LIBRARIES})
vtk_module_autoinit(TARGETS fall_n_seismic_v5 MODULES ${VTK_LIBRARIES})
target_include_directories(fall_n_seismic_v5 PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_definitions(fall_n_seismic_v5 PRIVATE
    FALL_N_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
if(OpenMP_CXX_FOUND)
  target_link_libraries(fall_n_seismic_v5 PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(fall_n_seismic_v6 main_multiscale_seismic.cpp)
target_link_libraries(fall_n_seismic_v6 PUBLIC MPI::MPI_CXX PkgConfig::PETSC Eigen3::Eigen ${VTK_LIBRARIES})
vtk_module_autoinit(TARGETS fall_n_seismic_v6 MODULES ${VTK_LIBRARIES})
target_include_directories(fall_n_seismic_v6 PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_definitions(fall_n_seismic_v6 PRIVATE
    FALL_N_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
if(OpenMP_CXX_FOUND)
  target_link_libraries(fall_n_seismic_v6 PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(fall_n_seismic_v7 main_multiscale_seismic.cpp)
target_link_libraries(fall_n_seismic_v7 PUBLIC MPI::MPI_CXX PkgConfig::PETSC Eigen3::Eigen ${VTK_LIBRARIES})
vtk_module_autoinit(TARGETS fall_n_seismic_v7 MODULES ${VTK_LIBRARIES})
target_include_directories(fall_n_seismic_v7 PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_definitions(fall_n_seismic_v7 PRIVATE
    FALL_N_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
if(OpenMP_CXX_FOUND)
  target_link_libraries(fall_n_seismic_v7 PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(fall_n_seismic_v8 main_multiscale_seismic.cpp)
target_link_libraries(fall_n_seismic_v8 PUBLIC MPI::MPI_CXX PkgConfig::PETSC Eigen3::Eigen ${VTK_LIBRARIES})
vtk_module_autoinit(TARGETS fall_n_seismic_v8 MODULES ${VTK_LIBRARIES})
target_include_directories(fall_n_seismic_v8 PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_definitions(fall_n_seismic_v8 PRIVATE
        FALL_N_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
if(OpenMP_CXX_FOUND)
    target_link_libraries(fall_n_seismic_v8 PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(fall_n_seismic_v9 main_multiscale_seismic.cpp)
target_link_libraries(fall_n_seismic_v9 PUBLIC MPI::MPI_CXX PkgConfig::PETSC Eigen3::Eigen ${VTK_LIBRARIES})
vtk_module_autoinit(TARGETS fall_n_seismic_v9 MODULES ${VTK_LIBRARIES})
target_include_directories(fall_n_seismic_v9 PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_definitions(fall_n_seismic_v9 PRIVATE
        FALL_N_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
if(OpenMP_CXX_FOUND)
    target_link_libraries(fall_n_seismic_v9 PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(fall_n_seismic_v10 main_multiscale_seismic.cpp)
target_link_libraries(fall_n_seismic_v10 PUBLIC MPI::MPI_CXX PkgConfig::PETSC Eigen3::Eigen ${VTK_LIBRARIES})
vtk_module_autoinit(TARGETS fall_n_seismic_v10 MODULES ${VTK_LIBRARIES})
target_include_directories(fall_n_seismic_v10 PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_definitions(fall_n_seismic_v10 PRIVATE
        FALL_N_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
if(OpenMP_CXX_FOUND)
    target_link_libraries(fall_n_seismic_v10 PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(fall_n_seismic_v11 main_multiscale_seismic.cpp)
target_link_libraries(fall_n_seismic_v11 PUBLIC MPI::MPI_CXX PkgConfig::PETSC Eigen3::Eigen ${VTK_LIBRARIES})
vtk_module_autoinit(TARGETS fall_n_seismic_v11 MODULES ${VTK_LIBRARIES})
target_include_directories(fall_n_seismic_v11 PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_definitions(fall_n_seismic_v11 PRIVATE
        FALL_N_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
if(OpenMP_CXX_FOUND)
    target_link_libraries(fall_n_seismic_v11 PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(fall_n_seismic_v12 main_multiscale_seismic.cpp)
target_link_libraries(fall_n_seismic_v12 PUBLIC MPI::MPI_CXX PkgConfig::PETSC Eigen3::Eigen ${VTK_LIBRARIES})
vtk_module_autoinit(TARGETS fall_n_seismic_v12 MODULES ${VTK_LIBRARIES})
target_include_directories(fall_n_seismic_v12 PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_definitions(fall_n_seismic_v12 PRIVATE
        FALL_N_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
if(OpenMP_CXX_FOUND)
    target_link_libraries(fall_n_seismic_v12 PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(fall_n_seismic_v13 main_multiscale_seismic.cpp)
target_link_libraries(fall_n_seismic_v13 PUBLIC MPI::MPI_CXX PkgConfig::PETSC Eigen3::Eigen ${VTK_LIBRARIES})
vtk_module_autoinit(TARGETS fall_n_seismic_v13 MODULES ${VTK_LIBRARIES})
target_include_directories(fall_n_seismic_v13 PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_definitions(fall_n_seismic_v13 PRIVATE
        FALL_N_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
if(OpenMP_CXX_FOUND)
    target_link_libraries(fall_n_seismic_v13 PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(fall_n_seismic_v14 main_multiscale_seismic.cpp)
target_link_libraries(fall_n_seismic_v14 PUBLIC MPI::MPI_CXX PkgConfig::PETSC Eigen3::Eigen ${VTK_LIBRARIES})
vtk_module_autoinit(TARGETS fall_n_seismic_v14 MODULES ${VTK_LIBRARIES})
target_include_directories(fall_n_seismic_v14 PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_definitions(fall_n_seismic_v14 PRIVATE
        FALL_N_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
if(OpenMP_CXX_FOUND)
    target_link_libraries(fall_n_seismic_v14 PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(fall_n_seismic_v15 main_multiscale_seismic.cpp)
target_link_libraries(fall_n_seismic_v15 PUBLIC MPI::MPI_CXX PkgConfig::PETSC Eigen3::Eigen ${VTK_LIBRARIES})
vtk_module_autoinit(TARGETS fall_n_seismic_v15 MODULES ${VTK_LIBRARIES})
target_include_directories(fall_n_seismic_v15 PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_definitions(fall_n_seismic_v15 PRIVATE
        FALL_N_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
if(OpenMP_CXX_FOUND)
    target_link_libraries(fall_n_seismic_v15 PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(fall_n_seismic_v16 main_multiscale_seismic.cpp)
target_link_libraries(fall_n_seismic_v16 PUBLIC MPI::MPI_CXX PkgConfig::PETSC Eigen3::Eigen ${VTK_LIBRARIES})
vtk_module_autoinit(TARGETS fall_n_seismic_v16 MODULES ${VTK_LIBRARIES})
target_include_directories(fall_n_seismic_v16 PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_definitions(fall_n_seismic_v16 PRIVATE
        FALL_N_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
if(OpenMP_CXX_FOUND)
    target_link_libraries(fall_n_seismic_v16 PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(fall_n_seismic_v17 main_multiscale_seismic.cpp)
target_link_libraries(fall_n_seismic_v17 PUBLIC MPI::MPI_CXX PkgConfig::PETSC Eigen3::Eigen ${VTK_LIBRARIES})
vtk_module_autoinit(TARGETS fall_n_seismic_v17 MODULES ${VTK_LIBRARIES})
target_include_directories(fall_n_seismic_v17 PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_definitions(fall_n_seismic_v17 PRIVATE
        FALL_N_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
if(OpenMP_CXX_FOUND)
    target_link_libraries(fall_n_seismic_v17 PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(fall_n_seismic_v18 main_multiscale_seismic.cpp)
target_link_libraries(fall_n_seismic_v18 PUBLIC MPI::MPI_CXX PkgConfig::PETSC Eigen3::Eigen ${VTK_LIBRARIES})
vtk_module_autoinit(TARGETS fall_n_seismic_v18 MODULES ${VTK_LIBRARIES})
target_include_directories(fall_n_seismic_v18 PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_definitions(fall_n_seismic_v18 PRIVATE
        FALL_N_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
if(OpenMP_CXX_FOUND)
    target_link_libraries(fall_n_seismic_v18 PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(fall_n_seismic_v19 main_multiscale_seismic.cpp)
target_link_libraries(fall_n_seismic_v19 PUBLIC MPI::MPI_CXX PkgConfig::PETSC Eigen3::Eigen ${VTK_LIBRARIES})
vtk_module_autoinit(TARGETS fall_n_seismic_v19 MODULES ${VTK_LIBRARIES})
target_include_directories(fall_n_seismic_v19 PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_definitions(fall_n_seismic_v19 PRIVATE
        FALL_N_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
if(OpenMP_CXX_FOUND)
    target_link_libraries(fall_n_seismic_v19 PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(fall_n_seismic_v20 main_multiscale_seismic.cpp)
target_link_libraries(fall_n_seismic_v20 PUBLIC MPI::MPI_CXX PkgConfig::PETSC Eigen3::Eigen ${VTK_LIBRARIES})
vtk_module_autoinit(TARGETS fall_n_seismic_v20 MODULES ${VTK_LIBRARIES})
target_include_directories(fall_n_seismic_v20 PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_definitions(fall_n_seismic_v20 PRIVATE
        FALL_N_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
if(OpenMP_CXX_FOUND)
    target_link_libraries(fall_n_seismic_v20 PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(fall_n_seismic_v21 main_multiscale_seismic.cpp)
target_link_libraries(fall_n_seismic_v21 PUBLIC MPI::MPI_CXX PkgConfig::PETSC Eigen3::Eigen ${VTK_LIBRARIES})
vtk_module_autoinit(TARGETS fall_n_seismic_v21 MODULES ${VTK_LIBRARIES})
target_include_directories(fall_n_seismic_v21 PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_definitions(fall_n_seismic_v21 PRIVATE
        FALL_N_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
if(OpenMP_CXX_FOUND)
    target_link_libraries(fall_n_seismic_v21 PUBLIC OpenMP::OpenMP_CXX)
endif()

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

fall_n_add_test(TARGET fall_n_truss_element_test
    NAME truss_element  SOURCE tests/test_truss_element.cpp
    LIBS FULL  LABEL unit)

fall_n_add_test(TARGET fall_n_reinforced_submodel_test
    NAME reinforced_submodel  SOURCE tests/test_reinforced_submodel.cpp
    LIBS FULL  LABEL integration)

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

fall_n_add_test(TARGET fall_n_incremental_control_test
    NAME incremental_control  SOURCE tests/test_incremental_control.cpp
    LIBS FULL  LABEL unit)

fall_n_add_test(TARGET fall_n_steppable_dynamic_test
    NAME steppable_dynamic  SOURCE tests/test_steppable_dynamic.cpp
    LIBS FULL  LABEL integration)

fall_n_add_test(TARGET fall_n_steppable_solver_test
    NAME steppable_solver  SOURCE tests/test_steppable_solver.cpp
    LIBS FULL  LABEL unit)

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
target_link_libraries(fall_n_seismic PUBLIC matplot)

# ===============================================================================

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
