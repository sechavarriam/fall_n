cmake_minimum_required(VERSION 3.21.0)
project(fall_n VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(fall_n main.cpp)
include(CTest)
enable_testing()

include (FetchContent) #

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# set root of location to find PETSc's pkg-config
set(PETSC $ENV{PETSC_DIR}/$ENV{PETSC_ARCH})
set(ENV{PKG_CONFIG_PATH} ${PETSC}/lib/pkgconfig)

  #  tells CMake to build the application ex1 from the source file ex1.c
  #  this must appear AFTER the compilers are set
  
find_package(PkgConfig REQUIRED)
pkg_search_module(PETSC REQUIRED IMPORTED_TARGET PETSc)
target_link_libraries(fall_n PUBLIC PkgConfig::PETSC)
  
# https://stackoverflow.com/questions/23163075/how-to-compile-an-mpi-included-c-program-using-cmake
find_package(MPI REQUIRED)
# add this line only when you are using openmpi which has a different c++ bindings
#add_definitions(-DOMPI_SKIP_MPICXX)
target_link_libraries(fall_n PUBLIC MPI::MPI_CXX)

#if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
#  # using Clang
#elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
#  # using GCC
#elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
#  # using Intel C++
#elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
#  # using Visual Studio C++
#endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  FetchContent_Declare(
      matplotplusplus
      GIT_REPOSITORY https://github.com/alandefreitas/matplotplusplus
      GIT_TAG origin/master) # or whatever tag you want

  FetchContent_GetProperties(matplotplusplus)
  if(NOT matplotplusplus_POPULATED)
      FetchContent_Populate(matplotplusplus)
      add_subdirectory(${matplotplusplus_SOURCE_DIR} ${matplotplusplus_BINARY_DIR} EXCLUDE_FROM_ALL)
  endif()

endif()



target_link_libraries(fall_n PUBLIC matplot)



### find python libraries
##find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
##include_directories(${PYTHON3_INCLUDE_DIRS} ${NumPy_INCLUDE_DIRS})
##
##
##FetchContent_Declare(
##    matplotlib
##    GIT_REPOSITORY https://github.com/lava/matplotlib-cpp.git
##    GIT_TAG master
##)
##FetchContent_GetProperties(matplotlib)
##if(NOT matplotlib_POPULATED)
##    FetchContent_Populate(matplotlib)
##endif()
##
##include_directories(SYSTEM ${matplotlib_SOURCE_DIR})
##
### link python and numpy
##target_link_libraries(fall_n
##    PRIVATE
##        ${Python3_LIBRARIES}
##        Python3::NumPy
##)



set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
