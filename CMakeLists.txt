cmake_minimum_required(VERSION 3.16.0)
project(fall_n VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(fall_n main.cpp)
include(CTest)
enable_testing()

# =============================================================
#include(FetchContent)
#
#FetchContent_Declare(
#  Eigen3
#  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
#  GIT_TAG        master
#)
#FetchContent_MakeAvailable(Eigen3)
#
#target_link_libraries(fall_n Eigen3::Eigen)
# =============================================================

find_package(Eigen3 NO_MODULE)
if (NOT TARGET Eigen3::Eigen)


    include(FetchContent)
    FetchContent_Declare(
    Eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG master
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE)
    set(EIGEN_BUILD_DOC OFF)
    # note: To disable eigen tests,
    # you should put this code in a add_subdirectory to avoid to change
    # BUILD_TESTING for your own project too since variables are directory
    # scoped
    set(BUILD_TESTING OFF)
    set(EIGEN_BUILD_PKGCONFIG OFF)
    set( OFF)
    FetchContent_MakeAvailable(Eigen)

endif (NOT TARGET Eigen3::Eigen)

target_link_libraries(fall_n PRIVATE Eigen3::Eigen)


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
