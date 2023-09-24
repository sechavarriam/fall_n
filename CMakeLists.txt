cmake_minimum_required(VERSION 3.20.0)
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


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
