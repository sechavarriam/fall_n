cmake_minimum_required(VERSION 3.21.0)
project(fall_n VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(fall_n main.cpp)
include(CTest)
enable_testing()

include (FetchContent) #


FetchContent_Declare(
    matplotplusplus
    GIT_REPOSITORY https://github.com/alandefreitas/matplotplusplus
    GIT_TAG origin/master) # or whatever tag you want

FetchContent_GetProperties(matplotplusplus)
if(NOT matplotplusplus_POPULATED)
    FetchContent_Populate(matplotplusplus)
    add_subdirectory(${matplotplusplus_SOURCE_DIR} ${matplotplusplus_BINARY_DIR} EXCLUDE_FROM_ALL)
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
