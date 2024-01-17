cmake_minimum_required(VERSION 3.16.0)
project(fall_n VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(fall_n main.cpp)
include(CTest)
enable_testing()

include (FetchContent) #


# find python libraries
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
include_directories(${PYTHON3_INCLUDE_DIRS} ${NumPy_INCLUDE_DIRS})


FetchContent_Declare(
    matplotlib
    GIT_REPOSITORY https://github.com/lava/matplotlib-cpp.git
    GIT_TAG master
)
FetchContent_GetProperties(matplotlib)
if(NOT matplotlib_POPULATED)
    FetchContent_Populate(matplotlib)
endif()

include_directories(SYSTEM ${matplotlib_SOURCE_DIR})

# link python and numpy
target_link_libraries(fall_n
    PRIVATE
        ${Python3_LIBRARIES}
        Python3::NumPy
)



set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
