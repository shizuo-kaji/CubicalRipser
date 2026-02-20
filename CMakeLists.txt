cmake_minimum_required(VERSION 3.15)  # Updated to a more recent version for better feature support
cmake_policy(SET CMP0028 NEW)

# Project settings
project(cripser LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_OSX_ARCHITECTURES arm64)

if(MSVC)
    add_compile_options(/O2 /EHsc)
else()
    add_compile_options(-O3)
endif()

# Ensure pybind11 uses modern CMake FindPython and the active interpreter
set(PYBIND11_FINDPYTHON ON CACHE BOOL "Use CMake's FindPython in pybind11" FORCE)

# Prefer Python3 package to avoid legacy FindPythonInterp/FindPythonLibs
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
#find_package(pybind11 CONFIG REQUIRED)


# Include directories
include_directories("src/")

# Create static libraries
add_library(mylib STATIC
    src/compute_pairs.cpp
    src/coboundary_enumerator.cpp
    src/joint_pairs.cpp
)

# V-construction library
add_library(vmylib STATIC
    src/dense_cubical_grids.cpp
)

# T-construction library
add_library(tmylib STATIC
    src/dense_cubical_grids_T.cpp
)

# Python modules using pybind11
add_subdirectory(pybind11)

pybind11_add_module(_cripser src/cubicalripser_pybind.cpp)
target_link_libraries(_cripser PRIVATE mylib vmylib)

# Build the T-construction python module from the same binding source,
# overriding module name and docs via compile definitions.
pybind11_add_module(tcripser src/cubicalripser_pybind.cpp)
target_link_libraries(tcripser PRIVATE mylib tmylib)
target_compile_definitions(tcripser PRIVATE
  CRIPSER_MODULE_NAME=tcripser
  CRIPSER_MODULE_DOC="Cubical Ripser T-construction plugin"
  CRIPSER_CURRENTMODULE="tcripser"
)

# Inject package version from setup.py if provided
if(DEFINED CRIPSER_VERSION)
  message(STATUS "Injecting version ${CRIPSER_VERSION} into pybind11 modules")
  target_compile_definitions(_cripser PRIVATE VERSION_INFO="${CRIPSER_VERSION}")
  target_compile_definitions(tcripser PRIVATE VERSION_INFO="${CRIPSER_VERSION}")
endif()

# Command-line executables
add_executable(cubicalripser src/cubicalripser.cpp)
target_link_libraries(cubicalripser PRIVATE mylib vmylib)

add_executable(tcubicalripser src/cubicalripser.cpp)
target_link_libraries(tcubicalripser PRIVATE mylib tmylib)

# Optional: Group libraries and executables in folders in IDEs
set_target_properties(mylib vmylib tmylib _cripser tcripser cubicalripser tcubicalripser PROPERTIES FOLDER "cripser")
