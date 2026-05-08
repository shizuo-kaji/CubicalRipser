cmake_minimum_required(VERSION 3.15)  # Updated to a more recent version for better feature support
cmake_policy(SET CMP0028 NEW)

# Project settings
project(cripser LANGUAGES CXX)
# nanobind requires C++17 or newer.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(MSVC)
    add_compile_options(/O2 /EHsc)
else()
    add_compile_options(-O3)
endif()

# nanobind requires CMake's unified `Python` package (not legacy `Python3`)
# and the SABIModule component for STABLE_ABI / Limited API builds.
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module Development.SABIModule)

# Locate nanobind installed by the active Python interpreter (declared in
# pyproject.toml's build-system.requires, so always available during builds).
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import nanobind, sys; sys.stdout.write(nanobind.cmake_dir())"
    OUTPUT_VARIABLE nanobind_DIR
    RESULT_VARIABLE _nanobind_status
)
if(NOT _nanobind_status EQUAL 0)
    message(FATAL_ERROR "Failed to locate nanobind via the active Python interpreter (${Python_EXECUTABLE}). Install it with `pip install nanobind`.")
endif()
find_package(nanobind CONFIG REQUIRED)


# Include directories
include_directories("src/")

# Create static libraries
add_library(mylib STATIC
    src/compute_pairs.cpp
    src/coboundary_enumerator.cpp
    src/joint_pairs.cpp
    src/ph_2d.cpp
)

# V-construction library
add_library(vmylib STATIC
    src/dense_cubical_grids.cpp
)

# T-construction library
add_library(tmylib STATIC
    src/dense_cubical_grids_T.cpp
)
# OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND AND NOT APPLE)
    message(STATUS "OpenMP found, enabling multithreading.")
    add_compile_options(${OpenMP_CXX_FLAGS})
    if(TARGET OpenMP::OpenMP_CXX)
        target_link_libraries(mylib PUBLIC OpenMP::OpenMP_CXX)
        target_link_libraries(vmylib PUBLIC OpenMP::OpenMP_CXX)
        target_link_libraries(tmylib PUBLIC OpenMP::OpenMP_CXX)
    endif()
else()
    set(OpenMP FALSE)
    message(STATUS "OpenMP not found, compiling sequentially.")
endif()

# Python modules using nanobind. STABLE_ABI lets a single wheel built on
# Python ≥3.12 cover all newer CPython versions (Limited API / abi3).
nanobind_add_module(_cripser STABLE_ABI src/cubicalripser_pybind.cpp)
target_link_libraries(_cripser PRIVATE mylib vmylib)

# Build the T-construction python module from the same binding source,
# overriding module name and docs via compile definitions.
nanobind_add_module(tcripser STABLE_ABI src/cubicalripser_pybind.cpp)
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
