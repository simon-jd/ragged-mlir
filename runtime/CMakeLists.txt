cmake_minimum_required(VERSION 3.25)
project(RaggedRuntime LANGUAGES CXX)

add_library(ragged_driver STATIC Driver.cpp)
target_compile_features(ragged_driver PUBLIC cxx_std_17)
set_target_properties(ragged_driver PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(ragged_driver PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ragged_driver PUBLIC ${CMAKE_DL_LIBS})

option(RAGGED_BUILD_PYTHON "Build optional nanobind CUDA Driver loader" ON)
if(RAGGED_BUILD_PYTHON)
  find_package(Python 3.12 COMPONENTS Interpreter Development.Module REQUIRED)
  execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
  find_package(nanobind CONFIG REQUIRED)
  nanobind_add_module(_native bindings.cpp)
  target_link_libraries(_native PRIVATE ragged_driver)
  install(TARGETS _native LIBRARY DESTINATION ragged)
endif()
