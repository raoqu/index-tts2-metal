cmake_minimum_required(VERSION 3.20)
project(metal_indextts2 LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 17)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Text-frontend dependencies: SentencePiece (BPE) + kaldifst (OpenFST engine
# for the wetext text-normalization grammars). These give the native runtime a
# full, faithful tokenizer matching the reference IndexTTS2 Python frontend.
# ---------------------------------------------------------------------------
include(FetchContent)
# OpenFST (vendored by kaldifst) ships a pre-3.10 cmake_minimum_required; allow it.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "" FORCE)
FetchContent_Declare(sentencepiece
  GIT_REPOSITORY https://github.com/google/sentencepiece.git
  GIT_TAG v0.2.0)
FetchContent_MakeAvailable(sentencepiece)

# kaldifst's CMake appends ${CMAKE_SOURCE_DIR}/cmake to CMAKE_MODULE_PATH and
# sets kaldifst_core's include dir to ${CMAKE_SOURCE_DIR}, both wrong under
# FetchContent. Populate manually, fix the module path, then add_subdirectory
# and repair the include dir.
set(KALDIFST_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(KALDIFST_BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(kaldifst
  GIT_REPOSITORY https://github.com/k2-fsa/kaldifst.git
  GIT_TAG v1.7.17)
FetchContent_GetProperties(kaldifst)
if(NOT kaldifst_POPULATED)
  FetchContent_Populate(kaldifst)
  list(APPEND CMAKE_MODULE_PATH ${kaldifst_SOURCE_DIR}/cmake ${kaldifst_SOURCE_DIR}/cmake/Modules)
  add_subdirectory(${kaldifst_SOURCE_DIR} ${kaldifst_BINARY_DIR})
  target_include_directories(kaldifst_core PUBLIC ${kaldifst_SOURCE_DIR})
endif()

# Embed the Metal kernel source into the binary so it has no runtime
# dependency on metal/kernels.metal.
set(METAL_EMBED_HEADER ${CMAKE_CURRENT_BINARY_DIR}/generated/metal_kernels_embedded.inc)
add_custom_command(
  OUTPUT ${METAL_EMBED_HEADER}
  COMMAND ${CMAKE_COMMAND}
          -DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/metal/kernels.metal
          -DOUTPUT=${METAL_EMBED_HEADER}
          -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_metal.cmake
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/metal/kernels.metal
          ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_metal.cmake
  COMMENT "Embedding metal/kernels.metal"
)

add_executable(mtts
  runtime/main.cpp
  runtime/impl/server.cpp
  runtime/impl/text_frontend.cpp
  runtime/audio.cpp
  runtime/bundle.cpp
  runtime/allocator.cpp
  runtime/metal_context.mm
  ${METAL_EMBED_HEADER}
)

# text_frontend.cpp pulls in third-party headers; keep the project's strict
# warning flags off for that single source so they don't trip -Wpedantic etc.
set_source_files_properties(runtime/impl/text_frontend.cpp PROPERTIES
  COMPILE_OPTIONS "-w")
set_source_files_properties(${METAL_EMBED_HEADER} PROPERTIES HEADER_FILE_ONLY ON)

set_source_files_properties(runtime/metal_context.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")

target_include_directories(mtts PRIVATE runtime/include ${CMAKE_CURRENT_BINARY_DIR}/generated
  ${kaldifst_SOURCE_DIR}
  ${openfst_SOURCE_DIR}/src/include
  ${sentencepiece_SOURCE_DIR}/src)
target_compile_options(mtts PRIVATE -Wall -Wextra -Wpedantic)
target_compile_definitions(mtts PRIVATE MIT2_TTS_LAUNCHER=1)
target_link_libraries(mtts PRIVATE
  sqlite3
  kaldifst_core
  sentencepiece-static
  "-framework Foundation"
  "-framework Metal"
  "-framework MetalPerformanceShaders"
)
