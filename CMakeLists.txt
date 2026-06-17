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
#
# The sources are vendored under third_party/ (sentencepiece v0.2.0,
# kaldifst v1.7.17, openfst sherpa-onnx-2024-06-13 already patched by kaldifst)
# so the build needs NO network access. See third_party/README.md.
# ---------------------------------------------------------------------------
# OpenFST ships a pre-3.10 cmake_minimum_required; allow it for the subbuilds.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
set(MIT2_TP ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

# --- SentencePiece (static lib, bundled abseil, no network) ---
set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "" FORCE)
set(SPM_ABSL_PROVIDER "internal" CACHE STRING "" FORCE)
add_subdirectory(${MIT2_TP}/sentencepiece ${CMAKE_BINARY_DIR}/sentencepiece-build EXCLUDE_FROM_ALL)

# --- OpenFST (disable every extension; only the core `fst` lib is needed) ---
foreach(_fst_opt HAVE_BIN HAVE_COMPACT HAVE_COMPRESS HAVE_CONST HAVE_FAR HAVE_GRM
                 HAVE_PDT HAVE_MPDT HAVE_LINEAR HAVE_LOOKAHEAD HAVE_NGRAM
                 HAVE_PYTHON HAVE_SPECIAL)
  set(${_fst_opt} OFF CACHE BOOL "" FORCE)
endforeach()
set(HAVE_SCRIPT ON CACHE BOOL "" FORCE)
add_subdirectory(${MIT2_TP}/openfst ${CMAKE_BINARY_DIR}/openfst-build EXCLUDE_FROM_ALL)

# --- kaldifst_core (text normalizer): built directly from the vendored sources,
#     bypassing kaldifst's own CMake (which downloads openfst via FetchContent). ---
set(KALDIFST_CSRC ${MIT2_TP}/kaldifst/kaldifst/csrc)
add_library(kaldifst_core STATIC
  ${KALDIFST_CSRC}/context-fst.cc
  ${KALDIFST_CSRC}/kaldi-fst-io.cc
  ${KALDIFST_CSRC}/kaldi-holder.cc
  ${KALDIFST_CSRC}/kaldi-io.cc
  ${KALDIFST_CSRC}/kaldi-math.cc
  ${KALDIFST_CSRC}/kaldi-semaphore.cc
  ${KALDIFST_CSRC}/kaldi-table.cc
  ${KALDIFST_CSRC}/parse-options.cc
  ${KALDIFST_CSRC}/text-normalizer.cc
  ${KALDIFST_CSRC}/text-utils.cc)
target_include_directories(kaldifst_core PUBLIC ${MIT2_TP}/kaldifst ${MIT2_TP}/openfst/src/include)
target_link_libraries(kaldifst_core PUBLIC fst)
target_compile_options(kaldifst_core PRIVATE -w)

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
  ${MIT2_TP}/kaldifst
  ${MIT2_TP}/openfst/src/include
  ${MIT2_TP}/sentencepiece/src)
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
