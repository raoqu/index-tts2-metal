cmake_minimum_required(VERSION 3.20)
project(metal_indextts2 LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 17)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

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
  runtime/audio.cpp
  runtime/bundle.cpp
  runtime/allocator.cpp
  runtime/metal_context.mm
  ${METAL_EMBED_HEADER}
)
set_source_files_properties(${METAL_EMBED_HEADER} PROPERTIES HEADER_FILE_ONLY ON)

set_source_files_properties(runtime/metal_context.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")

target_include_directories(mtts PRIVATE runtime/include ${CMAKE_CURRENT_BINARY_DIR}/generated)
target_compile_options(mtts PRIVATE -Wall -Wextra -Wpedantic)
target_compile_definitions(mtts PRIVATE MIT2_TTS_LAUNCHER=1)
target_link_libraries(mtts PRIVATE
  sqlite3
  "-framework Foundation"
  "-framework Metal"
  "-framework MetalPerformanceShaders"
)
