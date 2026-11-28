# Toolchain para cross-compilar o transcript (main.cpp) de Linux para Windows x64.
# Os binários pesados (transcribe/ggml, sherpa-onnx, TEN VAD, ONNX Runtime) NÃO
# são compilados aqui: as import libs e DLLs já estão em Modules/transcribe/windows/.
#
# Uso:
#   cmake -S . -B build-windows -DCMAKE_TOOLCHAIN_FILE=cmake/windows-mingw.cmake
#   cmake --build build-windows -j
#
# Pré-requisito (Debian/Ubuntu):
#   sudo apt install cmake gcc-mingw-w64-x86-64 g++-mingw-w64-x86-64 binutils-mingw-w64-x86-64

set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

set(CMAKE_C_COMPILER   x86_64-w64-mingw32-gcc)
set(CMAKE_CXX_COMPILER x86_64-w64-mingw32-g++)
set(CMAKE_RC_COMPILER  x86_64-w64-mingw32-windres)

# CMake normalmente infere MINGW do compilador; explicitar evita depender da
# detecção quando a toolchain é carregada num cache já existente.
set(MINGW TRUE)

# Não re-rotear buscas: o CMakeLists resolve as libs por caminhos absolutos em
# Modules/transcribe/windows/, não pelo sysroot do MinGW.
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE NEVER)
