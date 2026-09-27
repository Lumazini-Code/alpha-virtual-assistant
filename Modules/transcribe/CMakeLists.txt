cmake_minimum_required(VERSION 3.21)
project(transcript LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Tudo (headers, .so, modelo) fica NESTA pasta. O executável também é gerado aqui,
# então as .so ficam ao lado dele e nenhuma variável de ambiente é necessária.
set(ROOT ${CMAKE_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${ROOT})

add_executable(transcript main.cpp)

# transcribe.h, transcribe/, ten_vad.h na raiz; headers do ONNX Runtime em onnxruntime/
target_include_directories(transcript PRIVATE ${ROOT} ${ROOT}/onnxruntime)

find_library(TRANSCRIBE_LIB  transcribe  PATHS ${ROOT} NO_DEFAULT_PATH REQUIRED)
find_library(TENVAD_LIB      ten_vad     PATHS ${ROOT} NO_DEFAULT_PATH REQUIRED)
find_library(ONNXRUNTIME_LIB onnxruntime PATHS ${ROOT} NO_DEFAULT_PATH REQUIRED)

find_package(PkgConfig REQUIRED)                              # sudo apt install portaudio19-dev pkg-config
pkg_check_modules(PORTAUDIO REQUIRED IMPORTED_TARGET portaudio-2.0)

target_link_libraries(transcript PRIVATE
    ${TRANSCRIBE_LIB} ${TENVAD_LIB} ${ONNXRUNTIME_LIB} PkgConfig::PORTAUDIO)

# Achar as .so na pasta do executável (DT_RPATH vale também pras dependências das libs)
set_target_properties(transcript PROPERTIES BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN")
target_link_options(transcript PRIVATE -Wl,--disable-new-dtags -Wl,-rpath-link,${ROOT})