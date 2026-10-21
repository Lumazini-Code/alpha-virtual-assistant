cmake_minimum_required(VERSION 3.21)
project(transcript LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Tudo (headers, libs, modelo) fica NESTA pasta. O executável também é gerado aqui,
# então as bibliotecas ficam ao lado dele (Linux: .so via RPATH | Windows: .dll ao
# lado do .exe — copie as DLLs pra cá depois do build).
set(ROOT ${CMAKE_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${ROOT})

add_executable(transcript main.cpp)

# transcribe.h, ten_vad.h na raiz; headers do ONNX Runtime em onnxruntime/;
# headers da sherpa-onnx em sherpa-onnx/c-api/c-api.h (a partir da raiz)
target_include_directories(transcript PRIVATE ${ROOT} ${ROOT}/onnxruntime)

# --- PLATFORM SELECTION AND LIBRARY SEARCH SETUP ---
option(BUILD_PLATFORM "Set build platform (LINUX/WINDOWS)" "LINUX")

if(BUILD_PLATFORM STREQUAL "WINDOWS")
    set(LIBRARY_SEARCH_PATH ${ROOT}/windows)
    message(STATUS "--- Building for WINDOWS platform ---")
    # MSVC (link.exe) não linka direto contra .dll — precisa de .lib de importação
    set(EXPECTED_LIBS transcribe.lib ten_vad.lib onnxruntime.lib sherpa-onnx-c-api.lib)
else()
    set(LIBRARY_SEARCH_PATH ${ROOT}/linux)
    message(STATUS "--- Building for LINUX platform ---")
    set(EXPECTED_LIBS libtranscribe.so libten_vad.so libonnxruntime.so libsherpa-onnx-c-api.so)
endif()

# Diagnóstico: avisa quais bibliotecas esperadas não estão na pasta da plataforma
foreach(LIB_FILE ${EXPECTED_LIBS})
    if(NOT EXISTS ${LIBRARY_SEARCH_PATH}/${LIB_FILE})
        message(WARNING "Biblioteca não encontrada em ${LIBRARY_SEARCH_PATH}: ${LIB_FILE}")
    endif()
endforeach()

# Bibliotecas locais do módulo (dependendo da plataforma)
find_library(TRANSCRIBE_LIB  transcribe            PATHS ${LIBRARY_SEARCH_PATH} NO_DEFAULT_PATH REQUIRED)
find_library(TENVAD_LIB      ten_vad               PATHS ${LIBRARY_SEARCH_PATH} NO_DEFAULT_PATH REQUIRED)
find_library(ONNXRUNTIME_LIB onnxruntime           PATHS ${LIBRARY_SEARCH_PATH} NO_DEFAULT_PATH REQUIRED)
find_library(SHERPA_LIB      sherpa-onnx-c-api     PATHS ${LIBRARY_SEARCH_PATH} NO_DEFAULT_PATH REQUIRED)

# miniaudio (header-only, sem linkagem — a implementation é compilada no main.cpp)
# httplib (header-only, sem linkagem — o cliente HTTP é instanciado direto no main.cpp)

target_link_libraries(transcript PRIVATE
    ${TRANSCRIBE_LIB} ${TENVAD_LIB} ${SHERPA_LIB} ${ONNXRUNTIME_LIB})

if(WIN32)
    # Winsock (embed server), fallback de M_PI p/ MSVC e silencia warnings de CRT.
    # NOMINMAX: impede o windows.h de definir as macros min/max, que quebram
    # std::min / std::max / std::numeric_limits<T>::max().
    # WIN32_LEAN_AND_MEAN: evita conflito entre winsock.h e winsock2.h.
    target_link_libraries(transcript PRIVATE ws2_32)
    target_compile_definitions(transcript PRIVATE
        _CRT_SECURE_NO_WARNINGS
        _USE_MATH_DEFINES
        NOMINMAX
        WIN32_LEAN_AND_MEAN)
endif()

if(UNIX)
    # Achar as .so na pasta do executável (DT_RPATH vale também pras dependências das libs)
    set_target_properties(transcript PROPERTIES BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN")
    target_link_options(transcript PRIVATE -Wl,--disable-new-dtags -Wl,-rpath-link,${ROOT})
endif()