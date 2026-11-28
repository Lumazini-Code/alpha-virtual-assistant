cmake_minimum_required(VERSION 3.21)
project(transcript LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Tudo (headers, libs, modelo) fica NESTA pasta. O executável também é gerado aqui,
# então as bibliotecas ficam ao lado dele (Linux: .so via RPATH | Windows: .dll ao
# lado do .exe — copiadas do windows/ no POST_BUILD abaixo).
set(ROOT ${CMAKE_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${ROOT})
# Geradores multi-config (Visual Studio) ignoram o valor sem sufixo de config.
foreach(_cfg DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${_cfg} ${ROOT})
endforeach()

add_executable(transcript main.cpp)

# transcribe.h, ten_vad.h na raiz; headers do ONNX Runtime em onnxruntime/;
# c-api.h da sherpa-onnx na raiz.
target_include_directories(transcript PRIVATE ${ROOT} ${ROOT}/onnxruntime)

# --- SELEÇÃO DA PLATAFORMA DE DESTINO ---
# Usamos WIN32 (e não a plataforma que executa o cmake): no cross-build com
# MinGW o cmake roda no Linux, mas WIN32 é true e as libs vêm de windows/.
if(WIN32)
    set(BUILD_PLATFORM "WINDOWS")
else()
    set(BUILD_PLATFORM "LINUX")
endif()

if(BUILD_PLATFORM STREQUAL "WINDOWS")
    set(LIBRARY_SEARCH_PATH ${ROOT}/windows)
    message(STATUS "--- Building for WINDOWS platform ---")
    # MSVC (link.exe) não linka direto contra .dll — precisa de .lib de importação.
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

# Resolve cada módulo pelo nome base. As .lib da pasta windows/ são import libs
# COFF (formato "short import", geradas por generate_import_lib.ps1); o ld do
# MinGW as aceita direto. Se houver .dll.a (dlltool) ele tem prioridade.
function(find_local_lib out_var base)
    set(_candidates)
    if(WIN32 AND MINGW)
        list(APPEND _candidates "${base}.dll.a" "lib${base}.dll.a" "${base}.lib" "lib${base}.lib")
    elseif(WIN32)
        list(APPEND _candidates "${base}.lib")
    else()
        list(APPEND _candidates "lib${base}.so" "${base}.so")
    endif()
    foreach(_name IN LISTS _candidates)
        if(EXISTS "${LIBRARY_SEARCH_PATH}/${_name}")
            set(${out_var} "${LIBRARY_SEARCH_PATH}/${_name}" PARENT_SCOPE)
            return()
        endif()
    endforeach()
    message(FATAL_ERROR
        "Biblioteca '${base}' não encontrada em ${LIBRARY_SEARCH_PATH}. "
        "Procurou: ${_candidates}")
endfunction()

find_local_lib(TRANSCRIBE_LIB  transcribe)
find_local_lib(TENVAD_LIB      ten_vad)
find_local_lib(ONNXRUNTIME_LIB onnxruntime)
find_local_lib(SHERPA_LIB      sherpa-onnx-c-api)

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

    # O Windows procura DLLs na pasta do executável. As dependências vivem em
    # windows/, mas o .exe sai na raiz do módulo ( mesma pasta que o tray
    # procura em state.rs::find_transcript_binary ).
    file(GLOB WINDOWS_DLLS "${LIBRARY_SEARCH_PATH}/*.dll")
    if(WINDOWS_DLLS)
        add_custom_command(TARGET transcript POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${WINDOWS_DLLS} $<TARGET_FILE_DIR:transcript>
            COMMENT "Copiando DLLs de windows/ para junto do transcript.exe")
    else()
        message(WARNING
            "Nenhuma .dll em ${LIBRARY_SEARCH_PATH}: o transcript.exe não vai "
            "rodar sem as dependências ao lado dele.")
    endif()
endif()

if(UNIX)
    # Achar as .so na pasta do executável (DT_RPATH vale também pras dependências das libs)
    set_target_properties(transcript PROPERTIES BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN")
    target_link_options(transcript PRIVATE -Wl,--disable-new-dtags -Wl,-rpath-link,${ROOT} -Wl,-rpath-link,${LIBRARY_SEARCH_PATH})

    # libten_vad.so foi compilado com libc++; linka o runtime/ABI ao lado dele
    # para resolver os símbolos std::__1 (o resto do exe segue em libstdc++).
    foreach(_libcxx libc++ libc++abi)
        if(EXISTS "${LIBRARY_SEARCH_PATH}/lib${_libcxx}.so.1")
            target_link_libraries(transcript PRIVATE "${LIBRARY_SEARCH_PATH}/lib${_libcxx}.so.1")
        endif()
    endforeach()
endif()
