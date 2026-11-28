#!/usr/bin/env bash
# Build do transcript (main.cpp) para Linux e Windows x64.
#
# O motor de ASR NÃO é compilado aqui: transcribe/ggml, sherpa-onnx, TEN VAD e
# ONNX Runtime já vêm como binários pré-compilados em linux/ e windows/.
# Este script só compila o servidor/host e linka contra esses binários.
#
# Uso (de qualquer diretório):
#   ./Modules/transcribe/build.sh            # Linux + Windows
#   ./Modules/transcribe/build.sh linux      # só Linux (nativo)
#   ./Modules/transcribe/build.sh windows    # só Windows (cross-compile MinGW)
#
# Saída:
#   Modules/transcribe/transcript      (ELF x86-64, com as .so em linux/)
#   Modules/transcribe/transcript.exe  (PE x86-64, com as .dll copiadas pra raiz)
#
# Pré-requisitos:
#   Linux:   sudo apt install cmake
#   Windows: sudo apt install cmake gcc-mingw-w64-x86-64 g++-mingw-w64-x86-64 \
#                              binutils-mingw-w64-x86-64

set -euo pipefail

MODULE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TOOLCHAIN="$MODULE_DIR/cmake/windows-mingw.cmake"
BUILD_LINUX=1
BUILD_WINDOWS=1

info() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[ok]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[!]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[erro]\033[0m %s\n' "$*" >&2; exit 1; }

usage() {
    awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "${BASH_SOURCE[0]}"
}

while (($#)); do
    case "$1" in
        linux)   BUILD_LINUX=1; BUILD_WINDOWS=0 ;;
        windows) BUILD_LINUX=0; BUILD_WINDOWS=1 ;;
        all)     BUILD_LINUX=1; BUILD_WINDOWS=1 ;;
        -h|--help) usage; exit 0 ;;
        *) die "argumento desconhecido: $1 (use --help)" ;;
    esac
    shift
done

command -v cmake >/dev/null 2>&1 || die "cmake não encontrado no PATH."

build_linux() {
    info "Compilando transcript para Linux (nativo)…"
    cmake -S "$MODULE_DIR" -B "$MODULE_DIR/build-linux" -DCMAKE_BUILD_TYPE=Release
    cmake --build "$MODULE_DIR/build-linux" --parallel
    local exe="$MODULE_DIR/transcript"
    [[ -f "$exe" ]] || die "binário Linux não encontrado: $exe"
    ok "Linux: $exe ($(du -h "$exe" | cut -f1))"
}

build_windows() {
    local cc=x86_64-w64-mingw32-gcc cxx=x86_64-w64-mingw32-g++
    command -v "$cc"  >/dev/null 2>&1 || die "$cc não encontrado. Instale gcc-mingw-w64-x86-64."
    command -v "$cxx" >/dev/null 2>&1 || die "$cxx não encontrado. Instale g++-mingw-w64-x86-64."
    [[ -f "$TOOLCHAIN" ]] || die "toolchain não encontrada: $TOOLCHAIN"

    info "Compilando transcript para Windows x64 (cross-compile MinGW)…"
    cmake -S "$MODULE_DIR" -B "$MODULE_DIR/build-windows" \
        -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" -DCMAKE_BUILD_TYPE=Release
    cmake --build "$MODULE_DIR/build-windows" --parallel
    local exe="$MODULE_DIR/transcript.exe"
    [[ -f "$exe" ]] || die "binário Windows não encontrado: $exe"
    ok "Windows: $exe ($(du -h "$exe" | cut -f1))"

    local n
    n="$(find "$MODULE_DIR" -maxdepth 1 -name '*.dll' | wc -l)"
    if [[ "$n" -eq 0 ]]; then
        warn "Nenhuma .dll ao lado do transcript.exe — copie windows/*.dll manualmente."
    else
        ok "Windows: $n DLLs ao lado do executável."
    fi
}

(( BUILD_LINUX || BUILD_WINDOWS )) || die "nada para compilar"
(( BUILD_LINUX ))   && build_linux
(( BUILD_WINDOWS )) && build_windows
info "Build concluído."
