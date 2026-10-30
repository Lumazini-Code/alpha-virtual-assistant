#!/usr/bin/env bash
# Alpha AI — build do Manager (ava-tray) para Linux nativo e Windows x86_64.
#
# Uso:
#   ./compile.sh            # compila Linux + Windows
#   ./compile.sh linux      # apenas Linux (target nativo)
#   ./compile.sh windows    # apenas Windows (x86_64-pc-windows-gnu)
#   ./compile.sh --debug    # perfil debug (padrão: release)
#
# Saída (mesmo layout que install.sh/install.ps1 esperam):
#   Manager/ava-tray        (Linux)
#   Manager/ava-tray.exe    (Windows)
#
# Requisitos do build Windows (cross a partir do Linux):
#   sudo apt install gcc-mingw-w64-x86-64 binutils-mingw-w64-x86-64
#   rustup target add x86_64-pc-windows-gnu
#
# O script valida as ferramentas, adiciona o target via rustup se preciso e
# configura CC/AR/linker do mingw para o rusqlite (bundled) e o linker do Rust.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MANAGER_DIR="$SCRIPT_DIR/Manager"
LINUX_TARGET="x86_64-unknown-linux-gnu"
WINDOWS_TARGET="x86_64-pc-windows-gnu"
MINGW_CC="x86_64-w64-mingw32-gcc"
MINGW_AR="x86_64-w64-mingw32-ar"

PROFILE="release"
CARGO_FLAGS=(--release)
BUILD_LINUX=1
BUILD_WINDOWS=1

info() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[ok]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[!]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[erro]\033[0m %s\n' "$*" >&2; exit 1; }

usage() {
    sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

# ── Argumentos ────────────────────────────────────────────────────────────────
while (($#)); do
    case "$1" in
        linux)     BUILD_LINUX=1; BUILD_WINDOWS=0 ;;
        windows)   BUILD_LINUX=0; BUILD_WINDOWS=1 ;;
        all)       BUILD_LINUX=1; BUILD_WINDOWS=1 ;;
        --debug)   PROFILE="debug"; CARGO_FLAGS=(--profile dev) ;;
        --release) PROFILE="release"; CARGO_FLAGS=(--release) ;;
        -h|--help) usage; exit 0 ;;
        *) die "argumento desconhecido: $1 (use --help)" ;;
    esac
    shift
done

# --profile dev é equivalente a debug, mas o diretório de saída é target/debug.
if [[ "$PROFILE" == "debug" ]]; then
    CARGO_FLAGS=(--profile dev)
    OUT_PROFILE="debug"
else
    OUT_PROFILE="release"
fi

[[ -f "$MANAGER_DIR/Cargo.toml" ]] || die "Cargo.toml do Manager não encontrado em $MANAGER_DIR"
command -v cargo >/dev/null 2>&1 || die "cargo não encontrado no PATH (instale o Rust: https://rustup.rs)"

cd "$MANAGER_DIR"

ensure_rust_target() {
    local target="$1"
    if rustup target list --installed 2>/dev/null | grep -qx "$target"; then
        return 0
    fi
    command -v rustup >/dev/null 2>&1 \
        || die "target $target não instalado e rustup indisponível. Rode: rustup target add $target"
    info "Instalando target Rust $target..."
    rustup target add "$target" || die "falha ao instalar o target $target"
}

check_mingw() {
    command -v "$MINGW_CC" >/dev/null 2>&1 || die \
        "$MINGW_CC não encontrado. Instale o toolchain mingw-w64:
     sudo apt install gcc-mingw-w64-x86-64 binutils-mingw-w64-x86-64
   (ou o pacote equivalente da sua distro)"
    command -v "$MINGW_AR" >/dev/null 2>&1 || die "$MINGW_AR não encontrado (binutils-mingw-w64-x86-64)"
}

# ── Linux (compilação nativa) ─────────────────────────────────────────────────
build_linux() {
    info "Compilando Linux nativo ($PROFILE) — target $LINUX_TARGET"
    cargo build "${CARGO_FLAGS[@]}" --target "$LINUX_TARGET"

    local bin="$MANAGER_DIR/target/$LINUX_TARGET/$OUT_PROFILE/ava-tray"
    [[ -f "$bin" ]] || die "binário Linux não encontrado: $bin"
    cp -f "$bin" "$MANAGER_DIR/ava-tray"
    chmod +x "$MANAGER_DIR/ava-tray"
    ok "Linux: Manager/ava-tray ($(du -h "$MANAGER_DIR/ava-tray" | cut -f1))"
}

# ── Windows (cross-compile com mingw-w64) ─────────────────────────────────────
build_windows() {
    check_mingw
    ensure_rust_target "$WINDOWS_TARGET"

    info "Compilando Windows ($PROFILE) — target $WINDOWS_TARGET"

    # cc-rs busca CC_<target> para compilar o SQLite (rusqlite/bundled).
    export CC_x86_64_pc_windows_gnu="$MINGW_CC"
    export AR_x86_64_pc_windows_gnu="$MINGW_AR"
    export CFLAGS_x86_64_pc_windows_gnu="${CFLAGS_x86_64_pc_windows_gnu:--O2}"
    # O linker do Rust para o target gnu.
    export CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER="$MINGW_CC"

    cargo build "${CARGO_FLAGS[@]}" --target "$WINDOWS_TARGET"

    local bin="$MANAGER_DIR/target/$WINDOWS_TARGET/$OUT_PROFILE/ava-tray.exe"
    [[ -f "$bin" ]] || die "binário Windows não encontrado: $bin"
    cp -f "$bin" "$MANAGER_DIR/ava-tray.exe"
    ok "Windows: Manager/ava-tray.exe ($(du -h "$MANAGER_DIR/ava-tray.exe" | cut -f1))"
}

# ── Execução ──────────────────────────────────────────────────────────────────
(( BUILD_LINUX || BUILD_WINDOWS )) || die "nada para compilar"

(( BUILD_LINUX ))   && build_linux
(( BUILD_WINDOWS )) && build_windows

info "Build concluído ($PROFILE). Artefatos:"
if [[ -f "$MANAGER_DIR/ava-tray" ]]; then
    printf '  %s\n' "$MANAGER_DIR/ava-tray"
fi
if [[ -f "$MANAGER_DIR/ava-tray.exe" ]]; then
    printf '  %s\n' "$MANAGER_DIR/ava-tray.exe"
fi
