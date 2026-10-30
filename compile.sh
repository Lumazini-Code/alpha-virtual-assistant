#!/usr/bin/env bash
# Alpha AI — build do Manager (ava-tray + ava-web) para Linux e Windows x86_64.
#
# Uso:
#   ./compile.sh            # compila Linux + Windows
#   ./compile.sh linux      # apenas Linux (target nativo)
#   ./compile.sh windows    # apenas Windows (x86_64-pc-windows-gnu)
#   ./compile.sh --debug    # perfil debug (padrão: release)
#
# Saída (mesmo layout que install.sh/install.ps1 esperam):
#   Manager/ava-tray        (Linux, tray + API :9001)
#   Manager/ava-web         (Linux, janela nativa WebKitGTK)
#   Manager/ava-tray.exe    (Windows)
#   Manager/ava-web.exe     (Windows, WebView2)
#   Manager/WebView2Loader.dll (Windows, exigido pelo target gnu)
#
# Requisitos do build Linux (WebUI nativa):
#   sudo apt install libwebkit2gtk-4.1-dev libsoup-3.0-dev
#
# Requisitos do build Windows (cross a partir do Linux):
#   sudo apt install gcc-mingw-w64-x86-64 binutils-mingw-w64-x86-64
#   rustup target add x86_64-pc-windows-gnu
#
# O tray NÃO é compilado com a feature webapp: ele continua sem depender de
# WebKit/WebView2. Só o ava-web carrega wry/tao.

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
    # Imprime todo o cabeçalho comentado (até a primeira linha não-#).
    awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' \
        "${BASH_SOURCE[0]}"
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

check_webkit() {
    command -v pkg-config >/dev/null 2>&1 || die \
        "pkg-config não encontrado. Instale as dependências do WebKitGTK:
     sudo apt install pkg-config libwebkit2gtk-4.1-dev libsoup-3.0-dev"
    if ! pkg-config --exists webkit2gtk-4.1 gtk+-3.0 javascriptcoregtk-4.1 libsoup-3.0; then
        die "WebKitGTK dev não encontrado. Instale:
     sudo apt install libwebkit2gtk-4.1-dev libsoup-3.0-dev
   (Fedora: webkit2gtk4.1-devel libsoup3-devel)"
    fi
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
    check_webkit
    local out="$MANAGER_DIR/target/$LINUX_TARGET/$OUT_PROFILE"

    info "Compilando Linux nativo ($PROFILE) — target $LINUX_TARGET"
    # Duas invocações de propósito: o tray NÃO entra na feature webapp, então
    # não linka WebKit/WebView2 e não ganha dependências de runtime novas.
    cargo build "${CARGO_FLAGS[@]}" --target "$LINUX_TARGET" --bin ava-tray
    cargo build "${CARGO_FLAGS[@]}" --target "$LINUX_TARGET" --bin ava-web --features webapp

    [[ -f "$out/ava-tray" ]] || die "binário Linux não encontrado: $out/ava-tray"
    [[ -f "$out/ava-web" ]] || die "binário Linux não encontrado: $out/ava-web"
    cp -f "$out/ava-tray" "$MANAGER_DIR/ava-tray"
    cp -f "$out/ava-web" "$MANAGER_DIR/ava-web"
    chmod +x "$MANAGER_DIR/ava-tray" "$MANAGER_DIR/ava-web"
    ok "Linux: Manager/ava-tray ($(du -h "$MANAGER_DIR/ava-tray" | cut -f1))"
    ok "Linux: Manager/ava-web  ($(du -h "$MANAGER_DIR/ava-web" | cut -f1))"
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

    local out="$MANAGER_DIR/target/$WINDOWS_TARGET/$OUT_PROFILE"
    cargo build "${CARGO_FLAGS[@]}" --target "$WINDOWS_TARGET" --bin ava-tray
    cargo build "${CARGO_FLAGS[@]}" --target "$WINDOWS_TARGET" --bin ava-web --features webapp

    [[ -f "$out/ava-tray.exe" ]] || die "binário Windows não encontrado: $out/ava-tray.exe"
    [[ -f "$out/ava-web.exe" ]] || die "binário Windows não encontrado: $out/ava-web.exe"
    cp -f "$out/ava-tray.exe" "$MANAGER_DIR/ava-tray.exe"
    cp -f "$out/ava-web.exe" "$MANAGER_DIR/ava-web.exe"
    ok "Windows: Manager/ava-tray.exe ($(du -h "$MANAGER_DIR/ava-tray.exe" | cut -f1))"
    ok "Windows: Manager/ava-web.exe  ($(du -h "$MANAGER_DIR/ava-web.exe" | cut -f1))"

    # windows-gnu não linka o loader do WebView2 estaticamente (isso só é
    # possível no MSVC): o DLL precisa ficar ao lado do ava-web.exe.
    local dll
    dll="$(find "$out/build" -path '*/webview2-com-sys-*/out/x64/WebView2Loader.dll' -type f 2>/dev/null | head -n1)"
    if [[ -z "$dll" ]]; then
        dll="$(find "${CARGO_HOME:-$HOME/.cargo}/registry/src" \
            -path '*/webview2-com-sys-*/x64/WebView2Loader.dll' -type f 2>/dev/null | head -n1)"
    fi
    [[ -n "$dll" ]] || die "WebView2Loader.dll não encontrado (webview2-com-sys)"
    cp -f "$dll" "$MANAGER_DIR/WebView2Loader.dll"
    cp -f "$dll" "$out/WebView2Loader.dll"
    ok "Windows: Manager/WebView2Loader.dll ($(du -h "$MANAGER_DIR/WebView2Loader.dll" | cut -f1))"
}

# ── Execução ──────────────────────────────────────────────────────────────────
(( BUILD_LINUX || BUILD_WINDOWS )) || die "nada para compilar"

(( BUILD_LINUX ))   && build_linux
(( BUILD_WINDOWS )) && build_windows

info "Build concluído ($PROFILE). Artefatos:"
for artifact in ava-tray ava-web ava-tray.exe ava-web.exe WebView2Loader.dll; do
    if [[ -f "$MANAGER_DIR/$artifact" ]]; then
        printf '  %s\n' "$MANAGER_DIR/$artifact"
    fi
done
