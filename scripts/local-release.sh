#!/usr/bin/env bash
# Release local da Alpha AI: build, pacote, tag, upload e instalação de teste.
#
# Uso:
#   ./scripts/local-release.sh --version v5.1 [--yes] [--skip-build] [--skip-install]
#   ./scripts/local-release.sh --version v5.1 --local-only [--yes] [--skip-build]
#
# Modo --local-only: compila, monta os zips e reinstala a instalação local.
# Não exige working tree limpo, não cria tag, não publica no GitHub e não faz
# commits — útil para testar mudanças ainda não commitadas.
#
# No modo completo, o working tree PRECISA estar limpo: a tag aponta para o
# HEAD, que é o mesmo código usado para montar os zips. O script avisa e aborta
# se houver mudanças não commitadas.
#
# Saída: dist/alpha-ai-linux-<VER>.zip, dist/alpha-ai-windows-<VER>.zip,
#        dist/RELEASE_NOTES.md, dist/SHA256SUMS
# Depois (modo completo): cria/push da tag Alpha-AI-<VER>, release no GitHub via gh.
# Por fim (salvo --skip-install): reinstala localmente com ALPHA_RELEASE_ZIP.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

VERSION="v5.1"
YES=0
SKIP_BUILD=0
SKIP_INSTALL=0
LOCAL_ONLY=0

info() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[ok]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[!]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[erro]\033[0m %s\n' "$*" >&2; exit 1; }

usage() {
    awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "${BASH_SOURCE[0]}"
}

while (($#)); do
    case "$1" in
        --version)      VERSION="${2:?--version requer um valor}"; shift 2 ;;
        --yes|-y)       YES=1; shift ;;
        --skip-build)   SKIP_BUILD=1; shift ;;
        --skip-install) SKIP_INSTALL=1; shift ;;
        --local-only)   LOCAL_ONLY=1; shift ;;
        -h|--help)      usage; exit 0 ;;
        *) die "argumento desconhecido: $1 (use --help)" ;;
    esac
done

TAG="Alpha-AI-${VERSION}"
TITLE="Alpha AI ${VERSION}"
LINUX_ZIP="alpha-ai-linux-${VERSION}.zip"
WINDOWS_ZIP="alpha-ai-windows-${VERSION}.zip"
DIST="$ROOT/dist"
TMP_ROOT="$(mktemp -d -t alpha-release-XXXXXX)"
trap 'rm -rf "$TMP_ROOT"' EXIT

# ── Preflight ─────────────────────────────────────────────────────────────────
for tool in git cmake zip unzip cargo rustup python3 \
            x86_64-w64-mingw32-gcc x86_64-w64-mingw32-g++; do
    command -v "$tool" >/dev/null 2>&1 || die "'$tool' não encontrado no PATH."
done

if (( LOCAL_ONLY )); then
    info "Modo local: build + instalação local, sem tag/GitHub/commits."
else
    command -v gh >/dev/null 2>&1 || die "'gh' não encontrado no PATH."
    gh auth status >/dev/null 2>&1 || die "gh não autenticado (rode 'gh auth login')."

    if [[ -n "$(git status --porcelain)" ]]; then
        warn "Working tree sujo — as mudanças precisam ser commitadas antes do release."
        git status --short >&2
        die "Commite e rode novamente. A tag precisa apontar para o mesmo código dos zips."
    fi

    if git rev-parse -q --verify "refs/tags/$TAG" >/dev/null; then
        die "Tag $TAG já existe no git."
    fi
fi

PREV_TAG="$(git describe --tags --abbrev=0 HEAD 2>/dev/null || true)"
[[ -n "$PREV_TAG" ]] || PREV_TAG="$(git rev-list --max-parents=0 HEAD | head -n1)"
if (( LOCAL_ONLY )); then
    info "Versão local $VERSION (working tree atual)."
else
    info "Versão $VERSION — tag $TAG — commits desde $PREV_TAG"
fi

if (( ! YES )); then
    if (( LOCAL_ONLY )); then
        printf 'Compilar e reinstalar localmente a versão %s? [s/N] ' "$TITLE"
    else
        printf 'Criar release %s, publicar no GitHub e reinstalar localmente? [s/N] ' "$TITLE"
    fi
    read -r ans || ans=""
    case "$ans" in
        s*|S*) ;;
        *) info "Cancelado."; exit 0 ;;
    esac
fi

# ── Build ─────────────────────────────────────────────────────────────────────
if (( ! SKIP_BUILD )); then
    info "Build do Manager (Linux + Windows)…"
    ./compile.sh
    info "Build do transcript (Linux + Windows)…"
    ./Modules/transcribe/build.sh
else
    warn "Build pulado (--skip-build) — usando binários existentes."
fi

[[ -x Manager/ava-tray ]]      || die "Manager/ava-tray ausente."
[[ -x Manager/ava-web ]]       || die "Manager/ava-web ausente."
[[ -f Manager/ava-tray.exe ]]  || die "Manager/ava-tray.exe ausente."
[[ -f Manager/ava-web.exe ]]   || die "Manager/ava-web.exe ausente."
[[ -f Manager/WebView2Loader.dll ]] || die "Manager/WebView2Loader.dll ausente."
[[ -x Modules/transcribe/transcript ]]    || die "Modules/transcribe/transcript ausente."
[[ -f Modules/transcribe/transcript.exe ]] || die "Modules/transcribe/transcript.exe ausente."
[[ -f Modules/transcribe/linux/libtranscribe.so ]] || die "libs Linux do transcript ausentes."
[[ -f Modules/transcribe/windows/transcribe.dll ]] || die "DLLs do transcript ausentes."
ok "Binários verificados."

mkdir -p "$DIST"

# ── Notas de release (commits da versão) ──────────────────────────────────────
if (( LOCAL_ONLY )); then
    printf '# %s\n\nBuild local (sem tag e sem publicação no GitHub).\n' "$TITLE" > "$DIST/RELEASE_NOTES.md"
else
    {
        printf '# %s\n\n' "$TITLE"
        printf 'Commits desde %s:\n\n' "$PREV_TAG"
        git log --no-merges --pretty=format:'- %h %s' "${PREV_TAG}..HEAD"
        printf '\n'
    } > "$DIST/RELEASE_NOTES.md"
fi
ok "Notas geradas: $DIST/RELEASE_NOTES.md"

# ── Models.zip (somente os modelos em uso) ────────────────────────────────────
MODELS=( 
    ms-marco-MiniLM-L-6-v2
    multilingual-e5-small
    sherpa-onnx
    3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx
    nemotron-3.5-asr-streaming-0.6b-Q4_K_M.gguf
)
for m in "${MODELS[@]}"; do
    [[ -e "Modules/Models/$m" ]] || die "Modelo ausente: Modules/Models/$m"
done
info "Gerando Modules/Models.zip (apenas os 5 modelos em uso)…"
(
    cd Modules/Models
    zip -qr "$TMP_ROOT/Models.zip" "${MODELS[@]}"
)
expected="$(printf '%s\n' "${MODELS[@]}" | cut -d/ -f1 | sort)"
actual="$(unzip -Z1 "$TMP_ROOT/Models.zip" | cut -d/ -f1 | sort -u)"
[[ "$actual" == "$expected" ]] || die "Models.zip contém entradas inesperadas:\n$actual"
ok "Models.zip: $(du -h "$TMP_ROOT/Models.zip" | cut -f1)"

# ── Stage (working tree, respeitando .gitignore) ──────────────────────────────
stage_tree() {
    local dest="$1"
    mkdir -p "$dest"
    rsync -a --files-from=<(git ls-files -co --exclude-standard) ./ "$dest/"
    mkdir -p "$dest/Modules"
    cp -a "$TMP_ROOT/Models.zip" "$dest/Modules/Models.zip"
}

STAGE_L="$TMP_ROOT/linux"
STAGE_W="$TMP_ROOT/windows"
stage_tree "$STAGE_L"
stage_tree "$STAGE_W"
ok "Stage montado (working tree)."

# ── Overlay dos binários ignorados pelo git ───────────────────────────────────
cp -a Manager/ava-tray Manager/ava-web "$STAGE_L/Manager/"
mkdir -p "$STAGE_L/Modules/transcribe"
cp -a Modules/transcribe/transcript "$STAGE_L/Modules/transcribe/"
# O RPATH do exe é $ORIGIN: todas as .so (incluindo libc++/libc++abi/libunwind,
# exigidas por libten_vad.so) precisam ficar ao lado dele, não em linux/.
cp -a Modules/transcribe/linux/. "$STAGE_L/Modules/transcribe/"

cp -a Manager/ava-tray.exe Manager/ava-web.exe Manager/WebView2Loader.dll "$STAGE_W/Manager/"
mkdir -p "$STAGE_W/Modules/transcribe"
cp -a Modules/transcribe/transcript.exe "$STAGE_W/Modules/transcribe/"
cp -a Modules/transcribe/windows/*.dll "$STAGE_W/Modules/transcribe/"

# ── Poda por plataforma ───────────────────────────────────────────────────────
# Linux: nada de Windows.
find "$STAGE_L" -type f \( -name '*.exe' -o -name '*.dll' \) -delete
rm -rf "$STAGE_L/Modules/transcribe/windows"
rm -f "$STAGE_L/install.ps1" "$STAGE_L/install.bat" "$STAGE_L/docker-start.bat"

# Windows: nada de Linux (inclui os ELFs sem extensão).
find "$STAGE_W" -type f -name '*.so' -delete
rm -rf "$STAGE_W/Modules/transcribe/linux"
rm -f "$STAGE_W/Modules/transcribe/transcript"
rm -f "$STAGE_W/Manager/ava-tray" "$STAGE_W/Manager/ava-web"
rm -f "$STAGE_W/install.sh" "$STAGE_W/docker-start.sh" "$STAGE_W/start.sh"
ok "Poda por plataforma concluída."

# ── Zips ──────────────────────────────────────────────────────────────────────
info "Compactando $LINUX_ZIP …"
(cd "$STAGE_L" && zip -qr "$DIST/$LINUX_ZIP" .)
info "Compactando $WINDOWS_ZIP …"
(cd "$STAGE_W" && zip -qr "$DIST/$WINDOWS_ZIP" .)
ok "Zips: $(du -h "$DIST/$LINUX_ZIP" | cut -f1) | $(du -h "$DIST/$WINDOWS_ZIP" | cut -f1)"

# ── Validação dos zips ────────────────────────────────────────────────────────
validate_zip() {
    local zip="$1" platform="$2"
    local list
    list="$(unzip -Z1 "$zip" | sed 's#^\./##')"
    local required
    if [[ "$platform" == linux ]]; then
        required=(install.sh 'Manager/ava-tray' 'Manager/ava-web' 'Modules/Models.zip'
                  'Modules/transcribe/transcript' 'Modules/transcribe/libtranscribe.so')
    else
        required=(install.ps1 'Manager/ava-tray.exe' 'Manager/ava-web.exe'
                  'Manager/WebView2Loader.dll' 'Modules/Models.zip'
                  'Modules/transcribe/transcript.exe' 'Modules/transcribe/transcribe.dll')
    fi
    local f
    for f in "${required[@]}"; do
        grep -qx -- "$f" <<<"$list" || die "$zip: falta '$f'"
    done
    if grep -qE '(^|/)\.env$|node_modules/|\.db$|Modules/Models/(depth_anything|dinov3|edgeface|EdgeSAM|face_detection)' <<<"$list"; then
        die "$zip: arquivo proibido no pacote (.env, node_modules, db ou modelo de visão)."
    fi
    ok "$zip validado."
}
validate_zip "$DIST/$LINUX_ZIP" linux
validate_zip "$DIST/$WINDOWS_ZIP" windows

(cd "$DIST" && sha256sum "$LINUX_ZIP" "$WINDOWS_ZIP" > SHA256SUMS)
ok "SHA256SUMS gerado."

if (( ! LOCAL_ONLY )); then
    # ── Tag + push ──────────────────────────────────────────────────────────────
    info "Criando tag $TAG …"
    git tag -a "$TAG" -m "$TITLE"
    git push origin "$TAG"
    ok "Tag publicada: $TAG"

    # ── Release no GitHub ───────────────────────────────────────────────────────
    if gh release view "$TAG" >/dev/null 2>&1; then
        info "Release $TAG já existe — fazendo upload dos assets…"
        gh release upload "$TAG" "$DIST/$LINUX_ZIP" "$DIST/$WINDOWS_ZIP" --clobber
    else
        info "Criando release $TITLE …"
        gh release create "$TAG" --title "$TITLE" --notes-file "$DIST/RELEASE_NOTES.md" \
            "$DIST/$LINUX_ZIP" "$DIST/$WINDOWS_ZIP"
    fi
    ok "Release publicado: https://github.com/Lumazini-Code/alpha-virtual-assistant/releases/tag/$TAG"
else
    warn "Tag/release no GitHub pulados (--local-only)."
fi

# ── Reinstalação local em modo teste ──────────────────────────────────────────
if (( SKIP_INSTALL )); then
    warn "Instalação local pulada (--skip-install)."
else
    info "Reinstalando localmente com ALPHA_RELEASE_ZIP (modo teste)…"
    export ALPHA_RELEASE_ZIP="$DIST/$LINUX_ZIP"
    # Os \n aceitam os defaults do instalador: manter token, perfil econômico,
    # não mexer no PATH, etc. O .env da instalação é preservado (zip não tem .env).
    printf '\n\n\n\n\n\n' | bash ./install.sh --no-launch
    ok "Instalação local atualizada."
fi

if (( LOCAL_ONLY )); then
    info "Versão local $VERSION atualizada. Assets em $DIST/"
else
    info "Release $VERSION concluído. Assets em $DIST/"
fi
