# Transcript (pipeline de voz, C++ no host)

`main.cpp` é o único fonte compilado neste módulo: miniaudio → TEN VAD → wake word
(sherpa-onnx KWS) → ASR streaming → speaker embedding CAM++, com um mini servidor
HTTP em `:9100` (`/health`, `/speaker/embed`). O tray supervisiona o executável
como processo do host (`Manager/src/transcript.rs`).

## Binários pré-compilados

O motor pesado **não é compilado aqui**. Cada plataforma tem seus binários na
pasta correspondente:

| Pasta | Conteúdo |
|---|---|
| `linux/` | `libtranscribe.so`, `libten_vad.so`, `libonnxruntime.so`, `libsherpa-onnx-c-api.so`, `libsherpa-onnx-cxx-api.so`, `libggml*.so` |
| `windows/` | `transcribe.dll`/`.lib`, `ten_vad.dll`/`.lib`, `onnxruntime.dll`/`.lib`, `sherpa-onnx-*.dll`/`.lib`, `ggml*.dll` |

Esses binários são **ignorados pelo git** (`.gitignore`: `*.so`, `Modules/transcribe/linux/`,
`Modules/transcribe/windows/*`). O que está versionado em `windows/` são as import
libs (`.lib`), o `transcribe.def`, os headers de runtime e `contract.json` (versão,
hash do header e backends do conjunto de DLLs). Se as DLLs/`.so` estiverem faltando
num checkout novo, elas precisam ser copiadas/guardadas de algum lugar — o build
não as recria.

## Build

```bash
# Linux + Windows (cross-compile), a partir de qualquer diretório:
./Modules/transcribe/build.sh

# apenas um alvo:
./Modules/transcribe/build.sh linux
./Modules/transcribe/build.sh windows
```

Pré-requisitos (Debian/Ubuntu):

```bash
sudo apt install cmake
sudo apt install gcc-mingw-w64-x86-64 g++-mingw-w64-x86-64 binutils-mingw-w64-x86-64
```

Saída:

- `Modules/transcribe/transcript` — ELF x86-64. O RPATH `$ORIGIN` faz achar as
  `.so` na pasta do executável.
- `Modules/transcribe/transcript.exe` — PE x86-64. As `windows/*.dll` são copiadas
  para junto do `.exe` no `POST_BUILD` (o Windows procura DLLs na pasta do exe).

`make.sh` e `compiler.sh` continuam funcionando como atalhos apenas para Linux.

## Por que MinGW em vez de MSVC

As `.lib` versionadas são *short import libraries* COFF, geradas por
`windows/generate_import_lib.ps1` a partir dos exports do DLL. O `ld` do MinGW
as linka direto, então não é preciso toolchain da Microsoft nem REGENERAR as libs
para cross-compilar de Linux. Se aparecer `.dll.a` (via `dlltool`), o CMake prefere
ele. A API do `transcribe.h` é `extern "C"`, então não há incompatibilidade de ABI
C++ entre o compilador do exe e o compilador das DLLs.

A toolchain é `cmake/windows-mingw.cmake`. Ela define `CMAKE_SYSTEM_NAME=Windows`,
o que faz o `WIN32` do `CMakeLists.txt` ligar `ws2_32` e escolher `windows/` — a
plataforma é decidida pelo **alvo**, não pelo sistema onde o cmake roda.

## Runtime

O tray procura `Modules/transcribe/transcript` (Linux) ou `transcript.exe`
(Windows), com override por `AVA_TRANSCRIPT_BIN` (`Manager/src/state.rs`). O
`cwd` do processo é a pasta do binário: os modelos são relativos a
`../Models/...` (`nemotron-3.5-asr-streaming-0.6b-Q4_K_M.gguf`,
`3dspeaker_...onnx`, `sherpa-onnx/*`). Sem binário, o estado fica `NotInstalled`
e o tray não spawna nada.
