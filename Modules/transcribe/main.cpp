// Teste de transcrição streaming: nemotron-3.5-asr-streaming-0.6b (GGUF Q4_K_M)
// via C API nativa do transcribe.cpp, + embedding de locutor (CAM++ 3D-Speaker).
//
// Fale no microfone e o texto aparece no terminal:
//   - texto já confirmado (committed) em branco
//   - texto provisório (tentative) em cinza, muda enquanto você fala
//   - ao detectar silêncio, a frase é finalizada e fica gravada na linha
//   - logo depois, o embedding do locutor daquela frase é extraído
//
// Arquitetura:
//   callback do miniaudio  ->  ring buffer lock-free  ->  thread principal
//                                                     ->  thread de speaker ID
// O callback de áudio só copia amostras (tempo real, sem inferência dentro
// dele). VAD + transcribe_stream_* rodam todos na thread principal, então
// não há corrida entre feed/finalize/begin e nenhum áudio se perde entre
// uma frase e outra. O fbank + CAM++ rodam numa thread separada, fora do
// caminho crítico: a próxima frase já pode começar enquanto o embedding
// da anterior ainda está sendo calculado.
//
// Otimizações do caminho de embedding:
//   - só extrai de frases que o ASR realmente transcreveu (descartados saem fora)
//   - silêncio final do endpoint é cortado antes de virar feature
//   - frases longas são recortadas no centro (SPK_MAX_MS) — CAM++ satura rápido
//   - fbank próprio (Kaldi-compatível), com janela, tabelas de FFT e mel
//     pré-computadas e buffers reaproveitados entre chamadas
//   - uma única Ort::Session, criada no boot, com grafo otimizado
//
// Wake word (sherpa-onnx KeywordSpotter, zipformer streaming de ~3M params):
//   - roda NA MESMA thread principal, mas SÓ dentro de segmentos que o TEN VAD
//     já marcou como fala: em silêncio o custo é zero
//   - reaproveita exatamente os frames que o ASR já recebia (preroll incluso)
//     e a mesma conversão int16 -> float
//   - WAKE_GATE_ASR = true: o ASR de 0.6B só é aberto quando a wake word
//     dispara; o áudio acumulado da frase (incl. a wake word) é reenviado ao
//     ASR de uma vez. Frases sem wake word não gastam nenhuma inferência de ASR
//     nem de speaker embedding.
//   - WAKE_GATE_ASR = false: o ASR roda como antes e a wake word só anota.
//
// Embed server on-demand (rota A do plano — cadastro de voz via /settings):
//   - mini servidor HTTP (POSIX sockets) SÓ no loopback, TRANSCRIBE_HTTP_PORT
//     (default 9100), numa thread dedicada: GET /health e POST /speaker/embed
//     (body = PCM s16le mono; X-Sample-Rate, default 16000 — só 16k aceito).
//   - o orchestrator (POST /voice/register) encaminha o áudio gravado pela
//     TUI e persiste o embedding via /voice-dict/write na memory.
//   - embed() é síncrono e serializado com a thread assíncrona via ort_mu_
//     (Ort::Session + fbank_ compartilhados).
//
// Deps: miniaudio, TEN VAD (ten_vad.h + libten_vad), transcribe.cpp, ONNX Runtime,
//       sherpa-onnx (C API: sherpa-onnx/c-api/c-api.h), httplib.h (cliente HTTP)

#include <transcribe.h>
#include <parakeet.h>

#include <ten_vad.h>
#include <onnxruntime_cxx_api.h>
#include <c-api.h>

// miniaudio: captura de áudio (header-only; a implementation é compilada
// AQUI — este é o único TU que usa a lib).
#define MINIAUDIO_IMPLEMENTATION
#include <miniaudio.h>

// cpp-httplib: cliente HTTP p/ o orchestrator (header-only, sem linking).
#include <httplib.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------
// Compatibilidade Windows / Linux (embed server + console)
// ---------------------------------------------------------------------
#ifdef _WIN32
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #include <winsock2.h>      // ANTES de windows.h
  #include <ws2tcpip.h>
  #include <windows.h>

  using socket_t = SOCKET;
  static constexpr socket_t SOCK_INVALID = INVALID_SOCKET;
  static inline bool sock_valid(socket_t s) { return s != INVALID_SOCKET; }
  static inline void close_socket(socket_t s) { closesocket(s); }

  static inline bool sock_init() {
      WSADATA wsa;
      return WSAStartup(MAKEWORD(2, 2), &wsa) == 0;
  }
  static inline void sock_cleanup() { WSACleanup(); }

  // Ativa ANSI (\033[...) e UTF-8 no console do Windows, mantendo os prints
  // idênticos aos do Linux.
  #ifndef ENABLE_VIRTUAL_TERMINAL_PROCESSING
    #define ENABLE_VIRTUAL_TERMINAL_PROCESSING 0x0004
  #endif
  static void init_console() {
      SetConsoleOutputCP(CP_UTF8);
      SetConsoleCP(CP_UTF8);
      HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
      if (h != INVALID_HANDLE_VALUE) {
          DWORD mode = 0;
          if (GetConsoleMode(h, &mode))
              SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
      }
  }
#else
  #include <arpa/inet.h>
  #include <netinet/in.h>
  #include <sys/socket.h>
  #include <unistd.h>

  using socket_t = int;
  static constexpr socket_t SOCK_INVALID = -1;
  static inline bool sock_valid(socket_t s) { return s >= 0; }
  static inline void close_socket(socket_t s) { close(s); }
  static inline bool sock_init() { return true; }
  static inline void sock_cleanup() {}
  static void init_console() {}
#endif

#ifndef MSG_NOSIGNAL
  #define MSG_NOSIGNAL 0
#endif
#ifndef M_PI
  #define M_PI 3.14159265358979323846
#endif

// ---- Config ----
static constexpr const char* MODEL_PATH =
    "../Models/nemotron-3.5-asr-streaming-0.6b-Q4_K_M.gguf";
static constexpr const char* SPEAKER_MODEL_PATH =
    "../Models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx";
static constexpr const char* LANGUAGE = "pt-BR";
static constexpr int SAMPLE_RATE = 16000;
static constexpr int FRAME_SAMPLES = 256;                 // hop do TEN VAD (160 ou 256 são os otimizados)
static constexpr int FRAME_MS = FRAME_SAMPLES * 1000 / SAMPLE_RATE;   // 16ms
static constexpr float VAD_THRESHOLD = 0.5f;             // 0.0-1.0: maior = mais rígido
static constexpr int SILENCE_MS_TO_ENDPOINT = 400;       // silêncio que fecha a frase
static constexpr int SILENCE_MS_AFTER_WAKE = 1500;       // após a wake word: espera mais pra não cortar o comando
static constexpr int ONSET_FRAMES = 4;                   // ~64ms de fala seguida pra abrir
static constexpr int PREROLL_FRAMES = 13;                // ~208ms guardados antes do onset
static constexpr bool DEBUG_EMPTY = true;                // mostra frases descartadas (vazias) em cinza
static constexpr int ATT_CONTEXT_RIGHT = 3;              // menu do nemotron-3.5: {13, 6, 3, 0} => {1040,480,240,0}ms de lookahead

// ---- Config do speaker embedding ----
static constexpr int SPK_MIN_MS = 600;                   // abaixo disso o embedding não presta
static constexpr int SPK_MAX_MS = 6000;                  // recorte central pra frases longas
static constexpr int SPK_THREADS = 2;                    // threads intra-op do ONNX Runtime
static constexpr size_t SPK_QUEUE_MAX = 4;               // frases pendentes antes de descartar as velhas

// ---- Config da wake word (sherpa-onnx KWS) ----
// Modelo: sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01 (ajuste os nomes
// dos arquivos conforme o que veio no .tar.bz2).
static constexpr const char* KWS_ENCODER = "../Models/sherpa-onnx/encoder-epoch-13-avg-2-chunk-8-left-64.onnx";
static constexpr const char* KWS_DECODER = "../Models/sherpa-onnx/decoder-epoch-13-avg-2-chunk-8-left-64.onnx";
static constexpr const char* KWS_JOINER  = "../Models/sherpa-onnx/joiner-epoch-13-avg-2-chunk-8-left-64.onnx";
static constexpr const char* KWS_TOKENS  = "../Models/sherpa-onnx/tokens.txt";
static constexpr const char* KWS_KEYWORDS = "../Models/sherpa-onnx/keywords.txt";   // uma wake word por linha (já tokenizada)
static constexpr bool  WAKE_GATE_ASR = true;             // true: ASR só abre depois da wake word
// mostra frases ignoradas por falta de wake word (ativa se ALPHA_DEBUG_WAKE estiver setada no ambiente)
static bool debug_wake() { return std::getenv("ALPHA_DEBUG_WAKE") != nullptr; }
static constexpr float KWS_THRESHOLD = 0.25f;            // maior = menos falsos positivos, mais misses
static constexpr float KWS_SCORE = 1.0f;                 // boost global (cada linha pode ter o seu com :x.x)
static constexpr int   KWS_THREADS = 1;                  // modelo minúsculo: 1 thread basta
static std::atomic<bool> g_running{true};
static void on_sigint(int) { g_running = false; }

// stdout é escrito pela thread principal (transcrição) e pela thread de
// embedding; o mutex evita linhas embaralhadas.
static std::mutex g_print_mu;

static void check(transcribe_status st, const char* where) {
    if (st != TRANSCRIBE_OK) {
        std::fprintf(stderr, "%s: %s\n", where, transcribe_status_string(st));
        std::exit(1);
    }
}

// ---------------------------------------------------------------------
// Ring buffer SPSC (1 produtor = callback de áudio, 1 consumidor = main)
// ---------------------------------------------------------------------
class Ring {
public:
    explicit Ring(size_t capacity) : buf_(capacity) {}

    bool push(const int16_t* data, size_t n) {              // produtor
        size_t w = w_.load(std::memory_order_relaxed);
        size_t r = r_.load(std::memory_order_acquire);
        if (buf_.size() - (w - r) < n) return false;        // cheio: descarta
        for (size_t i = 0; i < n; i++) buf_[(w + i) % buf_.size()] = data[i];
        w_.store(w + n, std::memory_order_release);
        return true;
    }

    bool pop(int16_t* out, size_t n) {                      // consumidor
        size_t r = r_.load(std::memory_order_relaxed);
        size_t w = w_.load(std::memory_order_acquire);
        if (w - r < n) return false;
        for (size_t i = 0; i < n; i++) out[i] = buf_[(r + i) % buf_.size()];
        r_.store(r + n, std::memory_order_release);
        return true;
    }

private:
    std::vector<int16_t> buf_;
    std::atomic<size_t> w_{0}, r_{0};
};

static Ring g_ring(SAMPLE_RATE * 10);                       // 10s de folga
static std::atomic<unsigned long> g_dropped{0};

static void audio_data_callback(ma_device* pDevice, void* pOutput, const void* pInput,
                                ma_uint32 frameCount) {
    (void)pDevice; (void)pOutput;
    if (pInput && !g_ring.push(static_cast<const int16_t*>(pInput), frameCount)) {
        g_dropped++;
    }
}

// ---------------------------------------------------------------------
// FFT radix-2 (tabelas pré-computadas, in-place)
// ---------------------------------------------------------------------
class FFT {
public:
    explicit FFT(int n) : n_(n) {
        int logn = 0;
        while ((1 << logn) < n) logn++;
        rev_.resize(n);
        for (int i = 0; i < n; i++) {
            int r = 0;
            for (int b = 0; b < logn; b++)
                if ((i >> b) & 1) r |= 1 << (logn - 1 - b);
            rev_[i] = r;
        }
        cos_.resize(n / 2);
        sin_.resize(n / 2);
        for (int i = 0; i < n / 2; i++) {
            const double a = -2.0 * M_PI * i / n;
            cos_[i] = (float)std::cos(a);
            sin_[i] = (float)std::sin(a);
        }
    }

    void run(float* re, float* im) const {
        for (int i = 0; i < n_; i++) {
            const int j = rev_[i];
            if (j > i) { std::swap(re[i], re[j]); std::swap(im[i], im[j]); }
        }
        for (int len = 2; len <= n_; len <<= 1) {
            const int half = len / 2, step = n_ / len;
            for (int i = 0; i < n_; i += len) {
                for (int k = 0; k < half; k++) {
                    const float wr = cos_[k * step], wi = sin_[k * step];
                    const float xr = re[i + k + half], xi = im[i + k + half];
                    const float tr = xr * wr - xi * wi;
                    const float ti = xr * wi + xi * wr;
                    re[i + k + half] = re[i + k] - tr;
                    im[i + k + half] = im[i + k] - ti;
                    re[i + k] += tr;
                    im[i + k] += ti;
                }
            }
        }
    }

private:
    int n_;
    std::vector<int> rev_;
    std::vector<float> cos_, sin_;
};

// ---------------------------------------------------------------------
// Fbank Kaldi-compatível (80 mel bins, 25ms/10ms, janela Povey, CMN)
// É exatamente o que o 3D-Speaker usa no treino:
//   torchaudio.compliance.kaldi.fbank(wav * 32768, num_mel_bins=80)
//   feat = feat - feat.mean(dim=0)
// ---------------------------------------------------------------------
class Fbank {
public:
    static constexpr int kNumBins = 80;
    static constexpr int kFrameLen = 400;    // 25ms @16k
    static constexpr int kShift = 160;       // 10ms @16k
    static constexpr int kPadded = 512;
    static constexpr float kPreemph = 0.97f;
    static constexpr float kLowFreq = 20.0f;
    static constexpr float kHighFreq = SAMPLE_RATE / 2.0f;

    Fbank() : fft_(kPadded), re_(kPadded), im_(kPadded) {
        window_.resize(kFrameLen);
        for (int i = 0; i < kFrameLen; i++) {
            const double a = 2.0 * M_PI * i / (kFrameLen - 1);
            window_[i] = (float)std::pow(0.5 - 0.5 * std::cos(a), 0.85);  // Povey
        }

        const int num_fft_bins = kPadded / 2;
        const float bin_w = (float)SAMPLE_RATE / kPadded;
        const float mel_low = mel(kLowFreq), mel_high = mel(kHighFreq);
        const float delta = (mel_high - mel_low) / (kNumBins + 1);
        bins_.resize(kNumBins);
        for (int b = 0; b < kNumBins; b++) {
            const float left = mel_low + b * delta;
            const float center = left + delta;
            const float right = center + delta;
            MelBin& mb = bins_[b];
            mb.start = -1;
            for (int j = 0; j < num_fft_bins; j++) {
                const float m = mel(bin_w * j);
                if (m <= left) continue;
                if (m >= right) break;
                if (mb.start < 0) mb.start = j;
                mb.w.push_back(m <= center ? (m - left) / delta : (right - m) / delta);
            }
            if (mb.start < 0) mb.start = 0;
        }
    }

    // out = [frames * 80], já com CMN. Retorna o número de frames.
    int compute(const int16_t* pcm, size_t n, std::vector<float>& out) {
        if (n < (size_t)kFrameLen) { out.clear(); return 0; }
        const int frames = 1 + (int)((n - kFrameLen) / kShift);   // snip_edges=true
        out.resize((size_t)frames * kNumBins);

        for (int t = 0; t < frames; t++) {
            const int16_t* src = pcm + (size_t)t * kShift;
            float sum = 0.0f;
            for (int i = 0; i < kFrameLen; i++) { re_[i] = (float)src[i]; sum += re_[i]; }
            const float mean = sum / kFrameLen;
            for (int i = 0; i < kFrameLen; i++) re_[i] -= mean;              // remove DC
            for (int i = kFrameLen - 1; i > 0; i--) re_[i] -= kPreemph * re_[i - 1];
            re_[0] -= kPreemph * re_[0];                                     // pré-ênfase
            for (int i = 0; i < kFrameLen; i++) re_[i] *= window_[i];
            std::fill(re_.begin() + kFrameLen, re_.end(), 0.0f);
            std::fill(im_.begin(), im_.end(), 0.0f);
            fft_.run(re_.data(), im_.data());

            float* o = out.data() + (size_t)t * kNumBins;
            for (int b = 0; b < kNumBins; b++) {
                const MelBin& mb = bins_[b];
                float e = 0.0f;
                for (size_t k = 0; k < mb.w.size(); k++) {
                    const int j = mb.start + (int)k;
                    e += mb.w[k] * (re_[j] * re_[j] + im_[j] * im_[j]);
                }
                o[b] = std::log(std::max(e, std::numeric_limits<float>::epsilon()));
            }
        }

        // CMN: média por dimensão ao longo do tempo
        std::array<double, kNumBins> acc{};
        for (int t = 0; t < frames; t++) {
            const float* o = out.data() + (size_t)t * kNumBins;
            for (int b = 0; b < kNumBins; b++) acc[b] += o[b];
        }
        for (int b = 0; b < kNumBins; b++) acc[b] /= frames;
        for (int t = 0; t < frames; t++) {
            float* o = out.data() + (size_t)t * kNumBins;
            for (int b = 0; b < kNumBins; b++) o[b] -= (float)acc[b];
        }
        return frames;
    }

private:
    struct MelBin { int start; std::vector<float> w; };
    static float mel(float f) { return 1127.0f * std::log(1.0f + f / 700.0f); }

    FFT fft_;
    std::vector<float> re_, im_, window_;
    std::vector<MelBin> bins_;
};

// ---------------------------------------------------------------------
// Poster do orchestrator: centraliza os POSTs HTTP do pipeline de voz.
//  - /voice/ingest: embedding do locutor (fire-and-forget, nunca bloqueia
//    a thread do CAM++ nem a thread principal).
//  - /execute: o texto transcrito da fala vira um turno completo (tool loop
//    normal). Sequencial: um turno por vez — se um POST anterior ainda está
//    pendente/em andamento, a nova utterance é recusada com log (evita
//    acumular turnos fire-and-forget).
// session_id=null: o sistema é single-session — o orchestrator cacheia como
// "__anon__" e o /execute consome essa entrada.
// ---------------------------------------------------------------------
static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if ((unsigned char)c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", (unsigned char)c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

class OrchestratorPoster {
public:
    bool init() {
        const char* port = std::getenv("ORCHESTRATOR_PORT");
        base_ = std::string("http://127.0.0.1:")
             + (port && *port ? port : "9000");
        const char* tok = std::getenv("AVA_INTERNAL_TOKEN");
        if (tok && *tok) token_ = tok;
        worker_ = std::thread(&OrchestratorPoster::loop, this);
        return true;
    }

    void submit_ingest(std::vector<float>&& emb) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (queue_.size() >= QUEUE_MAX) queue_.pop_front();   // atrasou: joga a mais velha fora
            queue_.push_back(Job{Kind::Ingest, std::move(emb), {}});
        }
        cv_.notify_one();
    }

    // Recusa se já existe um turno de voz pendente/em andamento.
    void submit_execute(const std::string& text) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            for (const auto& j : queue_)
                if (j.kind == Kind::Execute) {
                    std::lock_guard<std::mutex> plk(g_print_mu);
                    std::fprintf(stderr, "\033[2m[voice exec: turno anterior em andamento — novo pedido ignorado]\033[0m\n");
                    return;
                }
            queue_.push_back(Job{Kind::Execute, {}, text});
        }
        cv_.notify_one();
    }

    // ---- sequenciamento ingest → execute (risco 1 do plano) ----
    // O texto do turno é "palco" (staged) no finalize e só é postado DEPOIS
    // que o embedding correspondente foi ingerido — garante que o contexto
    // de voz (quem está falando) está no cache do orchestrator antes do
    // _recall_for_turn do /execute consumi-lo.
    uint64_t next_seq() {
        std::lock_guard<std::mutex> lk(mu_);
        return ++seq_;
    }

    void stage_execute(uint64_t seq, std::string text) {
        std::lock_guard<std::mutex> lk(mu_);
        pending_execute_[seq] = std::move(text);
    }

    // Chamado pelo SpeakerEmbedder logo após o submit_ingest do turno.
    void take_and_execute(uint64_t seq) {
        std::string text;
        {
            std::lock_guard<std::mutex> lk(mu_);
            auto it = pending_execute_.find(seq);
            if (it != pending_execute_.end()) {
                text = std::move(it->second);
                pending_execute_.erase(it);
            }
        }
        if (!text.empty()) submit_execute(text);
    }

    // Turno descartado (frase curta demais etc.) — nada pra executar.
    void drop_pending(uint64_t seq) {
        std::lock_guard<std::mutex> lk(mu_);
        pending_execute_.erase(seq);
    }

    void shutdown() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        if (worker_.joinable()) worker_.join();
    }

    // POST fire-and-forget para o MANAGER (tray, :9001): abre a janela direto
    // na tela de chat assim que a wake word dispara — ANTES da transcrição
    // terminar. Thread destacada: nunca bloqueia a thread de processamento
    // de áudio.
    void notify_ui_chat_open() {
        std::thread([] {
            const char* port = std::getenv("AVA_MANAGER_PORT");
            const std::string base = std::string("http://127.0.0.1:")
                 + (port && *port ? port : "9001");
            httplib::Client cli(base);
            cli.set_connection_timeout(0, 500000);   // 500ms
            cli.set_read_timeout(1, 500000);         // ~1.5s no total
            auto res = cli.Post("/ui/chat-open", "{}", "application/json");
            if (!res) {
                std::lock_guard<std::mutex> lk(g_print_mu);
                std::fprintf(stderr, "\033[2m[ui chat-open: %s]\033[0m\n",
                             httplib::to_string(res.error()).c_str());
            }
        }).detach();
    }

private:
    enum class Kind { Ingest, Execute };
    struct Job {
        Kind kind;
        std::vector<float> emb;
        std::string text;
    };


    void loop() {
        for (;;) {
            Job job;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [this] { return stop_ || !queue_.empty(); });
                if (queue_.empty()) return;
                job = std::move(queue_.front());
                queue_.pop_front();
            }
            if (job.kind == Kind::Ingest) post_ingest(job.emb);
            else post_execute(job.text);
        }
    }

    // Post genérico; retorna true em HTTP 200. `log_tag` mantém o padrão
    // [voice ingest: ...] / [voice exec: ...].
    // `idle_timeout_s` é o watchdog: o read_timeout do httplib vale por
    // recv() — aborta se a resposta ficar sem bytes por esse tempo, sem
    // matar respostas longas que estão progredindo normalmente (SSE do
    // /execute).
    bool post(const char* path, const std::string& body,
              long idle_timeout_s, const char* log_tag) {
        httplib::Client cli(base_);
        cli.set_connection_timeout(1, 0);            // 1s de connect
        cli.set_read_timeout(idle_timeout_s, 0);     // watchdog por recv()
        cli.set_write_timeout(30, 0);

        httplib::Headers hdrs;
        hdrs.emplace("Content-Type", "application/json");
        if (!token_.empty())
            hdrs.emplace("Authorization", "Bearer " + token_);

        const auto res = cli.Post(path, hdrs, body, "application/json");
        if (!res) {
            std::lock_guard<std::mutex> lk(g_print_mu);
            std::fprintf(stderr, "\033[2m[%s: %s]\033[0m\n", log_tag,
                         httplib::to_string(res.error()).c_str());
            return false;
        }
        if (res->status != 200) {
            std::lock_guard<std::mutex> lk(g_print_mu);
            std::fprintf(stderr, "\033[2m[%s: HTTP %d]\033[0m\n", log_tag, res->status);
            return false;
        }
        return true;
    }

    void post_ingest(const std::vector<float>& emb) {
        // {"session_id": null, "embedding": [f1, f2, ...]}
        std::string body;
        body.reserve(emb.size() * 10 + 64);
        body += "{\"session_id\":null,\"embedding\":[";
        char num[32];
        for (size_t i = 0; i < emb.size(); i++) {
            std::snprintf(num, sizeof(num), "%.7g", emb[i]);
            if (i) body += ',';
            body += num;
        }
        body += "]}";

        if (post("/voice/ingest", body, 3L, "voice ingest")) {
            std::lock_guard<std::mutex> lk(g_print_mu);
            std::printf("\033[2m[voice ingest: ok (%zu dims)]\033[0m\n", emb.size());
            std::fflush(stdout);
        }
    }

    void post_execute(const std::string& text) {
        // ExecuteRequest: {"input": ..., "session_id": null, "lang": "pt",
        //                  "tts": true, "tts_output": "device", "source": "voice"}
        std::string body = "{\"input\":\"" + json_escape(text) + "\","
                           "\"session_id\":null,\"lang\":\"pt\","
                           "\"tts\":true,\"tts_output\":\"device\",\"source\":\"voice\"}";

        // /execute pode levar minutos (tool loop). Timeout longo (10 min) +
        // watchdog de inatividade (sem bytes por 60s aborta).
        if (post("/execute", body, 60L, "voice exec")) {
            std::lock_guard<std::mutex> lk(g_print_mu);
            std::printf("\033[2m[voice exec: ok]\033[0m\n");
            std::fflush(stdout);
        }
    }

    static constexpr size_t QUEUE_MAX = 4;
    std::string base_;
    std::string token_;
    std::deque<Job> queue_;
    uint64_t seq_ = 0;
    std::map<uint64_t, std::string> pending_execute_;   // seq → texto aguardando o ingest
    std::mutex mu_;
    std::condition_variable cv_;
    bool stop_ = false;
    std::thread worker_;
};

static OrchestratorPoster g_orchestrator;

// ---------------------------------------------------------------------
// Speaker embedding (CAM++ / 3D-Speaker) numa thread dedicada
// ---------------------------------------------------------------------
class SpeakerEmbedder {
public:
    bool init(const char* model_path) {
        try {
            opts_.SetIntraOpNumThreads(SPK_THREADS);
            opts_.SetInterOpNumThreads(1);
            opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            session_ = std::make_unique<Ort::Session>(env_, model_path, opts_);

            Ort::AllocatorWithDefaultOptions alloc;
            in_name_ = session_->GetInputNameAllocated(0, alloc).get();
            out_name_ = session_->GetOutputNameAllocated(0, alloc).get();
        } catch (const Ort::Exception& e) {
            std::fprintf(stderr, "ONNX (%s): %s\n", model_path, e.what());
            return false;
        }
        worker_ = std::thread(&SpeakerEmbedder::loop, this);
        return true;
    }

    void submit(std::vector<int16_t>&& pcm, uint64_t seq) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (queue_.size() >= SPK_QUEUE_MAX) queue_.pop_front();   // atrasou: joga a mais velha fora
            queue_.push_back(SpkJob{std::move(pcm), seq});
        }
        cv_.notify_one();
    }

    // Drena o que sobrou e encerra a thread.
    void shutdown() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        if (worker_.joinable()) worker_.join();
        session_.reset();
    }

    struct EmbedResult {
        std::vector<float> emb;
        bool degenerate = false;   // áudio curto demais (fbank < 10 frames)
        std::string error;         // Ort::Exception (vazio = ok)
    };

    // Extração on-demand p/ cadastro de voz (rota /speaker/embed): síncrona
    // e thread-safe. A Ort::Session e o fbank_ são compartilhados com a
    // thread assíncrona — serializamos com ort_mu_ (infreqüente: só quando
    // alguém cadastra uma voz, ~1s por gravação).
    // pcm: s16le mono na SAMPLE_RATE (16k). Embeddings saem não-normalizados
    // (o memory normaliza no write).
    EmbedResult embed(const std::vector<int16_t>& pcm) {
        std::lock_guard<std::mutex> lk(ort_mu_);
        return embed_locked(pcm);
    }

private:
    // Core sem o lock ort_mu_ — chamado por embed() e pelo loop assíncrono.
    EmbedResult embed_locked(const std::vector<int16_t>& pcm) {
        EmbedResult r;
        const int frames = fbank_.compute(pcm.data(), pcm.size(), feats_);
        if (frames < 10) {                       // áudio degenerado
            r.degenerate = true;
            return r;
        }
        try {
            const std::array<int64_t, 3> shape{1, frames, Fbank::kNumBins};
            Ort::Value input = Ort::Value::CreateTensor<float>(
                mem_info_, feats_.data(), feats_.size(), shape.data(), shape.size());

            const char* in_names[] = {in_name_.c_str()};
            const char* out_names[] = {out_name_.c_str()};
            auto outputs = session_->Run(Ort::RunOptions{nullptr}, in_names, &input, 1,
                                         out_names, 1);

            const float* emb = outputs[0].GetTensorData<float>();
            const size_t dim = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
            r.emb.assign(emb, emb + dim);
        } catch (const Ort::Exception& e) {
            r.error = e.what();
        }
        return r;
    }

    struct SpkJob { std::vector<int16_t> pcm; uint64_t seq; };

    void loop() {
        for (;;) {
            SpkJob job;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [this] { return stop_ || !queue_.empty(); });
                if (queue_.empty()) return;            // stop_ e nada pendente
                job = std::move(queue_.front());
                queue_.pop_front();
            }
            process(job.pcm, job.seq);
        }
    }

    void process(std::vector<int16_t>& pcm, uint64_t seq) {
        const auto t0 = std::chrono::steady_clock::now();

        // fbanks/feats_ e a Ort::Session são compartilhados com o embed()
        // on-demand (rota /speaker/embed) — ort_mu_ serializa os dois.
        EmbedResult res;
        {
            std::lock_guard<std::mutex> lk(ort_mu_);
            res = embed_locked(pcm);
        }

        if (res.degenerate) {                    // áudio degenerado: turno não roda
            g_orchestrator.drop_pending(seq);
            return;
        }
        if (res.emb.empty()) {
            // Embedding falhou (Ort::Exception): o turno segue mesmo assim,
            // só sem contexto de voz.
            g_orchestrator.take_and_execute(seq);
            std::lock_guard<std::mutex> lk(g_print_mu);
            std::fprintf(stderr, "\nspeaker embedding: %s\n", res.error.c_str());
            return;
        }

        // ---- ponto de extensão ----
        // Aqui você pluga o tratamento (enrollment, cosine, diarização...).
        last_embedding_.assign(res.emb.begin(), res.emb.end());

        // Fase 2 — ingestão: dispara POST /voice/ingest com o embedding
        // (fire-and-forget na thread do OrchestratorPoster; não bloqueia o CAM++).
        const size_t emb_dim = res.emb.size();
        g_orchestrator.submit_ingest(std::move(res.emb));

        // O /execute só é enfileirado DEPOIS do ingest — a ordem no
        // orchestrator fica garantida (contexto de voz disponível no turno).
        g_orchestrator.take_and_execute(seq);

        const double took_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        const double audio_s = (double)pcm.size() / SAMPLE_RATE;

        std::lock_guard<std::mutex> lk(g_print_mu);
        std::printf("\033[2m[embedding gerado: %zu dims | %.2fs de fala | %.1f ms]\033[0m\n",
                    emb_dim, audio_s, took_ms);
        std::fflush(stdout);
    }

    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "speaker"};
    Ort::SessionOptions opts_;
    Ort::MemoryInfo mem_info_ =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    std::unique_ptr<Ort::Session> session_;
    std::string in_name_, out_name_;

    Fbank fbank_;
    std::vector<float> feats_;          // reaproveitado entre frases
    std::vector<float> last_embedding_;

    std::mutex ort_mu_;                 // serializa embed() (on-demand) vs thread assíncrona

    std::deque<SpkJob> queue_;
    std::mutex mu_;
    std::condition_variable cv_;
    bool stop_ = false;
    std::thread worker_;
};

static SpeakerEmbedder g_speaker;

// ---------------------------------------------------------------------
// Mini servidor HTTP local (rota A do plano — embedding on-demand):
//   GET  /health          → {"ok":true,"speaker":true|false}
//   POST /speaker/embed   → body: PCM s16le mono (X-Sample-Rate, default 16000)
//                           resp: {"ok":true,"dim":N,"embedding":[...]}
// Uma thread dedicada, uma conexão por vez (cadastro é infreqüente).
// Escuta SÓ no loopback — é consumido pelo orchestrator (que roda no
// container ava com network_mode: host no Linux) pra gerar o embedding
// de uma gravação feita pela TUI (/settings → cadastrar voz).
// ---------------------------------------------------------------------
class EmbedServer {
public:
    void init(SpeakerEmbedder* speaker) {
        speaker_ = speaker;
        const char* port_env = std::getenv("TRANSCRIBE_HTTP_PORT");
        port_ = port_env && *port_env ? std::atoi(port_env) : 9100;
        if (port_ <= 0 || port_ > 65535) port_ = 9100;

        if (!sock_init()) {
            std::fprintf(stderr, "embed server: WSAStartup falhou — cadastro on-demand indisponível\n");
            return;
        }
        listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
        if (!sock_valid(listen_fd_)) {
            std::fprintf(stderr, "embed server: socket() falhou — cadastro on-demand indisponível\n");
            sock_cleanup();
            return;
        }
        int one = 1;
        setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, (const char*)&one, sizeof(one));

        sockaddr_in addr{};
        addr.sin_family      = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);   // só loopback
        addr.sin_port        = htons((uint16_t)port_);

        if (bind(listen_fd_, (sockaddr*)&addr, sizeof(addr)) != 0 ||
            listen(listen_fd_, 4) != 0) {
            std::fprintf(stderr, "embed server: bind/listen :%d falhou — cadastro on-demand indisponível\n", port_);
            close_socket(listen_fd_);
            listen_fd_ = SOCK_INVALID;
            sock_cleanup();
            return;
        }
        worker_ = std::thread(&EmbedServer::loop, this);
        std::printf("Embed server on-demand: http://127.0.0.1:%d/speaker/embed\n", port_);
    }

    void shutdown() {
        stop_ = true;
        if (sock_valid(listen_fd_)) {
#ifdef _WIN32
            ::shutdown(listen_fd_, SD_BOTH);   // desbloqueia o accept()
#else
            ::shutdown(listen_fd_, SHUT_RDWR);   // desbloqueia o accept()
#endif
            close_socket(listen_fd_);
            listen_fd_ = SOCK_INVALID;
        }
        if (worker_.joinable()) worker_.join();
        sock_cleanup();
    }

private:
    void loop() {
        while (!stop_.load()) {
            sockaddr_in peer{};
            socklen_t peer_len = sizeof(peer);
            const socket_t fd = accept(listen_fd_, (sockaddr*)&peer, &peer_len);
            if (!sock_valid(fd)) {
                if (stop_.load()) return;
                continue;   // erro transitório (EINTR etc.)
            }
            handle(fd);
            close_socket(fd);
        }
    }

    // HTTP mínimo: request line + headers até \r\n\r\n, body por Content-Length.
    // Uma requisição por conexão (Connection: close).
    void handle(socket_t fd) {
        std::string req;
        char buf[4096];
        const size_t kHeaderCap = 16 * 1024;
        while (req.size() < kHeaderCap) {
            const int n = recv(fd, buf, (int)sizeof(buf), 0);
            if (n <= 0) return;
            req.append(buf, (size_t)n);
            if (req.find("\r\n\r\n") != std::string::npos) break;
        }
        const size_t header_end = req.find("\r\n\r\n");
        if (header_end == std::string::npos) { respond(fd, 400, "{\"ok\":false,\"error\":\"request sem fim de headers\"}"); return; }

        // request line: "POST /speaker/embed HTTP/1.1"
        const size_t line_end = req.find("\r\n");
        const std::string req_line = req.substr(0, line_end == std::string::npos ? req.size() : line_end);
        const size_t sp1 = req_line.find(' ');
        const size_t sp2 = req_line.find(' ', sp1 == std::string::npos ? 0 : sp1 + 1);
        const std::string method = sp1 == std::string::npos ? "" : req_line.substr(0, sp1);
        const std::string path   = (sp1 == std::string::npos || sp2 == std::string::npos)
                                       ? "" : req_line.substr(sp1 + 1, sp2 - sp1 - 1);

        // Content-Length (o body pode ter vindo parcialmente no 1º recv — recv_all cobre)
        size_t content_length = 0;
        {
            size_t pos = req.find("\r\nContent-Length:", 0);
            if (pos == std::string::npos) pos = req.find("\r\ncontent-length:", 0);
            if (pos != std::string::npos) {
                const size_t vstart = req.find(':', pos) + 1;
                const size_t vend   = req.find("\r\n", vstart);
                content_length = (size_t)std::strtoll(req.substr(vstart, vend - vstart).c_str(), nullptr, 10);
            }
        }

        if (path == "/health" && method == "GET") {
            char body[64];
            std::snprintf(body, sizeof(body), "{\"ok\":true,\"speaker\":%s}",
                          speaker_ ? "true" : "false");
            respond(fd, 200, body);
            return;
        }
        if (path == "/speaker/embed" && method == "POST") {
            if (!speaker_) { respond(fd, 503, "{\"ok\":false,\"error\":\"speaker (CAM++) nao carregado\"}"); return; }

            const size_t kBodyCap = 16000 * 2 * 180;   // teto: ~3min de PCM s16le mono 16k
            if (content_length == 0 || content_length > kBodyCap) {
                respond(fd, 400, "{\"ok\":false,\"error\":\"body vazio ou maior que o teto (PCM s16le mono)\"}");
                return;
            }

            // X-Sample-Rate (o CAM++/fbank é treinado a 16k — recusa o resto)
            int sample_rate = 16000;
            {
                size_t pos = req.find("\r\nX-Sample-Rate:");
                if (pos != std::string::npos) {
                    const size_t vstart = req.find(':', pos) + 1;
                    const size_t vend   = req.find("\r\n", vstart);
                    sample_rate = std::atoi(req.substr(vstart, vend - vstart).c_str());
                }
            }
            if (sample_rate != SAMPLE_RATE) {
                char body[128];
                std::snprintf(body, sizeof(body),
                              "{\"ok\":false,\"error\":\"sample_rate %d nao suportado (use %d)\"}", sample_rate, SAMPLE_RATE);
                respond(fd, 400, body);
                return;
            }

            // completa o body (o header end já consumido pode ter trazido bytes)
            const size_t content_length_orig = content_length;
            std::string body_bytes = req.substr(header_end + 4);
            if (body_bytes.size() > content_length_orig) body_bytes.resize(content_length_orig);
            size_t remaining = content_length_orig - body_bytes.size();
            while (remaining > 0) {
                const int n = recv(fd, buf, (int)std::min(sizeof(buf), remaining), 0);
                if (n <= 0) break;
                body_bytes.append(buf, (size_t)n);
                remaining -= (size_t)n;
            }
            if (remaining > 0) {
                respond(fd, 400, "{\"ok\":false,\"error\":\"body truncado\"}");
                return;
            }

            // bytes → s16le mono (descarta byte ímpar final, se houver)
            const size_t n_samples = body_bytes.size() / 2;
            std::vector<int16_t> pcm(n_samples);
            std::memcpy(pcm.data(), body_bytes.data(), n_samples * 2);

            const auto res = speaker_->embed(pcm);
            if (!res.emb.empty()) {
                // {"ok":true,"dim":N,"embedding":[f1, f2, ...]}
                std::string body;
                body.reserve(res.emb.size() * 10 + 48);
                body += "{\"ok\":true,\"dim\":";
                body += std::to_string(res.emb.size());
                body += ",\"embedding\":[";
                char num[32];
                for (size_t i = 0; i < res.emb.size(); i++) {
                    std::snprintf(num, sizeof(num), "%.7g", res.emb[i]);
                    if (i) body += ',';
                    body += num;
                }
                body += "]}";
                respond(fd, 200, body);
            } else if (res.degenerate) {
                respond(fd, 400, "{\"ok\":false,\"error\":\"audio muito curto/degenerado\"}");
            } else {
                std::string body = "{\"ok\":false,\"error\":\"embed falhou: ";
                body += json_escape(res.error);
                body += "\"}";
                respond(fd, 500, body);
            }
            return;
        }

        respond(fd, 404, "{\"ok\":false,\"error\":\"nao encontrado\"}");
    }

    void respond(socket_t fd, int status, const std::string& body) {
        const char* status_text = status == 200 ? "OK"
                                : status == 400 ? "Bad Request"
                                : status == 404 ? "Not Found"
                                : status == 503 ? "Service Unavailable"
                                : "Internal Server Error";
        char head[128];
        std::snprintf(head, sizeof(head),
                      "HTTP/1.1 %d %s\r\nContent-Type: application/json\r\n"
                      "Content-Length: %zu\r\nConnection: close\r\n\r\n",
                      status, status_text, body.size());
        send_all(fd, head, std::strlen(head));
        send_all(fd, body.data(), body.size());
    }

    static void send_all(socket_t fd, const void* data, size_t len) {
        const char* p = (const char*)data;
        while (len > 0) {
            // MSG_NOSIGNAL: sem SIGPIPE se o cliente sumir no meio da resposta
            // (no-op no Windows: lá send() não levanta SIGPIPE).
            const int n = send(fd, p, (int)std::min(len, (size_t)65536), MSG_NOSIGNAL);
            if (n <= 0) return;
            p += n;
            len -= (size_t)n;
        }
    }

    SpeakerEmbedder* speaker_ = nullptr;
    int port_ = 9100;
    socket_t listen_fd_ = SOCK_INVALID;
    std::atomic<bool> stop_{false};
    std::thread worker_;
};

static EmbedServer g_embed_server;

// Fecha o turno de voz: postura do embedding + disparo do /execute.
//  - Com CAM++: o texto fica "palco" (staged) e o SpeakerEmbedder dispara o
//    /execute logo depois do POST /voice/ingest — o contexto de voz chega ao
//    orchestrator ANTES do turno consumi-lo no _recall_for_turn.
//  - Sem CAM++ (speaker_ok=false): execute direto, sem contexto de voz.
//  - Frase curta demais (abaixo de SPK_MIN_MS, ex.: "hm", "ok"): descartada —
//    nem embedding nem LLM (mesmo critério de descarte do embedding).
static void submit_turn(const std::string& text, std::vector<int16_t>& utt,
                        int silence_ms, bool speaker_ok) {
    const size_t trim = (size_t)(silence_ms / FRAME_MS) * FRAME_SAMPLES;
    const size_t n = utt.size() > trim ? utt.size() - trim : 0;
    if (n < (size_t)SPK_MIN_MS * SAMPLE_RATE / 1000) return;   // curto demais: nem embedding nem LLM

    const size_t max_n = (size_t)SPK_MAX_MS * SAMPLE_RATE / 1000;
    size_t off = 0;
    size_t cut = n;
    if (cut > max_n) { off = (cut - max_n) / 2; cut = max_n; }

    if (speaker_ok) {
        const uint64_t seq = g_orchestrator.next_seq();
        g_orchestrator.stage_execute(seq, text);
        g_speaker.submit(std::vector<int16_t>(utt.begin() + off, utt.begin() + off + cut), seq);
    } else {
        g_orchestrator.submit_execute(text);
    }
}

// ---------------------------------------------------------------------
// Wake word (sherpa-onnx KeywordSpotter)
// Um spotter criado no boot (modelo + grafo de keywords carregados uma vez);
// um stream novo por frase (barato: só os estados iniciais do encoder).
// Só a thread principal usa esta classe.
// ---------------------------------------------------------------------
class WakeWord {
public:
    bool init() {
        SherpaOnnxKeywordSpotterConfig cfg;
        std::memset(&cfg, 0, sizeof(cfg));
        cfg.feat_config.sample_rate = SAMPLE_RATE;
        cfg.feat_config.feature_dim = 80;
        cfg.model_config.transducer.encoder = KWS_ENCODER;
        cfg.model_config.transducer.decoder = KWS_DECODER;
        cfg.model_config.transducer.joiner = KWS_JOINER;
        cfg.model_config.tokens = KWS_TOKENS;
        cfg.model_config.provider = "cpu";
        cfg.model_config.num_threads = KWS_THREADS;
        cfg.model_config.debug = 0;
        cfg.max_active_paths = 4;
        cfg.num_trailing_blanks = 1;
        cfg.keywords_score = KWS_SCORE;
        cfg.keywords_threshold = KWS_THRESHOLD;
        cfg.keywords_file = KWS_KEYWORDS;
        spotter_ = SherpaOnnxCreateKeywordSpotter(&cfg);
        return spotter_ != nullptr;
    }

    // Início de uma frase: stream limpo.
    void begin() {
        end();
        last_unmatched_.clear();
        stream_ = SherpaOnnxCreateKeywordStream(spotter_);
    }

    // Alimenta um frame (float [-1,1]). true se alguma wake word disparou.
    bool feed(const float* f32, int n, std::string& name) {
        if (!stream_) return false;
        SherpaOnnxOnlineStreamAcceptWaveform(stream_, SAMPLE_RATE, f32, n);
        return drain(name);
    }

    // Fim da frase sem disparo: força o tail padding do modelo, pra pegar uma
    // wake word dita no finalzinho (ex.: só "Ava", com pouco silêncio depois).
    bool flush(std::string& name) {
        if (!stream_) return false;
        SherpaOnnxOnlineStreamInputFinished(stream_);
        return drain(name);
    }

    void end() {
        if (stream_) { SherpaOnnxDestroyOnlineStream(stream_); stream_ = nullptr; }
    }

    void shutdown() {
        end();
        if (spotter_) { SherpaOnnxDestroyKeywordSpotter(spotter_); spotter_ = nullptr; }
    }

    // O que o KWS identificou na frase atual sem corresponder a nenhuma keyword (para debug).
    const std::string& last_unmatched() const { return last_unmatched_; }

private:
    bool drain(std::string& name) {
        while (SherpaOnnxIsKeywordStreamReady(spotter_, stream_)) {
            SherpaOnnxDecodeKeywordStream(spotter_, stream_);
            const SherpaOnnxKeywordResult* r = SherpaOnnxGetKeywordResult(spotter_, stream_);
            const bool hit = r && r->keyword && r->keyword[0] != '\0';
            if (hit) name = r->keyword;
            // debug: guarda o que o KWS identificou mesmo sem ser uma keyword alvo
            else if (r && r->keyword && r->keyword[0] != '\0') last_unmatched_ = r->keyword;
            if (r) SherpaOnnxDestroyKeywordResult(r);
            if (hit) return true;
        }
        return false;
    }

    const SherpaOnnxKeywordSpotter* spotter_ = nullptr;
    const SherpaOnnxOnlineStream* stream_ = nullptr;
    std::string last_unmatched_;
};

static WakeWord g_wake;

// ---------------------------------------------------------------------
// Saída no terminal
// ---------------------------------------------------------------------
// Reescreve a linha atual: committed normal + tentative em cinza.
static bool has_content(const char* s, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (!std::isspace((unsigned char)s[i])) return true;
    }
    return false;
}

// Retorna true se havia texto. Numa linha final vazia, só limpa a linha (sem "\n").
// `out_text` (opcional) recebe o texto final — committed + tentative — para o
// turno de voz (POST /execute).
static bool print_snapshot(struct transcribe_session* session, bool final_line,
                           std::string* out_text = nullptr) {
    struct transcribe_stream_text text;
    transcribe_stream_text_init(&text);
    if (transcribe_stream_get_text(session, &text) != TRANSCRIBE_OK) return false;

    const bool has_committed = has_content(text.committed_text, text.committed_text_bytes);
    const bool has_tentative = has_content(text.tentative_text, text.tentative_text_bytes);
    const bool has_text = has_committed || has_tentative;

    if (out_text) {
        if (has_committed) out_text->append(text.committed_text, text.committed_text_bytes);
        if (has_tentative) out_text->append(text.tentative_text, text.tentative_text_bytes);
    }

    std::lock_guard<std::mutex> lk(g_print_mu);
    if (final_line && !has_text) {
        std::printf("\r\033[K");
        std::fflush(stdout);
        return false;
    }

    std::printf("\r\033[K%.*s\033[2m%.*s\033[0m",
                (int)text.committed_text_bytes, text.committed_text,
                (int)text.tentative_text_bytes, text.tentative_text);
    if (final_line) std::printf("\n");
    std::fflush(stdout);
    return has_text;
}

// ---------------------------------------------------------------------
// Utterance
// ---------------------------------------------------------------------
static void begin_utterance(struct transcribe_session* session) {
    struct transcribe_run_params run_params;
    transcribe_run_params_init(&run_params);
    run_params.language = LANGUAGE;

    struct transcribe_stream_params stream_params;
    transcribe_stream_params_init(&stream_params);
    stream_params.commit_policy = TRANSCRIBE_STREAM_COMMIT_STABLE_PREFIX;

    // static: stream_params.family aponta pra cá até o begin consumir.
    static struct transcribe_parakeet_stream_ext parakeet_ext;
    transcribe_parakeet_stream_ext_init(&parakeet_ext);
    parakeet_ext.att_context_right = ATT_CONTEXT_RIGHT;

    if (transcribe_model_accepts_ext_kind(transcribe_get_model(session),
                                          TRANSCRIBE_EXT_SLOT_STREAM,
                                          TRANSCRIBE_EXT_KIND_PARAKEET_STREAM)) {
        stream_params.family = &parakeet_ext.ext;
    }

    check(transcribe_stream_begin(session, &run_params, &stream_params),
          "transcribe_stream_begin");
}

static void to_f32(const int16_t* pcm16, std::vector<float>& f32) {
    f32.resize(FRAME_SAMPLES);
    for (int i = 0; i < FRAME_SAMPLES; i++) f32[i] = pcm16[i] / 32768.0f;
}

// `print=false` no replay do áudio acumulado (evita redesenhar a linha 60x).
static void asr_feed(struct transcribe_session* session, const float* f32, bool print) {
    struct transcribe_stream_update update;
    transcribe_stream_update_init(&update);
    transcribe_status s =
        transcribe_stream_feed(session, f32, FRAME_SAMPLES, &update);

    if (s != TRANSCRIBE_OK) {
        std::fprintf(stderr, "\ntranscribe_stream_feed: %s\n",
                     transcribe_status_string(s));
    } else if (print && update.result_changed) {
        print_snapshot(session, false);
    }
}

static bool finalize_utterance(struct transcribe_session* session,
                               std::string* out_text = nullptr) {
    struct transcribe_stream_update update;
    transcribe_stream_update_init(&update);
    transcribe_status s = transcribe_stream_finalize(session, &update);
    if (s != TRANSCRIBE_OK) {
        std::fprintf(stderr, "\ntranscribe_stream_finalize: %s\n",
                     transcribe_status_string(s));
    }
    return print_snapshot(session, true, out_text);
}

// ---------------------------------------------------------------------
int main() {
    init_console();
    std::signal(SIGINT, on_sigint);
#ifdef SIGPIPE
    std::signal(SIGPIPE, SIG_IGN);   // embed server: cliente sumindo não pode matar o processo (POSIX)
#endif

    transcribe_init_backends_default();

    struct transcribe_model_load_params load_params;
    transcribe_model_load_params_init(&load_params);
    load_params.backend = TRANSCRIBE_BACKEND_AUTO;

    struct transcribe_session_params session_params;
    transcribe_session_params_init(&session_params);
    session_params.n_threads = 6;

    std::printf("Carregando modelo...\n");
    struct transcribe_session* session = nullptr;
    check(transcribe_open(MODEL_PATH, &load_params, &session_params, &session),
          "transcribe_open");

    std::printf("Carregando CAM++ (speaker embedding)...\n");
    const bool speaker_ok = g_speaker.init(SPEAKER_MODEL_PATH);
    if (!speaker_ok) {
        std::fprintf(stderr, "Seguindo sem speaker embedding.\n");
    }

    std::printf("Conectando ao orchestrator (/voice/ingest + /execute)...\n");
    g_orchestrator.init();

    // Rota /speaker/embed (cadastro on-demand p/ /settings da TUI) — sem
    // CAM++ o servidor sobe mesmo assim e responde 503 nas tentativas.
    g_embed_server.init(&g_speaker);

    std::printf("Carregando wake word (sherpa-onnx KWS)...\n");
    const bool wake_ok = g_wake.init();
    if (!wake_ok) {
        std::fprintf(stderr, "Wake word indisponível (modelo/keywords). Seguindo sem gating.\n");
    }
    const bool gate = WAKE_GATE_ASR && wake_ok;   // sem KWS, cai pro comportamento antigo

    ten_vad_handle_t vad = nullptr;
    if (ten_vad_create(&vad, FRAME_SAMPLES, VAD_THRESHOLD) != 0) {
        std::fprintf(stderr, "ten_vad_create falhou\n");
        return 1;
    }

    // ---- Áudio (miniaudio: captura do microfone padrão, s16le mono 16k) ----
    static ma_context g_ma_ctx;
    static ma_device  g_ma_dev;
    ma_result mr = ma_context_init(nullptr, 0, nullptr, &g_ma_ctx);
    if (mr != MA_SUCCESS) {
        std::fprintf(stderr, "ma_context_init: %s\n", ma_result_description(mr));
        return 1;
    }

    // Enumera p/ mostrar o nome do dispositivo padrão de captura (índice 0).
    ma_device_info* pCaptures = nullptr;
    ma_uint32 captureCount = 0;
    const char* dev_name = "padrão do sistema";
    if (ma_context_get_devices(&g_ma_ctx, nullptr, nullptr,
                               &pCaptures, &captureCount) == MA_SUCCESS &&
        captureCount > 0) {
        dev_name = pCaptures[0].name;
    }

    ma_device_config cfg = ma_device_config_init(ma_device_type_capture);
    cfg.capture.format       = ma_format_s16;
    cfg.capture.channels     = 1;
    cfg.sampleRate           = SAMPLE_RATE;
    cfg.periodSizeInFrames   = FRAME_SAMPLES;   // 256 frames = 16ms por callback
    cfg.periods              = 2;               // poucos períodos = baixa latência
    cfg.dataCallback         = audio_data_callback;
    // device.id zerado = microfone padrão do sistema.

    mr = ma_device_init(&g_ma_ctx, &cfg, &g_ma_dev);
    if (mr != MA_SUCCESS) {
        std::fprintf(stderr, "ma_device_init: %s\n", ma_result_description(mr));
        ma_context_uninit(&g_ma_ctx);
        return 1;
    }
    mr = ma_device_start(&g_ma_dev);
    if (mr != MA_SUCCESS) {
        std::fprintf(stderr, "ma_device_start: %s\n", ma_result_description(mr));
        ma_device_uninit(&g_ma_dev);
        ma_context_uninit(&g_ma_ctx);
        return 1;
    }

    std::printf("Microfone: %s\n", dev_name);
    std::printf("Escutando... fale algo (Ctrl+C pra sair)\n\n");

    // ---- Loop principal: VAD + transcrição ----
    std::vector<float> f32;
    std::deque<std::vector<int16_t>> preroll;
    std::vector<int16_t> frame(FRAME_SAMPLES);

    // Áudio bruto da frase atual: alimenta o speaker embedding e, no modo gate,
    // é o que se reenvia ao ASR quando a wake word dispara.
    std::vector<int16_t> utt_pcm;
    utt_pcm.reserve((size_t)SAMPLE_RATE * 60 / 1000);   // pré-aloca ~60s; cresce além disso se a frase for maior (sem limite)

    bool active = false;      // segmento de fala aberto (VAD)
    bool asr_open = false;    // transcribe_stream_begin já foi chamado nesta frase
    bool woke = false;        // wake word já disparou nesta frase
    std::string wake_name;
    int speech_run = 0;
    int silence_ms = 0;
    int utterance_ms = 0;
    int speech_frames = 0;
    float max_prob = 0.0f;

    // Abre o ASR e reenvia tudo que já foi capturado da frase (inclui a wake word).
    auto open_asr_replay = [&]() {
        begin_utterance(session);
        asr_open = true;
        for (size_t off = 0; off + FRAME_SAMPLES <= utt_pcm.size(); off += FRAME_SAMPLES) {
            to_f32(utt_pcm.data() + off, f32);
            asr_feed(session, f32.data(), false);
        }
        print_snapshot(session, false);
    };

    auto on_wake = [&]() {
        woke = true;
        {
            std::lock_guard<std::mutex> lk(g_print_mu);
            std::printf("\r\033[K\033[1;32m[wake word: %s]\033[0m\n", wake_name.c_str());
            std::fflush(stdout);
        }
        // Abre a janela do tray na tela de chat já no disparo da wake word
        // (fire-and-forget, thread destacada — nunca bloqueia o pipeline).
        g_orchestrator.notify_ui_chat_open();
        if (!asr_open) open_asr_replay();
    };

    // Um frame de uma frase aberta: converte UMA vez e distribui.
    auto process_frame = [&](const int16_t* pcm) {
        utt_pcm.insert(utt_pcm.end(), pcm, pcm + FRAME_SAMPLES);
        to_f32(pcm, f32);
        if (asr_open) asr_feed(session, f32.data(), true);
        // depois do disparo o KWS para de ser alimentado (economiza CPU e evita repetição)
        if (wake_ok && !woke && g_wake.feed(f32.data(), FRAME_SAMPLES, wake_name)) on_wake();
    };

    while (g_running.load()) {
        if (!g_ring.pop(frame.data(), FRAME_SAMPLES)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }

        float vad_prob = 0.0f;
        int vad_flag = 0;
        if (ten_vad_process(vad, frame.data(), FRAME_SAMPLES, &vad_prob, &vad_flag) != 0) {
            vad_flag = 0;
        }
        const bool is_speech = vad_flag == 1;

        if (!active) {
            preroll.push_back(frame);
            while ((int)preroll.size() > PREROLL_FRAMES) preroll.pop_front();

            speech_run = is_speech ? speech_run + 1 : 0;
            if (speech_run >= ONSET_FRAMES) {
                utt_pcm.clear();
                asr_open = false;
                woke = false;
                if (wake_ok) g_wake.begin();
                if (!gate) { begin_utterance(session); asr_open = true; }   // modo antigo: ASR abre já no onset

                utterance_ms = (int)preroll.size() * FRAME_MS;
                for (auto& f : preroll) process_frame(f.data());
                preroll.clear();
                active = true;
                silence_ms = 0;
                speech_frames = ONSET_FRAMES;
                max_prob = vad_prob;
            }
        } else {
            process_frame(frame.data());
            utterance_ms += FRAME_MS;
            silence_ms = is_speech ? 0 : silence_ms + FRAME_MS;
            if (is_speech) speech_frames++;
            max_prob = std::max(max_prob, vad_prob);

            // após a wake word, toleramos uma pausa maior (usuário ainda está falando o comando)
            const int silence_limit = woke ? SILENCE_MS_AFTER_WAKE : SILENCE_MS_TO_ENDPOINT;
            if (silence_ms >= silence_limit) {
                // última chance da wake word (tail padding do KWS)
                if (wake_ok && !woke && g_wake.flush(wake_name)) on_wake();

                if (asr_open) {
                    std::string utt_text;
                    const bool printed = finalize_utterance(session, &utt_text);
                    if (printed) {
                        // frase virou texto: fecha o turno de voz (embedding +
                        // /execute; sem CAM++, só /execute). Frases curtas e
                        // triviais são descartadas dentro do submit_turn.
                        submit_turn(utt_text, utt_pcm, silence_ms, speaker_ok);
                    } else if (DEBUG_EMPTY) {
                        std::lock_guard<std::mutex> lk(g_print_mu);
                        std::printf("\033[2m[descartado: %d ms, %d frames de fala, prob. max %.2f]\033[0m\n",
                                    utterance_ms, speech_frames, max_prob);
                        std::fflush(stdout);
                    }
                } else if (debug_wake()) {
                    std::lock_guard<std::mutex> lk(g_print_mu);
                    std::printf("\033[2m[sem wake word: %d ms, prob. max %.2f, identificado: \"%s\"]\033[0m\n",
                                utterance_ms, max_prob, g_wake.last_unmatched().c_str());
                    std::fflush(stdout);
                }
                if (wake_ok) g_wake.end();
                asr_open = false;
                active = false;
                speech_run = 0;
            }
        }
    }

    // ---- Encerramento ----
    ma_device_uninit(&g_ma_dev);
    ma_context_uninit(&g_ma_ctx);

    if (active) {
        if (wake_ok && !woke && g_wake.flush(wake_name)) on_wake();
        if (asr_open) {
            std::string utt_text;
            if (finalize_utterance(session, &utt_text))
                submit_turn(utt_text, utt_pcm, silence_ms, speaker_ok);
        }
    }
    if (wake_ok) g_wake.shutdown();
    g_embed_server.shutdown();              // para de aceitar antes de drenar o speaker
    if (speaker_ok) g_speaker.shutdown();   // drena a fila antes de sair
    g_orchestrator.shutdown();              // drena ingestões/turnos pendentes

    if (g_dropped.load() > 0) {
        std::fprintf(stderr, "Aviso: %lu blocos de áudio descartados (inferência lenta demais?)\n",
                     g_dropped.load());
    }

    ten_vad_destroy(&vad);
    transcribe_session_free(session);   // libera sessão E modelo (veio de transcribe_open)
    return 0;
}