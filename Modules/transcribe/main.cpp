// Teste de transcrição streaming: nemotron-3.5-asr-streaming-0.6b (GGUF Q4_K_M)
// via C API nativa do transcribe.cpp, + embedding de locutor (CAM++ 3D-Speaker).
//
// O ASR só é ativado por palavra-chave (Porcupine, pt): "ei alfa", "ei, alta",
// "alfa" ou "alpha". Enquanto ninguém chama, só o Porcupine escuta e o áudio
// vai para um buffer circular; ao detectar, o ASR abre a frase já com o que
// foi falado logo depois da palavra e segue até o silêncio de fim de frase.
//
// Depois de chamar, fale no microfone e o texto aparece no terminal:
//   - texto já confirmado (committed) em branco
//   - texto provisório (tentative) em cinza, muda enquanto você fala
//   - ao detectar silêncio, a frase é finalizada e fica gravada na linha
//   - logo depois, o embedding do locutor daquela frase é extraído
//
// Arquitetura:
//   callback do PortAudio  ->  ring buffer lock-free  ->  thread principal
//                                                     ->  thread de speaker ID
//   thread principal: Porcupine (sempre) + buffer circular -> ASR (após a palavra-chave)
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
// Deps: PortAudio, TEN VAD (ten_vad.h + libten_vad), transcribe.cpp, ONNX Runtime,
//       Porcupine (pv_porcupine.h + libpv_porcupine)

#include <transcribe.h>
#include <parakeet.h>

#include <portaudio.h>
#include <ten_vad.h>
#include <onnxruntime_cxx_api.h>
#include <pv_porcupine.h>

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
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

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
static constexpr int ONSET_FRAMES = 4;                   // ~64ms de fala seguida pra abrir
static constexpr int PREROLL_FRAMES = 13;                // ~208ms guardados antes do onset
static constexpr int MAX_UTTERANCE_MS = 30000;           // força finalize em frases longas
static constexpr bool DEBUG_EMPTY = true;                // mostra frases descartadas (vazias) em cinza
static constexpr int ATT_CONTEXT_RIGHT = 3;              // menu do nemotron-3.5: {13, 6, 3, 0} => {1040,480,240,0}ms de lookahead

// ---- Config do speaker embedding ----
static constexpr int SPK_MIN_MS = 600;                   // abaixo disso o embedding não presta
static constexpr int SPK_MAX_MS = 6000;                  // recorte central pra frases longas
static constexpr int SPK_THREADS = 2;                    // threads intra-op do ONNX Runtime
static constexpr size_t SPK_QUEUE_MAX = 4;               // frases pendentes antes de descartar as velhas

// ---- Config do wake word (Porcupine) ----
static constexpr bool USE_WAKE_WORD = true;              // false = comportamento antigo (o VAD abre a frase sozinho)
static constexpr const char* PORCUPINE_ACCESS_KEY_ENV = "PICOVOICE_ACCESS_KEY";   // AccessKey lida do ambiente
static constexpr const char* PORCUPINE_MODEL_PATH = "../Models/porcupine_params_pt.pv";   // modelo de idioma (pt)
static constexpr const char* PORCUPINE_DEVICE = "best";

// Um .ppn por frase, treinado no Picovoice Console (idioma Português, plataforma Linux).
struct WakeKeyword { const char* label; const char* ppn; float sensitivity; };
static constexpr WakeKeyword WAKE_KEYWORDS[] = {
    {"ei alfa",  "../Models/wake/ei-alfa_pt_linux.ppn",  0.5f},
    {"ei, alta", "../Models/wake/ei-alta_pt_linux.ppn",  0.5f},
    {"alfa",     "../Models/wake/alfa_pt_linux.ppn",     0.5f},
    {"alpha",    "../Models/wake/alpha_pt_linux.ppn",    0.5f},
};
static constexpr int N_WAKE_KEYWORDS = (int)(sizeof(WAKE_KEYWORDS) / sizeof(WAKE_KEYWORDS[0]));

static constexpr int WAKE_BUFFER_MS = 2000;              // capacidade do buffer circular de áudio
static constexpr int WAKE_BACKTRACK_MS = 150;            // quanto do buffer vai pro ASR na detecção (o Porcupine
                                                         // dispara ~1-2 frames após o fim da palavra; suba p/ ~1500
                                                         // se quiser que a própria palavra vá junto pro ASR)
static constexpr int WAKE_SETTLE_MS = 300;               // após a detecção, ignora a cauda da palavra p/ o endpoint
static constexpr int WAKE_TIMEOUT_MS = 6000;             // chamou e não falou nada: volta a dormir
static constexpr int WAKE_BUFFER_FRAMES = WAKE_BUFFER_MS / FRAME_MS;
static constexpr int WAKE_BACKTRACK_FRAMES = WAKE_BACKTRACK_MS / FRAME_MS;
static_assert(WAKE_BACKTRACK_MS <= WAKE_BUFFER_MS, "WAKE_BACKTRACK_MS deve caber em WAKE_BUFFER_MS");

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

static int audio_callback(const void* input, void*, unsigned long frame_count,
                          const PaStreamCallbackTimeInfo*, PaStreamCallbackFlags,
                          void*) {
    if (input && !g_ring.push(static_cast<const int16_t*>(input), frame_count)) {
        g_dropped++;
    }
    return paContinue;
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

    void submit(std::vector<int16_t>&& pcm) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (queue_.size() >= SPK_QUEUE_MAX) queue_.pop_front();   // atrasou: joga a mais velha fora
            queue_.push_back(std::move(pcm));
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

private:
    void loop() {
        for (;;) {
            std::vector<int16_t> pcm;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [this] { return stop_ || !queue_.empty(); });
                if (queue_.empty()) return;            // stop_ e nada pendente
                pcm = std::move(queue_.front());
                queue_.pop_front();
            }
            process(pcm);
        }
    }

    void process(const std::vector<int16_t>& pcm) {
        const auto t0 = std::chrono::steady_clock::now();

        const int frames = fbank_.compute(pcm.data(), pcm.size(), feats_);
        if (frames < 10) return;

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

            // ---- ponto de extensão ----
            // Aqui você pluga o tratamento (enrollment, cosine, diarização...).
            // `emb` tem `dim` floats e é válido só até o fim deste escopo.
            last_embedding_.assign(emb, emb + dim);

            const double took_ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            const double audio_s = (double)pcm.size() / SAMPLE_RATE;

            std::lock_guard<std::mutex> lk(g_print_mu);
            std::printf("\033[2m[embedding gerado: %zu dims | %.2fs de fala | %.1f ms]\033[0m\n",
                        dim, audio_s, took_ms);
            std::fflush(stdout);
        } catch (const Ort::Exception& e) {
            std::lock_guard<std::mutex> lk(g_print_mu);
            std::fprintf(stderr, "\nspeaker embedding: %s\n", e.what());
        }
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

    std::deque<std::vector<int16_t>> queue_;
    std::mutex mu_;
    std::condition_variable cv_;
    bool stop_ = false;
    std::thread worker_;
};

static SpeakerEmbedder g_speaker;

// Corta o silêncio do endpoint, aplica o recorte central e manda pra fila.
static void submit_speaker(std::vector<int16_t>& utt, int silence_ms) {
    const size_t trim = (size_t)(silence_ms / FRAME_MS) * FRAME_SAMPLES;
    size_t n = utt.size() > trim ? utt.size() - trim : 0;
    if (n < (size_t)SPK_MIN_MS * SAMPLE_RATE / 1000) return;

    const size_t max_n = (size_t)SPK_MAX_MS * SAMPLE_RATE / 1000;
    size_t off = 0;
    if (n > max_n) { off = (n - max_n) / 2; n = max_n; }

    g_speaker.submit(std::vector<int16_t>(utt.begin() + off, utt.begin() + off + n));
}

// ---------------------------------------------------------------------
// Histórico circular de áudio (frames de FRAME_SAMPLES) — alimenta o ASR
// com o que foi dito logo antes/durante a detecção da palavra-chave
// ---------------------------------------------------------------------
class AudioHistory {
public:
    explicit AudioHistory(size_t frames)
        : cap_(frames), buf_(frames * FRAME_SAMPLES) {}

    void push(const int16_t* frame) {
        std::copy(frame, frame + FRAME_SAMPLES, buf_.begin() + head_ * FRAME_SAMPLES);
        head_ = (head_ + 1) % cap_;
        if (count_ < cap_) count_++;
    }

    // Chama fn(const int16_t* frame) nos últimos n frames, do mais antigo ao mais novo.
    template <class F>
    size_t last(size_t n, F&& fn) const {
        n = std::min(n, count_);
        size_t idx = (head_ + cap_ - n) % cap_;
        for (size_t i = 0; i < n; i++) {
            fn(&buf_[idx * FRAME_SAMPLES]);
            idx = (idx + 1) % cap_;
        }
        return n;
    }

private:
    size_t cap_;
    std::vector<int16_t> buf_;
    size_t head_ = 0, count_ = 0;
};

// ---------------------------------------------------------------------
// Wake word (Porcupine). O Porcupine consome frames de
// pv_porcupine_frame_length() amostras (512 @16k); acumulamos os frames de
// 256 do VAD até completar um.
// ---------------------------------------------------------------------
class WakeWord {
public:
    WakeWord() = default;
    WakeWord(const WakeWord&) = delete;
    WakeWord& operator=(const WakeWord&) = delete;
    ~WakeWord() { if (h_) pv_porcupine_delete(h_); }

    bool init(const char* access_key) {
        std::array<const char*, N_WAKE_KEYWORDS> paths{};
        std::array<float, N_WAKE_KEYWORDS> sens{};
        for (int i = 0; i < N_WAKE_KEYWORDS; i++) {
            paths[i] = WAKE_KEYWORDS[i].ppn;
            sens[i] = WAKE_KEYWORDS[i].sensitivity;
        }
        const pv_status_t st = pv_porcupine_init(access_key, PORCUPINE_MODEL_PATH, PORCUPINE_DEVICE,
                                                 N_WAKE_KEYWORDS, paths.data(), sens.data(), &h_);
        if (st != PV_STATUS_SUCCESS) {
            std::fprintf(stderr, "pv_porcupine_init: %s\n"
                                 "  (confira AccessKey, %s e os .ppn em WAKE_KEYWORDS)\n",
                         pv_status_to_string(st), PORCUPINE_MODEL_PATH);
            h_ = nullptr;
            return false;
        }
        if (pv_sample_rate() != SAMPLE_RATE) {
            std::fprintf(stderr, "Porcupine espera %d Hz, o app usa %d Hz\n", pv_sample_rate(), SAMPLE_RATE);
            return false;
        }
        frame_len_ = (size_t)pv_porcupine_frame_length();
        acc_.reserve(frame_len_);
        return true;
    }

    // Alimenta amostras (qualquer quantidade). Devolve o índice da palavra-chave
    // detectada em WAKE_KEYWORDS, ou -1.
    int process(const int16_t* pcm, size_t n) {
        int hit = -1;
        for (size_t i = 0; i < n; i++) {
            acc_.push_back(pcm[i]);
            if (acc_.size() < frame_len_) continue;
            int32_t kw = -1;
            const pv_status_t st = pv_porcupine_process(h_, acc_.data(), &kw);
            acc_.clear();
            if (st != PV_STATUS_SUCCESS) {
                if (!warned_) {
                    std::fprintf(stderr, "\npv_porcupine_process: %s\n", pv_status_to_string(st));
                    warned_ = true;
                }
            } else if (kw >= 0 && hit < 0) {
                hit = kw;
            }
        }
        return hit;
    }

private:
    pv_porcupine_t* h_ = nullptr;
    size_t frame_len_ = 0;
    std::vector<int16_t> acc_;
    bool warned_ = false;
};

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
static bool print_snapshot(struct transcribe_session* session, bool final_line) {
    struct transcribe_stream_text text;
    transcribe_stream_text_init(&text);
    if (transcribe_stream_get_text(session, &text) != TRANSCRIBE_OK) return false;

    const bool has_text = has_content(text.committed_text, text.committed_text_bytes) ||
                          has_content(text.tentative_text, text.tentative_text_bytes);

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

static void feed_frame(struct transcribe_session* session,
                       const int16_t* pcm16, std::vector<float>& f32) {
    f32.resize(FRAME_SAMPLES);
    for (int i = 0; i < FRAME_SAMPLES; i++) f32[i] = pcm16[i] / 32768.0f;

    struct transcribe_stream_update update;
    transcribe_stream_update_init(&update);
    transcribe_status s =
        transcribe_stream_feed(session, f32.data(), FRAME_SAMPLES, &update);

    if (s != TRANSCRIBE_OK) {
        std::fprintf(stderr, "\ntranscribe_stream_feed: %s\n",
                     transcribe_status_string(s));
    } else if (update.result_changed) {
        print_snapshot(session, false);
    }
}

static bool finalize_utterance(struct transcribe_session* session) {
    struct transcribe_stream_update update;
    transcribe_stream_update_init(&update);
    transcribe_status s = transcribe_stream_finalize(session, &update);
    if (s != TRANSCRIBE_OK) {
        std::fprintf(stderr, "\ntranscribe_stream_finalize: %s\n",
                     transcribe_status_string(s));
    }
    return print_snapshot(session, true);
}

// ---------------------------------------------------------------------
int main() {
    std::signal(SIGINT, on_sigint);

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

    ten_vad_handle_t vad = nullptr;
    if (ten_vad_create(&vad, FRAME_SAMPLES, VAD_THRESHOLD) != 0) {
        std::fprintf(stderr, "ten_vad_create falhou\n");
        return 1;
    }

    WakeWord wake;
    if (USE_WAKE_WORD) {
        const char* access_key = std::getenv(PORCUPINE_ACCESS_KEY_ENV);
        if (!access_key || !*access_key) {
            std::fprintf(stderr, "Defina %s com a AccessKey do Picovoice Console.\n",
                         PORCUPINE_ACCESS_KEY_ENV);
            return 1;
        }
        std::printf("Carregando Porcupine (wake word)...\n");
        if (!wake.init(access_key)) return 1;
    }

    // ---- Áudio ----
    PaError pe = Pa_Initialize();
    if (pe != paNoError) {
        std::fprintf(stderr, "Pa_Initialize: %s\n", Pa_GetErrorText(pe));
        return 1;
    }
    PaStreamParameters in_params{};
    in_params.device = Pa_GetDefaultInputDevice();
    if (in_params.device == paNoDevice) {
        std::fprintf(stderr, "Nenhum microfone padrão encontrado.\n");
        return 1;
    }
    const PaDeviceInfo* dev = Pa_GetDeviceInfo(in_params.device);
    in_params.channelCount = 1;
    in_params.sampleFormat = paInt16;
    in_params.suggestedLatency = dev->defaultLowInputLatency;

    PaStream* pa_stream = nullptr;
    pe = Pa_OpenStream(&pa_stream, &in_params, nullptr, SAMPLE_RATE, FRAME_SAMPLES,
                       paNoFlag, audio_callback, nullptr);
    if (pe != paNoError) {
        std::fprintf(stderr, "Pa_OpenStream: %s\n", Pa_GetErrorText(pe));
        return 1;
    }
    pe = Pa_StartStream(pa_stream);
    if (pe != paNoError) {
        std::fprintf(stderr, "Pa_StartStream: %s\n", Pa_GetErrorText(pe));
        return 1;
    }

    std::printf("Microfone: %s\n", dev->name);
    if (USE_WAKE_WORD) {
        std::printf("Aguardando palavra-chave:");
        for (int i = 0; i < N_WAKE_KEYWORDS; i++)
            std::printf("%s \"%s\"", i ? " |" : "", WAKE_KEYWORDS[i].label);
        std::printf("  (Ctrl+C pra sair)\n\n");
    } else {
        std::printf("Escutando... fale algo (Ctrl+C pra sair)\n\n");
    }

    // ---- Loop principal: VAD + transcrição ----
    std::vector<float> f32;
    std::deque<std::vector<int16_t>> preroll;                // preroll do VAD (só sem wake word)
    AudioHistory history(WAKE_BUFFER_FRAMES);                // buffer circular (com wake word)
    std::vector<int16_t> frame(FRAME_SAMPLES);

    // Áudio bruto da frase atual, alimentando o speaker embedding.
    std::vector<int16_t> utt_pcm;
    utt_pcm.reserve((size_t)SAMPLE_RATE * MAX_UTTERANCE_MS / 1000);

    bool active = false;
    int speech_run = 0;
    int silence_ms = 0;
    int utterance_ms = 0;
    int speech_frames = 0;
    float max_prob = 0.0f;
    int since_wake_ms = 0;          // tempo desde a detecção da palavra-chave
    bool cmd_started = false;       // já começou a falar o comando depois da palavra-chave?

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

        // O Porcupine vê TODO o áudio (também durante o ASR, pra manter o
        // estado contínuo); o histórico guarda os últimos WAKE_BUFFER_MS.
        int wake_kw = -1;
        if (USE_WAKE_WORD) {
            history.push(frame.data());
            wake_kw = wake.process(frame.data(), FRAME_SAMPLES);
        }

        if (!active) {
            if (USE_WAKE_WORD) {
                if (wake_kw >= 0) {
                    {
                        std::lock_guard<std::mutex> lk(g_print_mu);
                        std::printf("\r\033[K\033[1;32m[palavra-chave: %s]\033[0m\n",
                                    WAKE_KEYWORDS[wake_kw].label);
                        std::fflush(stdout);
                    }
                    begin_utterance(session);
                    utt_pcm.clear();
                    // Entrega ao ASR o que está no buffer (inclui o frame atual)
                    const size_t n = history.last(WAKE_BACKTRACK_FRAMES, [&](const int16_t* f) {
                        feed_frame(session, f, f32);
                        if (speaker_ok) utt_pcm.insert(utt_pcm.end(), f, f + FRAME_SAMPLES);
                    });
                    utterance_ms = (int)n * FRAME_MS;
                    since_wake_ms = 0;
                    cmd_started = false;
                    active = true;
                    speech_run = 0;
                    silence_ms = 0;
                    speech_frames = 0;
                    max_prob = vad_prob;
                }
            } else {
                preroll.push_back(frame);
                while ((int)preroll.size() > PREROLL_FRAMES) preroll.pop_front();

                speech_run = is_speech ? speech_run + 1 : 0;
                if (speech_run >= ONSET_FRAMES) {
                    begin_utterance(session);
                    utterance_ms = (int)preroll.size() * FRAME_MS;
                    utt_pcm.clear();
                    for (auto& f : preroll) {
                        feed_frame(session, f.data(), f32);
                        if (speaker_ok) utt_pcm.insert(utt_pcm.end(), f.begin(), f.end());
                    }
                    preroll.clear();
                    active = true;
                    silence_ms = 0;
                    speech_frames = ONSET_FRAMES;
                    max_prob = vad_prob;
                }
            }
        } else {
            feed_frame(session, frame.data(), f32);
            if (speaker_ok) utt_pcm.insert(utt_pcm.end(), frame.begin(), frame.end());
            utterance_ms += FRAME_MS;
            silence_ms = is_speech ? 0 : silence_ms + FRAME_MS;
            if (is_speech) speech_frames++;
            max_prob = std::max(max_prob, vad_prob);

            bool endpoint;
            if (USE_WAKE_WORD) {
                // Só conta fala depois da "acomodação" (cauda da palavra-chave) e
                // só fecha por silêncio depois que o comando começou. Se ninguém
                // fala nada, o timeout devolve o app pro modo de espera.
                since_wake_ms += FRAME_MS;
                if (since_wake_ms > WAKE_SETTLE_MS) {
                    speech_run = is_speech ? speech_run + 1 : 0;
                    if (speech_run >= ONSET_FRAMES) cmd_started = true;
                }
                endpoint = cmd_started ? silence_ms >= SILENCE_MS_TO_ENDPOINT
                                       : since_wake_ms >= WAKE_TIMEOUT_MS;
            } else {
                endpoint = silence_ms >= SILENCE_MS_TO_ENDPOINT;
            }

            if (endpoint || utterance_ms >= MAX_UTTERANCE_MS) {
                const bool printed = finalize_utterance(session);
                if (printed) {
                    // só vale a pena extrair embedding de frase que virou texto
                    if (speaker_ok) submit_speaker(utt_pcm, silence_ms);
                } else if (DEBUG_EMPTY) {
                    std::lock_guard<std::mutex> lk(g_print_mu);
                    std::printf("\033[2m[descartado: %d ms, %d frames de fala, prob. max %.2f]\033[0m\n",
                                utterance_ms, speech_frames, max_prob);
                    std::fflush(stdout);
                }
                active = false;
                speech_run = 0;
            }
        }
    }

    // ---- Encerramento ----
    Pa_StopStream(pa_stream);
    Pa_CloseStream(pa_stream);
    Pa_Terminate();

    if (active) {
        if (finalize_utterance(session) && speaker_ok) {
            submit_speaker(utt_pcm, silence_ms);
        }
    }
    if (speaker_ok) g_speaker.shutdown();   // drena a fila antes de sair

    if (g_dropped.load() > 0) {
        std::fprintf(stderr, "Aviso: %lu blocos de áudio descartados (inferência lenta demais?)\n",
                     g_dropped.load());
    }

    ten_vad_destroy(&vad);
    transcribe_session_free(session);   // libera sessão E modelo (veio de transcribe_open)
    return 0;
}