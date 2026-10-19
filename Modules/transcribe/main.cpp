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
//   callback do PortAudio  ->  ring buffer lock-free  ->  thread principal
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
// Deps: PortAudio, TEN VAD (ten_vad.h + libten_vad), transcribe.cpp, ONNX Runtime,
//       sherpa-onnx (C API: sherpa-onnx/c-api/c-api.h)

#include <transcribe.h>
#include <parakeet.h>

#include <portaudio.h>
#include <ten_vad.h>
#include <onnxruntime_cxx_api.h>
#include <c-api.h>

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
#include <curl/curl.h>
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
// Ingestão de contexto de voz (Fase 2): envia o embedding do locutor ao
// orchestrator (POST /voice/ingest) numa thread dedicada — fire-and-forget,
// NUNCA bloqueia a thread do CAM++ nem a thread principal. A busca kNN roda
// fora do caminho do turno; o /execute só consome o cache do orchestrator.
// session_id=null: o sistema é single-session — o orchestrator cacheia como
// "__anon__" e o /execute consome essa entrada.
// ---------------------------------------------------------------------
class VoiceIngest {
public:
    bool init() {
        curl_global_init(CURL_GLOBAL_DEFAULT);
        const char* port = std::getenv("ORCHESTRATOR_PORT");
        url_ = std::string("http://127.0.0.1:")
             + (port && *port ? port : "9000") + "/voice/ingest";
        const char* tok = std::getenv("AVA_INTERNAL_TOKEN");
        if (tok && *tok) token_ = tok;
        worker_ = std::thread(&VoiceIngest::loop, this);
        return true;
    }

    void submit(std::vector<float>&& emb) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (queue_.size() >= QUEUE_MAX) queue_.pop_front();   // atrasou: joga a mais velha fora
            queue_.push_back(std::move(emb));
        }
        cv_.notify_one();
    }

    void shutdown() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        if (worker_.joinable()) worker_.join();
        if (curl_) curl_easy_cleanup(curl_);
        curl_global_cleanup();
    }

private:
    static size_t _discard(char*, size_t size, size_t nmemb) { return size * nmemb; }

    void loop() {
        for (;;) {
            std::vector<float> emb;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [this] { return stop_ || !queue_.empty(); });
                if (queue_.empty()) return;
                emb = std::move(queue_.front());
                queue_.pop_front();
            }
            post(emb);
        }
    }

    void post(const std::vector<float>& emb) {
        if (!curl_) curl_ = curl_easy_init();
        if (!curl_) return;

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

        curl_slist* hdrs = nullptr;
        hdrs = curl_slist_append(hdrs, "Content-Type: application/json");
        if (!token_.empty())
            hdrs = curl_slist_append(hdrs, ("Authorization: Bearer " + token_).c_str());

        curl_easy_reset(curl_);
        curl_easy_setopt(curl_, CURLOPT_URL, url_.c_str());
        curl_easy_setopt(curl_, CURLOPT_POST, 1L);
        curl_easy_setopt(curl_, CURLOPT_HTTPHEADER, hdrs);
        curl_easy_setopt(curl_, CURLOPT_POSTFIELDS, body.c_str());
        curl_easy_setopt(curl_, CURLOPT_POSTFIELDSIZE, (long)body.size());
        curl_easy_setopt(curl_, CURLOPT_TIMEOUT_MS, 3000L);
        curl_easy_setopt(curl_, CURLOPT_CONNECTTIMEOUT_MS, 1000L);
        curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION, _discard);

        const CURLcode rc = curl_easy_perform(curl_);
        curl_slist_free_all(hdrs);

        long code = 0;
        curl_easy_getinfo(curl_, CURLINFO_RESPONSE_CODE, &code);
        if (rc != CURLE_OK) {
            std::lock_guard<std::mutex> lk(g_print_mu);
            std::fprintf(stderr, "\033[2m[voice ingest: %s]\033[0m\n", curl_easy_strerror(rc));
        } else if (code != 200) {
            std::lock_guard<std::mutex> lk(g_print_mu);
            std::fprintf(stderr, "\033[2m[voice ingest: HTTP %ld]\033[0m\n", code);
        } else {
            std::lock_guard<std::mutex> lk(g_print_mu);
            std::printf("\033[2m[voice ingest: ok (%zu dims)]\033[0m\n", emb.size());
            std::fflush(stdout);
        }
    }

    static constexpr size_t QUEUE_MAX = 4;
    std::string url_;
    std::string token_;
    CURL* curl_ = nullptr;
    std::deque<std::vector<float>> queue_;
    std::mutex mu_;
    std::condition_variable cv_;
    bool stop_ = false;
    std::thread worker_;
};

static VoiceIngest g_voice_ingest;

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

            // Fase 2 — ingestão: dispara POST /voice/ingest com o embedding
            // (fire-and-forget na thread do VoiceIngest; não bloqueia o CAM++).
            g_voice_ingest.submit(std::vector<float>(emb, emb + dim));

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

    std::printf("Conectando ingestão de voz (orchestrator /voice/ingest)...\n");
    g_voice_ingest.init();

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
    Pa_StopStream(pa_stream);
    Pa_CloseStream(pa_stream);
    Pa_Terminate();

    if (active) {
        if (wake_ok && !woke && g_wake.flush(wake_name)) on_wake();
        if (asr_open && finalize_utterance(session) && speaker_ok) {
            submit_speaker(utt_pcm, silence_ms);
        }
    }
    if (wake_ok) g_wake.shutdown();
    if (speaker_ok) g_speaker.shutdown();   // drena a fila antes de sair
    g_voice_ingest.shutdown();              // drena ingestões pendentes

    if (g_dropped.load() > 0) {
        std::fprintf(stderr, "Aviso: %lu blocos de áudio descartados (inferência lenta demais?)\n",
                     g_dropped.load());
    }

    ten_vad_destroy(&vad);
    transcribe_session_free(session);   // libera sessão E modelo (veio de transcribe_open)
    return 0;
}