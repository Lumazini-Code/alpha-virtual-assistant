// Teste de transcrição streaming: nemotron-3.5-asr-streaming-0.6b (GGUF Q4_K_M)
// via C API nativa do transcribe.cpp.
//
// Fale no microfone e o texto aparece no terminal:
//   - texto já confirmado (committed) em branco
//   - texto provisório (tentative) em cinza, muda enquanto você fala
//   - ao detectar silêncio, a frase é finalizada e fica gravada na linha
//
// Arquitetura (diferente da versão anterior):
//   callback do PortAudio  ->  ring buffer lock-free  ->  thread principal
// O callback de áudio só copia amostras (tempo real, sem inferência dentro
// dele). VAD + transcribe_stream_* rodam todos na thread principal, então
// não há corrida entre feed/finalize/begin e nenhum áudio se perde entre
// uma frase e outra.
//
// A stream só é aberta quando o VAD detecta fala (com pré-roll de ~200ms
// pra não cortar o início da frase), e é finalizada após silêncio.
//
// Deps: PortAudio, TEN VAD (ten_vad.h + libten_vad), transcribe.cpp

#include <transcribe.h>
#include <parakeet.h>

#include <portaudio.h>
#include <ten_vad.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <thread>
#include <vector>

// ---- Config ----
static constexpr const char* MODEL_PATH =
    "../Models/nemotron-3.5-asr-streaming-0.6b-Q4_K_M.gguf";
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

static std::atomic<bool> g_running{true};
static void on_sigint(int) { g_running = false; }

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

    bool active = false;
    int speech_run = 0;
    int silence_ms = 0;
    int utterance_ms = 0;
    int speech_frames = 0;
    float max_prob = 0.0f;

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
                begin_utterance(session);
                utterance_ms = (int)preroll.size() * FRAME_MS;
                for (auto& f : preroll) feed_frame(session, f.data(), f32);
                preroll.clear();
                active = true;
                silence_ms = 0;
                speech_frames = ONSET_FRAMES;
                max_prob = vad_prob;
            }
        } else {
            feed_frame(session, frame.data(), f32);
            utterance_ms += FRAME_MS;
            silence_ms = is_speech ? 0 : silence_ms + FRAME_MS;
            if (is_speech) speech_frames++;
            max_prob = std::max(max_prob, vad_prob);

            if (silence_ms >= SILENCE_MS_TO_ENDPOINT || utterance_ms >= MAX_UTTERANCE_MS) {
                const bool printed = finalize_utterance(session);
                if (!printed && DEBUG_EMPTY) {
                    std::printf("\033[2m[descartado: %d ms, %d frames de fala, prob. max %.2f]\033[0m\n",
                                utterance_ms, speech_frames, max_prob);
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

    if (active) finalize_utterance(session);
    if (g_dropped.load() > 0) {
        std::fprintf(stderr, "Aviso: %lu blocos de áudio descartados (inferência lenta demais?)\n",
                     g_dropped.load());
    }

    ten_vad_destroy(&vad);
    transcribe_session_free(session);   // libera sessão E modelo (veio de transcribe_open)
    return 0;
}