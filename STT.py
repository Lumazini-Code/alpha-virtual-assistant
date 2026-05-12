import os
os.environ["OMP_NUM_THREADS"] = "10"
os.environ["OPENBLAS_NUM_THREADS"] = "10"
import speech_recognition as sr
import sys
import socket
import pygame
import psutil
import audioop
import numpy as np
import requests
import time
from pathlib import Path
from faster_whisper import WhisperModel, BatchedInferencePipeline


if getattr(sys, 'frozen', False):
    BASE_PATH = Path(sys._MEIPASS)
else:
    BASE_PATH = Path(__file__).parent

# força o caminho correto dos assets
os.environ["FasterWhisperAssets"] = str(BASE_PATH / "faster_whisper" / "assets")
LIMITE_SILENCIO = 200  # RMS minimo para considerar fala valida

print("[INFO] Carregando modelo Whisper...")
_model = WhisperModel(
    "medium",
    device="cpu",
    compute_type="int8",
    cpu_threads=10,
    num_workers=1,
)
model = BatchedInferencePipeline(model=_model)
print("[INFO] Modelo carregado.")


def is_app_running(process_name: str) -> bool:
    """Verifica se um processo esta em execucao."""
    try:
        for proc in psutil.process_iter(['name']):
            if proc.info['name'] and process_name.lower() in proc.info['name'].lower():
                return True
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return False


def check_internet() -> bool:
    """Verifica se ha conexao com a internet."""
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=2)
        return True
    except (OSError, socket.timeout):
        return False


def play_sound(caminho: str) -> bool:
    """Toca um arquivo de som. Retorna True se sucesso."""
    try:
        pygame.mixer.music.load(caminho)
        pygame.mixer.music.play()
        return True
    except pygame.error as e:
        print(f"[WARN] Erro ao tocar som: {e}")
        return False


def start_connection(timeout: float = 60.0):
    """Inicia o servidor socket e aguarda conexao."""
    HOST = '127.0.0.1'
    PORT = 5005
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.settimeout(timeout)
    try:
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)
        print("[INFO] Waiting for connection...")
        conn, addr = server_socket.accept()
        print(f"[INFO] Client connected: {addr}")
        return conn, server_socket
    except socket.timeout:
        print("[ERROR] Timeout esperando conexao")
        raise


# Configuracao AssemblyAI - adicionar sua API key aqui ou via variavel de ambiente
ASSEMBLYAI_API_KEY = os.environ.get("ASSEMBLYAI_API_KEY", "b1b4d20e1617403cac2f17d3bf88f3d9")
ASSEMBLYAI_BASE_URL = "https://api.assemblyai.com/v2"


def transcribe_with_assemblyai(audio_data: sr.AudioData) -> str | None:
    """
    Transcreve audio usando AssemblyAI API.
    Fluxo: upload -> criar transcript -> poll ate completar.
    Retorna o texto transcrito ou None em caso de falha.
    """

    headers = {"authorization": ASSEMBLYAI_API_KEY}
    t_total = time.time()
        
        # Tempo do upload
    t_upload = time.time()
    try:
        # 1. Upload do audio
        audio_bytes = audio_data.get_wav_data()
        upload_response = requests.post(
            f"{ASSEMBLYAI_BASE_URL}/upload",
            headers=headers,
            data=audio_bytes,
            timeout=30
        )
        upload_response.raise_for_status()
        upload_url = upload_response.json()["upload_url"]
        print(f"[TIMING] Upload: {time.time() - t_upload:.2f}s")
        print("[INFO] AssemblyAI: Audio enviado com sucesso")

        # 2. Criar request de transcript
        t_transcript = time.time()
        transcript_response = requests.post(
            f"{ASSEMBLYAI_BASE_URL}/transcript",
            json={
                "audio_url": upload_url,
                "language_code": "pt",
                "speech_models": ["universal-2"],  # ou "universal-3-pro"
                "punctuate": True,
                "format_text": True,
            },
            headers={
                **headers,
                "Content-Type": "application/json"  # <- garante o header correto
            },
            timeout=30
        )
        transcript_response.raise_for_status()
        transcript_id = transcript_response.json()["id"]
        print(f"[TIMING] Criar transcript: {time.time() - t_transcript:.2f}s")
        print(f"[INFO] AssemblyAI: Transcript criado - ID: {transcript_id}")

        # 3. Poll ate completar
        t_poll = time.time()
        polling_endpoint = f"{ASSEMBLYAI_BASE_URL}/transcript/{transcript_id}"
        max_attempts = 60  # Max 30 segundos (500ms * 60)
        attempts = 0

        while attempts < max_attempts:
            result = requests.get(polling_endpoint, headers=headers, timeout=10)
            result.raise_for_status()
            data = result.json()

            status = data.get("status")
            if status == "completed":
                texto = data.get("text", "").strip()
                if texto:
                    print(f"[TIMING] Polling até completar: {time.time() - t_poll:.2f}s ({attempts} tentativas)")
                    print(f"[TIMING] TOTAL: {time.time() - t_total:.2f}s")
                    print(f"[RESULT] {data.get('text', '')}")
                    print(f"[INFO] AssemblyAI reconhecido: {texto}")
                    return texto
                print("[WARN] AssemblyAI: Transcricao vazia")
                return None
            elif status == "error":
                error_msg = data.get("error", "Erro desconhecido")
                print(f"[ERROR] AssemblyAI: Erro na transcricao - {error_msg}")
                return None

            # Aguarda antes do proximo poll
            time.sleep(0.5)
            attempts += 1

        print("[WARN] AssemblyAI: Timeout aguardando transcricao")
        return None
    
    except requests.exceptions.HTTPError as e:
        print(f"[ERROR] HTTP {e.response.status_code}: {e.response.text}")
        return None
    
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] AssemblyAI: Erro na requisicao - {e}")
        return None




def transcribe_with_whisper(audio_data: sr.AudioData) -> str | None:
    """
    Transcreve audio usando faster-whisper (fallback).
    Retorna o texto transcrito ou None em caso de falha.
    """
    t_total = time.time()
    audio_bytes = audio_data.get_wav_data()

    # Deteccao de silencio
    rms = audioop.rms(audio_bytes, 2)
    print(f"[INFO] RMS: {rms}")
    if rms < LIMITE_SILENCIO:
        print("[INFO] Silencio detectado, ignorando.")
        return None

    try:
        audio_np = np.frombuffer(
            audio_data.get_raw_data(convert_rate=16000, convert_width=2),
            dtype=np.int16
        ).astype(np.float32) / 32768.0

        segments, _ = model.transcribe(
            audio_np,
            language="pt",
            beam_size=1,
            best_of=1,
            batch_size=16,
            vad_filter=True,
            vad_parameters=dict(
                threshold=0.3,
                min_silence_duration_ms=200,
                speech_pad_ms=50,
                min_speech_duration_ms=100,
                max_speech_duration_s=10,
            ),
            condition_on_previous_text=False,
            temperature=0.0,
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            without_timestamps=True,
            word_timestamps=False,
        )

        texto = " ".join(seg.text.strip() for seg in segments).strip()
        if texto:
            print(f"[INFO] Whisper reconhecido: {texto}")
            print(f"[TIMING] Whisper: {time.time() - t_total:.2f}s")
            return texto
    except Exception as e:
        print(f"[ERROR] Whisper: Erro na transcricao - {e}")
    return None


def recognize(audio_data: sr.AudioData, recognizer: sr.Recognizer, use_assemblyai: bool = True) -> str | None:
    """
    Transcreve audio usando AssemblyAI (se disponivel) ou Whisper como fallback.
    Retorna o texto transcrito ou None se ambos falharem.
    """
    # Tenta AssemblyAI primeiro se houver internet
    if use_assemblyai:
        resultado = transcribe_with_assemblyai(audio_data)
        if resultado:
            return resultado
        print("[INFO] Fallback para Whisper...")

    # Fallback para Whisper
    return transcribe_with_whisper(audio_data)


def main():
    """Funcao principal do modulo STT."""
    pygame.init()
    pygame.mixer.init()

    conn, server_socket = start_connection()

    # Inicializa o reconhecedor com configuracoes otimizadas
    microfone = sr.Recognizer()
    microfone.dynamic_energy_threshold = True
    microfone.dynamic_energy_adjustment_damping = 0.15
    microfone.dynamic_energy_ratio = 1.5
    microfone.pause_threshold = 0.8
    microfone.phrase_threshold = 0.1
    microfone.non_speaking_duration = 0.5
    microfone.operation_timeout = 5

    # Configura a fonte do microfone uma unica vez
    mic_source = sr.Microphone(sample_rate=16000)

    # Calibracao inicial
    with mic_source as source:
        print("[INFO] Calibrando microfone, fique em silencio...")
        microfone.adjust_for_ambient_noise(source, duration=2)
        print(f"[INFO] Threshold calibrado: {microfone.energy_threshold:.0f}")

    # Aquecimento do modelo Whisper
    print("[INFO] Aquecendo modelo...")
    try:
        dummy = np.zeros(16000, dtype=np.float32)
        list(model.transcribe(dummy, language="pt", beam_size=1)[0])
        print("[INFO] Modelo aquecido.")
    except Exception as e:
        print(f"[WARN] Falha no warmup do modelo: {e}")

    # Verifica conectividade inicial
    tem_internet = check_internet()
    if tem_internet:
        print("[INFO] Conexao com internet detectada - usando AssemblyAI como primario")
    else:
        print("[INFO] Sem conexao - usando Whisper como primario")

    # Caminhos dos sons
    som_abertura = str(BASE_PATH / "resource" / "Sounds" / "Mic_open.wav")
    som_fechamento = str(BASE_PATH / "resource" / "Sounds" / "Mic_close.wav")

    # Loop principal
    while True:
        # Verifica periodicamente se o app principal esta rodando
        if not is_app_running("Alpha AI Foreground.exe"):
            print("[INFO] Alpha AI Foreground encerrado - encerrando STT")
            break

        try:
            # Verifica conectividade a cada iteracao
            tem_internet = check_internet()

            # Recebe sinal do cliente
            conn.settimeout(1.0)  # Timeout curto para nao bloquear indefinidamente
            try:
                entrada = conn.recv(4096).decode("utf-8").strip().lower()
            except socket.timeout:
                continue
            finally:
                conn.settimeout(None)

            if entrada != "<listen>":
                print(f"[INFO] Chave incorreta: {entrada}")
                continue

            print("[INFO] Chave recebida, ouvindo...")

            # Toca som de abertura
            play_sound(som_abertura)

            # Captura audio
            with mic_source as source:
                try:
                    audio = microfone.listen(source, timeout=5, phrase_time_limit=15)
                except sr.WaitTimeoutError:
                    print("[WARN] Timeout: nenhuma fala detectada")
                    play_sound(som_fechamento)
                    conn.sendall("\n<mic Closed>".encode("utf-8"))
                    continue

            # Toca som de fechamento
            play_sound(som_fechamento)

            # Transcreve (AssemblyAI primario, Whisper fallback)
            comando = recognize(audio, microfone, use_assemblyai=tem_internet)

            if comando:
                print(f"[INFO] Reconhecido: {comando}")
                conn.sendall(f"{comando}\n<mic Closed>".encode("utf-8"))
            else:
                print("[WARN] Nada reconhecido.")
                conn.sendall("\n<mic Closed>".encode("utf-8"))

        except ConnectionResetError:
            print("[ERROR] Conexao resetada pelo cliente")
            break
        except ConnectionAbortedError:
            print("[ERROR] Conexao abortada pelo cliente")
            break
        except Exception as e:
            print(f"[ERROR] Inesperado: {e}")
            try:
                conn.sendall("\n<mic Closed>".encode("utf-8"))
            except:
                pass

    # Cleanup
    print("[INFO] Encerrando STT...")
    conn.close()
    server_socket.close()
    pygame.quit()


if __name__ == "__main__":
    main()
