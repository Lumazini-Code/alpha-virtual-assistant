import sounddevice as sd
import queue
import json
from vosk import Model, KaldiRecognizer
import requests
import socket
import emoji
import random
import pygame
import io
import audioop
import speech_recognition as sr
import datetime
import pywhatkit
import time
import webbrowser
import os
import json
import asyncio
import websockets
import json
import base64
import numpy as np
import sounddevice as sd

URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"

with open("resource/VoiceCtxCfg.dll", "r", encoding="utf-8") as VoiceFile:
    voiceCtxCfg = VoiceFile.read()

with open(fR"VoiceCtx/{voiceCtxCfg}.bin", "r", encoding="utf-8") as voiceCtxFile:
    voiceContext = voiceCtxFile.read()

with open(R"resource\VoiceModel.dll", "r", encoding="utf-8") as VoiceModelfile:
    voiceModel = VoiceModelfile.read()
    if voiceModel == "Voz padrão":
        voiceModel = "sage"

with open(R"resource\username.dll", "r", encoding="utf-8") as file:
    username = file.read()

with open("resource/AiConfig.dll", "r", encoding="utf-8") as file2:
    model = file2.read()
    model = model.replace("on-", "")

with open(fR"resource/ctxConfig.dll", "r", encoding="utf-8") as ctxFile:
    ctxUsed = ctxFile.read()

with open(fR"resource/Ctxbin\{ctxUsed}.bin", "r", encoding="utf-8") as contFile:
    context = contFile.read()

MODEL_PATH = R"resource\Vosk Model"
SAMPLE_RATE = 16000

q = queue.Queue()


def callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(bytes(indata))

def main():
    print("🔊 Carregando modelo...")
    model = Model(MODEL_PATH)
    rec = KaldiRecognizer(model, SAMPLE_RATE)

    print("🎤 Escutando...")
    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=8000, dtype='int16',
                           channels=1, callback=callback):
        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "").lower().strip()
                if text:
                    print("✅ Frase reconhecida:", text)

                    # Detecta frase de ativação
                    if "e alfa" in text or "ok alfa" in text or "ei alfa" in text:
                        print("⏸ Ativando assistente...")
                        asyncio.run(assistant())  # Espera o assistant terminar
                        print("🎤 Retomando escuta...")


            text = ""
            q.queue.clear()
async def response_def(comando):

    headers = [
        ("Authorization", f"Bearer {API_KEY}"),
        ("OpenAI-Beta", "realtime=v1")
    ]

    async with websockets.connect(URL, extra_headers=headers) as ws:
        print("🎤 Conectado ao GPT-4o Realtime Preview")

        # Cria a resposta
        await ws.send(json.dumps({
            "type": "response.create",
            "response": {
                "instructions": f"você acabou de executar a ação {comando}. Confirme ao usuário de uma forma divertida e curta oque você fez",
                "modalities": ["text", "audio"],
                "voice": "shimmer",
            }
        }))

        # Inicializa o stream de áudio (PCM16 mono, 24 kHz)
        stream = sd.OutputStream(samplerate=24000, channels=1, dtype='int16')
        stream.start()

        try:
            async for msg in ws:
                data = json.loads(msg)
                event = data.get("type", "")

                # Texto delta em tempo real
                if event == "response.output_text.delta":
                    print(data.get("delta", ""), end="", flush=True)

                # Áudio delta em tempo real
                elif event == "response.audio.delta":
                    chunk_b64 = data.get("delta", "")
                    if chunk_b64:
                        pcm_bytes = base64.b64decode(chunk_b64)
                        samples = np.frombuffer(pcm_bytes, dtype=np.int16).reshape(-1, 1)
                        stream.write(samples)

                # Resposta finalizada
                elif event == "response.completed":
                    print("\n✅ Resposta finalizada.")
                    break

        finally:
            stream.stop()
            stream.close()


def tem_conexao():
    try:
        # Tenta se conectar ao DNS do Google
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False

def pyaudio(caminho):
    pygame.init()
    pygame.mixer.music.load(caminho)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
        
def process_recog():
    microfone = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            microfone.adjust_for_ambient_noise(source)
            print("[INFO] Microphone listening...")
            pyaudio(R"resource\Sounds\Mic_open.wav")
            comando1 = microfone.listen(source, timeout=3)
            pyaudio(R"resource\Sounds\Mic_close.wav")



            try:
                comando = recognize(comando1)
                if comando == "":
                    raise sr.UnknownValueError()
                
                return comando

                

            except sr.UnknownValueError:
                print("[WARN] Nada reconhecido na fala.")

    except sr.WaitTimeoutError:
        print("[WARN] Timeout: ninguém falou nada.")
        pyaudio(R"resource\Sounds\Mic_close.wav")

    except Exception as mic_error:
        print(f"[ERROR] Falha ao acessar o microfone: {mic_error}")
        pyaudio(R"resource\Sounds\Mic_close.wav")


def recognize(command):

    audio_bytes = command.get_wav_data()

    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "voice.wav" 
    rms = audioop.rms(audio_bytes, 2)  # 2 = sample width (bytes)

    print(f"RMS do áudio: {rms}")

    # Define um limite mínimo (ajuste conforme seu microfone e ambiente)
    limite_silencio = 300  

    if rms < limite_silencio:
        print("Áudio silencioso detectado, ignorando reconhecimento.")
        raise sr.UnknownValueError()


    headers = {
        "Authorization": f"Bearer {API_KEY}",
    }

    files = {
        "file": audio_file,
    }

    data = {
        "model": "gpt-4o-mini-transcribe",
        "response_format": "text",
        "language": "pt"
    }

    response = requests.post("https://api.openai.com/v1/audio/transcriptions", headers=headers, files=files, data=data)
    return response.text


netCon = tem_conexao()

def openApp(approot, appname):
    appname = appname.replace('.lnk', '')
    print(approot)
    os.startfile(f'{approot}')
    iafala(f"abrindo, {appname}")




def iafala(comando1):
    
    if netCon == False:
        print("\n<no net>\n")
        return
    
    else:
        comando = emoji.replace_emoji(comando1, replace='')  # Remove emojis
        tokenList = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        audio1 = random.choices(tokenList, k=10)
        audioname = "".join(audio1)
        data = {
        "model": "gpt-4o-mini-tts",
        "input": comando,
        "voice": voiceModel,
        "instructions": voiceContext,
        "response_format": "wav",

        }

        # Cabeçalhos
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        urlVoice = 'https://api.openai.com/v1/audio/speech' 

        # Fazendo a requisição
        
        response = requests.post(urlVoice, headers=headers, data=json.dumps(data))
        
        # Salvando o áudio
        if response.status_code == 200:
            with open(fR"resource\voices\{audioname}.wav", "wb") as f:
                f.write(response.content)
        else:
            print("\n[ERROR] Failed to generate audio:", response.text)
        # Sua chave da API da OpenAI

        pyaudio(f"resource/Voices/{audioname}.wav")


async def assistant():
    matched = False 

    iafala("oi! como posso ajudar?")
    comando = process_recog()
    print(comando)
    try:
        if 'horas' in comando or 'horário' in comando:
            hora = datetime.datetime.now().strftime('%H:%M')
            iafala(f'{username} agora são hora {hora}')
            matched = True
            
            
        if 'toque' in comando or "tocar" in comando or 'toca' in comando:
            pywhatkit.playonyt(comando)
            await response_def("tocar uma musica")
            matched = True
            

        if 'piadas' in comando or 'piada' in comando:
            time.sleep(1)
            iafala(random.choice(piadas))
            matched = True
            

        if 'dia' in comando and 'hoje' in comando:
            hora = datetime.datetime.now().strftime(f'{username} hoje é dia %d de %m de %Y')
            iafala(hora)
            matched = True
            

        if 'navegador' in comando:
            webbrowser.open('https://www.google.com')
            await response_def(f'Abrir Navegador')
            matched = True
            

        if 'abrir youtube' in comando:
            webbrowser.open('https://www.youtube.com')
            await response_def(f'abrir youtube')
            matched = True
            


        if 'jogue' in comando and 'moeda' in comando:
            moeda = ['cara', 'coroa']
            iafala(f'{username}, o resultado foi. {random.choice(moeda)}')
            matched = True
            

        if 'Abrir Google Maps para' in comando or "abrir maps para" in comando:
            navegar = comando.replace('Abrir Google Maps para', '')
            navegar = navegar.replace('Abrir maps para', '')
            webbrowser.open(f'https://www.google.com.br/maps/search/{navegar}/')
            
            await response_def(f'abrindo google maps para {navegar}')
            matched = True
            

        if 'pesquise no youtube' in comando:
            search = comando.replace('pesquise no youtube ','')
            webbrowser.open(f"https://www.youtube.com/results?search_query={search}")
            await response_def(f'pesquisando no youtube: {search}')
            matched = True
            

        if 'abrir no youtube' in comando or 'abra no youtube' in comando:
            search = comando.replace('abrir no youtube ','')
            search = search.replace('abra no youtube ', '')
            pywhatkit.playonyt(search)
            await response_def(f'abrindo no youtube: {search}')
            matched = True
            

        if 'abrir' in comando:
            matched = True
            comando3 = comando.replace("abrir ", '')
            comando2 = comando3.title()
            comando2 += '.lnk'
            fileroot = R"C:\ProgramData\Microsoft\Windows\Start Menu\Programs"
            nome_do_arquivo = comando2


            if not os.path.exists(fileroot):
                iafala(f"eu não consegui encontrar o aplicativo {comando3},  você pode repetir?")
            else:
                # Varre a pasta
                arquivo_encontrado = False
                for root, dirs, files in os.walk(fileroot):
                    if nome_do_arquivo in files:
                        caminho_completo = os.path.join(root, nome_do_arquivo)
                        openApp(caminho_completo, comando2)
                        
                        arquivo_encontrado = True
                        await response_def(f'abrir o aplicativo: {comando3}')
                        break
                
                if not arquivo_encontrado:
                    iafala(f"eu não consegui encontrar o aplicativo {comando3},  você pode repetir?")
                # Se algo de errado acontecer:
                
        
    except Exception as e:
        print(f"[ERROR] Ocorreu um erro: {e}")
        iafala("desculpe, ocorreu um erro ao processar seu comando.")

    if not matched:

        headers = [
            ("Authorization", f"Bearer {API_KEY}"),
            ("OpenAI-Beta", "realtime=v1")
        ]

        async with websockets.connect(URL, extra_headers=headers) as ws:
            print("🎤 Conectado ao GPT-4o Realtime Preview")

            # Cria a resposta
            await ws.send(json.dumps({
                "type": "response.create",
                "response": {
                    "instructions": comando,
                    "modalities": ["text", "audio"],
                    "voice": "shimmer",
                }
            }))

            # Inicializa o stream de áudio (PCM16 mono, 24 kHz)
            stream = sd.OutputStream(samplerate=24000, channels=1, dtype='int16')
            stream.start()

            try:
                async for msg in ws:
                    data = json.loads(msg)
                    print(data)
                    event = data.get("type", "")


                    # Texto delta em tempo real
                    if event == "response.output_text.delta":
                        print(data.get("delta", ""), end="", flush=True)

                    # Áudio delta em tempo real
                    elif event == "response.audio.delta":
                        chunk_b64 = data.get("delta", "")
                        if chunk_b64:
                            pcm_bytes = base64.b64decode(chunk_b64)
                            samples = np.frombuffer(pcm_bytes, dtype=np.int16).reshape(-1, 1)
                            stream.write(samples)

                    # Resposta finalizada
                    elif event == "response.done":
                        print("\n✅ Resposta finalizada.")
                        await ws.close() 
                        break

            finally:
                stream.stop()
                stream.close()
                return


piadas = [
"Por que a aranha é o animal mais carente do mundo? Porque ela é um arac'needyou'.",
"Por que o pinheiro não se perde na floresta? Porque ele tem uma pinha.",
"Sabe como é a piada do pintinho caipira? Pir.",
"Um caipira chega na casa de um amigo que está vendo TV e pergunta: E aí, firme? O outro responde: Não, futebor",
"O que o pagodeiro foi fazer na igreja? Foi canta 'pá god'.",
"Por que Napoleão era chamado sempre para as festas na França? Porque ele era 'Bonapári'.",
"O que aconteceu com os lápis quando souberam que o dono da Faber Castell morreu? Ficaram desapontados",
"A plantinha foi ao hospital, mas não foi atendida. Por quê? Porque lá só tinha médico de 'plantão'.",
"Sabe qual o nome do site do cavalo? O www ponto cavalo ponto com ponto com ponto com ponto com ponto com'.",
"Você conhece a piadinha do ponêi? Pô nei eu",
"Qual é a fórmula da água benta? H Deus O.",
"Qual é o rei dos queijos? rei queijão.",
"O que o pato falou para a pata? Vem quá",
"Por que a velhinha não usa relógio? Porque ela é uma sem hora.",
"O que a vaca disse para o boi. Te amuuuuuuu.",
"havia dois caminhões voando. Um caiu. Por que o outro continuou voando? Porque era caminhão-pipa.",
"Por que a formiga tem quatro patas? Porque se tivesse cinco patas se chamaria fivemiga",
"Quando os americanos comeram carne pela primeira vez? Quando chegou Cristóvão Com Lombo",
"Por que as plantinhas não falam? Porque elas são mudinhas.",
"O que os estilistas fazem no tempo livre? Inventam moda.",
"Qual a cidade brasileira onde não tem táxis? Uberlândia.",
"Qual é o peixe que caiu do 10º andar? O aaaaaaatum.",
"Por que o jacaré tirou o filho da escola? Porque ele réptil de ano.",
"Para que servem os óculos vermelhos? Para vermelhor.",
"Como fazer um nó em duas motos? Pega as duas e Yamaha.",
"Sabe quais são as palavras que mais abrem portas na vida de todos nós? Puxe e empurre.",
"sabe qual o nome do filme de quando alguem te pede para extourar um balão?: Tó, estoure.",
"Qual a diferença de 14h para 2h? 12 horas.",
"Era uma vez um pintinho chamado Relam. Toda vez que chovia, Relam piava.",
"O que é um pontinho marrom no Brasil em 1500? Pedro Álvares Cabrown.",
"O que o sal disse para a batata? É nóis na frita!",

]

main()