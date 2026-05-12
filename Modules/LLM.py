import sys
import os
from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import json
import psutil
import socket
import datetime
from threading import Event
import threading
import tkinter as tk
from tkinter import messagebox
from pathlib import Path
from langdetect import detect
import re


stop_event = Event()

# pasta do script atual
BASEFOLDER = Path(__file__).parent.parent


def errorpop(msg):
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror("Erro", msg)
    root.destroy()

try:

    # Limpeza de Cache
    caminho_pasta = BASEFOLDER / "resource/voices"
    arquivos = os.listdir(caminho_pasta)
    for arquivo in arquivos:
        caminho_arquivo = os.path.join(caminho_pasta, arquivo)
        if os.path.isfile(caminho_arquivo):
            os.remove(caminho_arquivo)

    log_dir = BASEFOLDER / "logs"
    log_filename = f"LLM_log_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    log_path = os.path.join(log_dir, log_filename)

    voiceCom = False

    class LogDuplicado:
        def __init__(self, terminal, caminho_log):
            self.terminal = terminal
            self.log = open(caminho_log, "w", encoding="utf-8")

        def write(self, mensagem):
            try:
                self.terminal.write(mensagem)
            except Exception:
                pass
            self.log.write(mensagem)

        def flush(self):
            try:
                self.terminal.flush()
            except Exception:
                pass
            self.log.flush()

        def isatty(self):
            return False


    # Por isso:
    sys.stdout = LogDuplicado(sys.__stdout__, log_path)
    sys.stderr = LogDuplicado(sys.__stderr__, log_path)



    # ─────────────────────────────────────────────────────────────
    #                        FUNÇÕES UTILITÁRIAS
    # ─────────────────────────────────────────────────────────────

    def is_app_running(process_name: str) -> bool:
        for proc in psutil.process_iter(['name']):
            if proc.info['name'] and process_name.lower() in proc.info['name'].lower():
                return True
        return False


    def startConnection_close():
        HOST = '127.0.0.1'
        PORT = 5500
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)
        print("\n[INFO] Waiting for connection (close)...")
        conn, addr = server_socket.accept()
        print(f"\n[INFO] Client connected: {addr}")
        return conn


    def startConnection():
        HOST = '127.0.0.1'
        PORT = 5050
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)
        print("\n[INFO] Waiting for connection...")
        conn, addr = server_socket.accept()
        print(f"\n[INFO] Client connected: {addr}")
        return conn


    def tem_conexao():
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except OSError:
            return False


    def break_listener(conn_close, conn, stop_event):
        """Listen for break signal on the close connection and set stop_event."""
        while True:
            try:
                data = conn_close.recv(4096)
                if not data:
                    break
                message = data.decode("utf-8").strip()
                if message == "<break>":
                    # Send interruption message to the main connection
                    try:
                        conn.sendall("\n\n   Geração interrompida.".encode("utf-8"))
                    except:
                        pass
                    stop_event.set()
            except Exception as e:
                print(f"[ERROR] Break listener: {e}")
                break


    def buscar_arquivo(nome_arquivo, pasta_raiz):
        for raiz, _, arquivos in os.walk(pasta_raiz):
            if nome_arquivo in arquivos:
                pathModel = os.path.join(raiz, nome_arquivo)
                print(f"Caminho do arquivo encontrado: {pathModel}")
                return True, pathModel
        return False


    def salvar_chat_history(chat_history, caminho=BASEFOLDER / "chat_history.json"):
        with open(caminho, "w", encoding="utf-8") as f:
            json.dump(chat_history, f, ensure_ascii=False, indent=2)


    def carregar_chat_history(caminho=BASEFOLDER / "chat_history.json"):
        try:
            with open(caminho, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                else:
                    return [data]
        except FileNotFoundError:
            return []


    def listen_for_break(conn_close, conn, stop_event):
        """Listen for break signal on the close connection and set stop_event."""
        while not stop_event.is_set():
            try:
                data = conn_close.recv(4096)
                if not data:
                    break
                message = data.decode("utf-8").strip()
                if message == "<break>":
                    # Send interruption message to the main connection
                    conn.sendall("\n\n   Geração interrompida.".encode("utf-8"))
                    stop_event.set()
                    break
            except Exception as e:
                print(f"[ERROR] Break listener: {e}")
                break


    # ─────────────────────────────────────────────────────────────
    #                     BUSCA WEB (DuckDuckGo)
    # ─────────────────────────────────────────────────────────────

    # Compila padrões uma vez — reuse em todas as chamadas sem recompilar
    _PATTERNS_SEARCH = [
        r'\bhoje\b|\bagora\b|\batual(mente)?\b',
        r'\beste\s*(semana|mês|ano)\b|\boutem\b|\bamanhã\b',
        r'\bnotícias?\b|\bnovidade[ds]?\b|\beventos?\b|\burgent',
        r'\bpreço[s]?\b|\bcotação\b|\bdólar\b|\beuro\b|\bbitcoin\b',
        r'\bbolsa\b|\binflação\b|\bjuros\b|\bações\b',
        r'\bplacar\b|\bresultado[s]?\b|\bjogo[s]?\b|\bpartida[s]?\b',
        r'\bgol[s]?\b|\bcampeonato\b|\btorneio\b|\bcopa\b',
        r'\btempo\b|\bclima\b|\bprevisão\b|\btemperatura\b|\bchuva\b',
        r'\blançamento[s]?\b|\bnovo\b|\bnovos?\b|\bestreiou?\b|\bdisponível\b|\batualizado\b',
        r'\bquem\s+(é|foi|ganhou|perdeu|venceu)\b',
        r'\bpresidente\b|\bgoverno\b|\bfaleceu\b|\bmorreu\b|\bnasceu\b',
        r'\bquando\b|\bque\s+horas\b|\bquanto\s+tempo\b',
        r'\binteligência\s+artificial\b|\bia\b|\bllm\b|\bgpt\b|\bgemini\b',
        r'\bfilme[s]?\b|\bsérie[s]?\b|\bstreaming\b|\bnetflix\b',
        r'\blei\b|\bdecreto\b|\bimposto\b|\bprazo\b',
    ]
    _COMPILED_PATTERNS = [re.compile(p) for p in _PATTERNS_SEARCH]


    def precisa_buscar(user_input: str) -> bool:
        """Heurística com padrões regex — verifica rede inline a cada chamada."""

        # Rede verificada em tempo real (não rely em netCon estático)
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=2)
        except OSError:
            return False

        entrada = user_input.lower()

        for pattern in _COMPILED_PATTERNS:
            if pattern.search(entrada):
                print(f"[SEARCH] Pattern matched: {pattern.pattern}")
                return True
        return False

    def buscar_web(query: str) -> str:  
        pass

        



    # ─────────────────────────────────────────────────────────────
    #                        PROMPT
    # ─────────────────────────────────────────────────────────────

    chat_history = carregar_chat_history()

    def build_messages(system_content: str, user_input: str, max_turns: int = 10) -> list:
        messages = [{"role": "system", "content": system_content}]

        history_slice = chat_history[-(max_turns * 2):]
        for turn in history_slice:
            if isinstance(turn, dict) and "role" in turn and "content" in turn:
                content = turn["content"].strip()[:500]
                messages.append({"role": turn["role"], "content": content})

        # Injeta /no_think na mensagem do usuário, não no system
        final_input = user_input

        messages.append({"role": "user", "content": final_input})
        return messages


    def build_prompt(user_message: str, max_turns: int = 3) -> str:
        """Usado apenas para contexto de busca web e RAG, não para o LLM principal."""
        prompt = ""
        history_slice = chat_history[-(max_turns * 2):]

        if len(chat_history) > max_turns * 2:
            prompt += "[Previous conversation omitted]\n"

        for turn in history_slice:
            if isinstance(turn, dict) and "role" in turn and "content" in turn:
                role    = "User" if turn["role"] == "user" else "AI"
                content = turn["content"].strip()[:150]
                prompt += f"{role}: {content}\n"

        prompt += f"User: {user_message.strip()}\n"
        return prompt

    # ─────────────────────────────────────────────────────────────
    #                     LEITURA DE CONFIGS
    # ─────────────────────────────────────────────────────────────

    with open(BASEFOLDER / R"resource\VoiceModel.dll", "r", encoding="utf-8") as VoiceModelfile:
        voiceModel = VoiceModelfile.read()
        if voiceModel == "Voz padrão":
            voiceModel = "sage"

    with open(BASEFOLDER / R"resource\username.dll", "r", encoding="utf-8") as file:
        username = file.read()

    with open(BASEFOLDER / "resource/AiConfig.dll", "r", encoding="utf-8") as file2:
        model = file2.read().strip()
        model = model.replace("on-", "")
        model = model.strip()

    with open(BASEFOLDER / fR"resource/ctxConfig.dll", "r", encoding="utf-8") as ctxFile:
        ctxUsed = ctxFile.read()

    with open(BASEFOLDER / fR"Ctxbin\{ctxUsed}.bin", "r", encoding="utf-8") as contFile:
        context = contFile.read()

    with open(BASEFOLDER / f"resource/keyConfig.dll", "r", encoding="utf-8") as file3:
        keyConfig = file3.read()

    with open(BASEFOLDER / f"CfgModels/{model}.json", "r", encoding="utf-8") as f:
        MODELCFG = json.load(f)
        THREADS = int(MODELCFG["threads"])
        if MODELCFG["threads"] == "max":
            THREADS = max(4, os.cpu_count() - 2)

        KV_CACHE_QUANT = MODELCFG["kv_cache"]
            
        if KV_CACHE_QUANT != "q8_0" and KV_CACHE_QUANT != "q4_0":
            KV_CACHE_QUANT_CFG = "f16"

    with open(BASEFOLDER / f"resource\SearchCfg.dll", "r", encoding="utf-8") as searchFile:
        searchCfg = searchFile.read().strip()

    with open(BASEFOLDER / Rf"resource\agentCfg.json", "r", encoding="utf-8") as agentFile:
        agentCfg = json.load(agentFile)
    # ─────────────────────────────────────────────────────────────
    #                     CARREGAMENTO DO MODELO
    # ─────────────────────────────────────────────────────────────


    try:
        dirname = fR"C:\Users\{os.getlogin()}\.cache\huggingface\hub"
        filename = model
        tryDir, modelPath = buscar_arquivo(filename, dirname)

        try:
            print(f"\n[INFO] Carregando modelo: {model}")



        except Exception as e:
            print(f"\n[ERROR] Erro ao carregar o modelo: {str(e)}\n")



        # ─────────────────────────────────────────────────────
        #              LOOP PRINCIPAL — MODELO LOCAL
        # ─────────────────────────────────────────────────────

        try:
            while True:

                if "<voice>" in user_input:
                    voiceCom = True
                    user_input = user_input.replace("<voice>", "")


                # ── Busca web ──────────────────────────────────
                contexto_web = ""
                if precisa_buscar(user_input):
                    if searchCfg == "on":
                        print("\n[INFO] Gerando query de busca...")
                        contexto_web = buscar_web(user_input)
                        print(contexto_web)
                        if contexto_web:
                            print("[INFO] Contexto web injetado no prompt.")
                # ───────────────────────────────────────────────
                askLang = detect(user_input)
                system_final = f"{context}, my name's {username}, today's date is {datetime.datetime.now().strftime('%d/%m/%Y')} and respond in {askLang}"
                if contexto_web:
                    trechos = "\n\n---\n".join(contexto_web)
                    system_final += (
                        f"\n\n[RESULTADO DA SUA BUSCA WEB]\n"
                        f"Você realizou uma busca na internet e encontrou as seguintes informações:\n\n"
                        f"{trechos}\n\n"
                        f"Use essas informações para embasar sua resposta quando relevante. "
                        f"Não mencione que o usuário pesquisou — foi você que buscou autonomamente."
                    )

                print(f"\n[INPUT]: {user_input}\n")
                response_text = ""
                print(f"{system_final}\n\n {user_input}")



                print("{end}")

                chat_history.append({"role": "user",      "content": user_input})
                chat_history.append({"role": "assistant", "content": response_text})
                salvar_chat_history(chat_history)

        except Exception as e:
            print(f"\n[ERROR] {e}")
            sys.exit(1)

    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

except Exception as e:
    errorpop(f"Um erro crítico ocorreu: {e}")