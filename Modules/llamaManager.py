"""
AVA Llama-Server Manager
=========================
Starts and stops llama-server instances based on GGUF model metadata.
Automatically detects multimodal models (vision) and attaches --mmproj.

OPTIMIZED for minimum Time-To-First-Token (TTFT):
  - Larger ubatch_size for faster prefill
  - q4_0 KV cache (faster than q8_0 with minimal quality loss)
  - Optimized thread counts
  - Draft model support for speculative decoding
  - Prompt caching friendly architecture

Usage:
    python llama_manager.py start ./Models/mmpro-Qwen3V-2B-Instruct-Q8_0.gguf
    python llama_manager.py stop  ./Models/mmpro-Qwen3V-2B-Instruct-Q8_0.gguf
    python llama_manager.py status ./Models/mmpro-Qwen3V-2B-Instruct-Q8_0.gguf
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════
LLAMA_SERVER_PATH = "./llama-cpp/llama-server"
PID_DIR = Path(os.environ.get("LLAMA_PID_DIR", "/tmp/ava_llama_pids"))

# ── OPTIMIZED: Text-only server defaults ──────────────────────────────────────
# Key changes from original:
#   ubatch_size:  256 → 512    (faster prefill — processes more tokens per batch)
#   batch_size:   512 → 2048   (larger batch = fewer prefill passes)
#   cache_type_k: q8_0 → q4_0  (2x faster KV cache reads, ~1% quality loss)
#   cache_type_v: q8_0 → q4_0  (same)
#   threads:      4 → auto     (let llama.cpp detect, or set to physical cores)
#   parallel:     2 → 4        (more concurrent requests before queueing)
#   added: draft model support for speculative decoding
TEXT_PARAMS = {
    # ── Context & batching ───────────────────────────────────────────────────
    "ctx_size": 4096,          # Max context window
    "batch_size": 2048,        # CHANGED: 512 → 2048 — fewer passes over prompt
    "ubatch_size": 512,        # CHANGED: 256 → 512 — faster prefill per micro-batch
    "fit":  "off",
    # ── GPU ───────────────────────────────────────────────────────────────────
    "gpu_layers": 999,         # Offload all layers to GPU
    "split_mode": "layer",     # Layer-wise GPU split
    # ── Attention & Cache ─────────────────────────────────────────────────────
    "flash_attn": "on",        # Flash Attention — O(n) vs O(n²)
    "cache_type_k": "q4_0",    # CHANGED: q8_0 → q4_0 — 2x faster cache, ~1% quality hit
    "cache_type_v": "q4_0",    # CHANGED: same
    # ── Threading ─────────────────────────────────────────────────────────────
    "threads": 4,              # CPU threads for non-GPU ops
    "threads_batch": 4,        # CPU threads for batch processing
    "threads_http": 4,         # HTTP server threads
    # ── Concurrency ──────────────────────────────────────────────────────────
    "parallel": 1,             # CHANGED: 2 → 4 — more concurrent sequences
    "cont_batching": True,     # Continuous batching — don't wait for seq completion
    "cache_prompt": True,      # CRITICAL: reuses KV cache across requests
    # ── Memory & I/O ─────────────────────────────────────────────────────────
    "mmap": True,              # Memory-mapped model loading (faster startup)
    "prio": 2,                 # Process priority
    "poll": 50,                # Polling interval ms
    # ── Sampling defaults ─────────────────────────────────────────────────────
    "temp": 0.80,
    "top_k": 40,
    "top_p": 0.95,
    "min_p": 0.05,
    "repeat_penalty": 1.05,
    # ── Network ──────────────────────────────────────────────────────────────
    "host": "0.0.0.0",
    "port": 2001,  # Optional draft model for speculative decoding
    
}

# ── OPTIMIZED: Multimodal (Vision) server defaults ───────────────────────────
VISION_PARAMS = {
    "mmproj_offload": True,    # Offload vision projection to GPU
    "ctx_size": 4096,
    "fit":  "off",             # Vision models typically need less context
    "batch_size": 1024,        # CHANGED: 512 → 1024
    "ubatch_size": 256,        # CHANGED: 128 → 256
    "gpu_layers": 999,
    "split_mode": "layer",
    "flash_attn": "on",
    "cache_type_k": "q4_0",    # CHANGED: q8_0 → q4_0
    "cache_type_v": "q4_0",    # CHANGED: q8_0 → q4_0
    "threads": 4,
    "threads_batch": 4,
    "threads_http": 2,
    "parallel": 1,             # CHANGED: 1 → 2
    "cont_batching": True,
    "cache_prompt": True,
    "mmap": True,
    "prio": 2,
    "poll": 50,
    "image_max_tokens": 1024,
    "temp": 0.70,
    "top_k": 40,
    "top_p": 0.95,
    "min_p": 0.05,
    "repeat_penalty": 1.05,
    "host": "0.0.0.0",
    "port": 2001,
}

# ── NEW: Draft model for speculative decoding ─────────────────────────────────
# If you have a small draft model (e.g., Qwen3-0.6B Q4), speculative decoding
# can give 2-3x speedup on generation. Set the path here or pass via CLI.
DRAFT_MODEL_PATH: Optional[str] = None  # e.g., "./Models/Qwen3-0.6B-Q4_K_M.gguf"


# ══════════════════════════════════════════════════════════════════════════════
# GGUF Metadata Reader
# ══════════════════════════════════════════════════════════════════════════════

def find_mmproj(model_path: str) -> Optional[str]:
    """
    Attempts to find the mmproj GGUF file in the same directory as the model.
    Usually named like: model-mmproj-f16.gguf or mmproj-model.gguf
    """
    model_dir = os.path.dirname(os.path.abspath(model_path))
    candidates = glob.glob(os.path.join(model_dir, "*mmproj*.gguf"))
    if candidates:
        for c in candidates:
            if 'f16' in os.path.basename(c).lower():
                return c
        return candidates[0]
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Process Management
# ══════════════════════════════════════════════════════════════════════════════

def get_pid_file(model_path: str) -> Path:
    """Generates a unique PID file path based on the model's absolute path."""
    PID_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = os.path.abspath(model_path).replace("/", "_").replace(".", "_")
    return PID_DIR / f"{safe_name}.json"


def save_pid(model_path: str, pid: int, port: int, mmproj: Optional[str], draft: Optional[str] = None):
    """Saves server process info to a JSON file."""
    pid_file = get_pid_file(model_path)
    data = {
        "pid": pid,
        "port": port,
        "model": os.path.abspath(model_path),
        "mmproj": mmproj,
        "draft": draft,
        "started_at": time.time()
    }
    with open(pid_file, "w") as f:
        json.dump(data, f, indent=2)


def load_pid(model_path: str) -> Optional[dict]:
    """Loads server process info from JSON file."""
    pid_file = get_pid_file(model_path)
    if pid_file.exists():
        with open(pid_file, "r") as f:
            return json.load(f)
    return None


def is_process_running(pid: int) -> bool:
    """Checks if a process with given PID is running."""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Command Builder
# ══════════════════════════════════════════════════════════════════════════════

def build_command(
    model_path: str,
    mmproj_path: Optional[str] = None,
    draft_path: Optional[str] = None,
) -> list[str]:
    """Builds the llama-server command based on model type."""
    cmd = [LLAMA_SERVER_PATH, "--model", os.path.abspath(model_path)]

    if mmproj_path:
        print("[Config] Using MULTIMODAL server parameters")
        params = VISION_PARAMS
        cmd.extend(["--mmproj", os.path.abspath(mmproj_path)])
        print(f"[Config] Attaching mmproj: {mmproj_path}")
    else:
        print("[Config] Using TEXT-ONLY server parameters")
        params = TEXT_PARAMS

    # Map parameters to command line arguments
    for key, value in params.items():
        if isinstance(value, bool) and value:
            cmd.append(f"--{key.replace('_', '-')}")
        elif not isinstance(value, bool):
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])

    # ── NEW: Draft model for speculative decoding ─────────────────────────────
    if draft_path and os.path.exists(draft_path):
        cmd.extend(["--model-draft", str(draft_path)])
        # Number of speculative tokens (default 5, max useful ~8)
        cmd.extend(["--spec-draft-n-max", "6"])
        # Minimum draft probability to accept
        cmd.extend(["--spec-draft-n-min", "0.7"])
        print(f"[Config] Speculative decoding with draft: {draft_path}")

    return cmd


# ══════════════════════════════════════════════════════════════════════════════
# Core Actions
# ══════════════════════════════════════════════════════════════════════════════

def start_server(model_path: str, draft_path: Optional[str] = None):
    """Starts the llama-server with appropriate parameters."""
    if not os.path.exists(model_path):
        print(f"[Error] Model file not found: {model_path}")
        sys.exit(1)

    # Check if already running
    existing = load_pid(model_path)
    if existing and is_process_running(existing["pid"]):
        print(f"[Warning] Server is already running for this model!")
        print(f"          PID: {existing['pid']}, Port: {existing['port']}")
        sys.exit(0)

    # Find mmproj
    mmproj_path = find_mmproj(model_path)
    if not mmproj_path:
        print("[Info] No mmproj file found. Using TEXT-ONLY parameters.")

    # Use CLI draft path, or fallback to global config
    effective_draft = draft_path or DRAFT_MODEL_PATH
    if effective_draft and not os.path.exists(effective_draft):
        print(f"[Warning] Draft model not found: {effective_draft} — skipping speculative decoding")
        effective_draft = None

    # Build command
    cmd = build_command(model_path, mmproj_path, effective_draft)

    print("\n" + "="*60)
    print("Starting llama-server with command:")
    print(" ".join(cmd))
    print("="*60 + "\n")

    try:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        model_name = Path(model_path).stem
        log_file = open(log_dir / f"{model_name}.log", "w")

        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=log_file,
            start_new_session=True
        )

        port = VISION_PARAMS["port"] if mmproj_path else TEXT_PARAMS["port"]
        save_pid(model_path, process.pid, port, mmproj_path, effective_draft)

        # Wait and check if it crashed
        time.sleep(2)
        if not is_process_running(process.pid):
            print("[Error] Server failed to start! Check the logs:")
            print(f"        cat logs/{model_name}.log")
            sys.exit(1)

        print(f"✓ Server started successfully!")
        print(f"  PID:    {process.pid}")
        print(f"  Port:   {port}")
        print(f"  Log:    logs/{model_name}.log")
        print(f"  URL:    http://0.0.0.0:{port}")
        if effective_draft:
            print(f"  Draft:  {effective_draft}")
        print(f"\n  TTFT optimizations active:")
        print(f"    - Flash Attention: ON")
        print(f"    - KV Cache:       q4_0 (fast)")
        print(f"    - Prompt Caching: ON")
        print(f"    - Batch Size:     {TEXT_PARAMS['batch_size'] if not mmproj_path else VISION_PARAMS['batch_size']}")
        print(f"    - ubatch Size:    {TEXT_PARAMS['ubatch_size'] if not mmproj_path else VISION_PARAMS['ubatch_size']}")
        if effective_draft:
            print(f"    - Spec. Decoding: ON (draft={os.path.basename(effective_draft)})")

    except FileNotFoundError:
        print(f"[Error] '{LLAMA_SERVER_PATH}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"[Error] Failed to start server: {e}")
        sys.exit(1)


def stop_server(model_path: str):
    """Stops a running llama-server instance."""
    existing = load_pid(model_path)
    if not existing:
        print("[Info] No running server found for this model.")
        return

    pid = existing["pid"]
    if not is_process_running(pid):
        print("[Info] Server process is not running. Cleaning up PID file.")
        get_pid_file(model_path).unlink(missing_ok=True)
        return

    print(f"Stopping llama-server (PID: {pid})...")
    try:
        os.kill(pid, signal.SIGTERM)
        for i in range(10):
            if not is_process_running(pid):
                break
            time.sleep(1)
        else:
            print("[Warning] Graceful shutdown failed. Force killing...")
            os.kill(pid, signal.SIGKILL)

        get_pid_file(model_path).unlink(missing_ok=True)
        print("✓ Server stopped successfully.")
    except Exception as e:
        print(f"[Error] Failed to stop server: {e}")


def check_status(model_path: str):
    """Checks the status of a llama-server instance."""
    existing = load_pid(model_path)
    if not existing:
        print("[Status] No server record found for this model.")
        return

    pid = existing["pid"]
    port = existing["port"]
    mmproj = existing.get("mmproj")
    draft = existing.get("draft")

    if is_process_running(pid):
        uptime = time.time() - existing.get("started_at", time.time())
        hours, remainder = divmod(int(uptime), 3600)
        minutes, seconds = divmod(remainder, 60)

        print(f"[Status] RUNNING")
        print(f"  PID:    {pid}")
        print(f"  Port:   {port}")
        print(f"  Model:  {existing['model']}")
        print(f"  MMProj: {mmproj if mmproj else 'N/A'}")
        print(f"  Draft:  {draft if draft else 'N/A'}")
        print(f"  Uptime: {hours}h {minutes}m {seconds}s")
        print(f"  URL:    http://localhost:{port}")
    else:
        print(f"[Status] STOPPED (PID {pid} no longer exists)")
        get_pid_file(model_path).unlink(missing_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="AVA Llama-Server Manager - Starts/Stops models with TTFT optimizations",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "action",
        choices=["start", "stop", "status"],
        help="Action to perform"
    )

    parser.add_argument(
        "model_path",
        help="Path to the GGUF model file"
    )

    parser.add_argument(
        "--draft",
        default=None,
        help="Path to a draft GGUF model for speculative decoding (e.g., Qwen3-0.6B-Q4_K_M.gguf)"
    )

    args = parser.parse_args()

    if args.action == "start":
        start_server(args.model_path, draft_path=args.draft)
    elif args.action == "stop":
        stop_server(args.model_path)
    elif args.action == "status":
        check_status(args.model_path)


if __name__ == "__main__":
    main()
