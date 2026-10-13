#!/usr/bin/env python3
"""
rebuild_indices.py — Reconstrói os índices vetoriais de TEXTO do LanceDB
depois de trocar o modelo de embedding (onnxManager).

Quando trocar?
  Você trocou o modelo via ONNX_EMBED_MODEL_PATH (ex.: e5-small →
  e5-base) e a dimensão mudou (384 → 768). Os vetores antigos ficam na
  tabela `vectors_384` e a busca passa a falhar/achar nada — os textos
  precisam ser re-embedados com o modelo NOVO.

O que NÃO mexe (dimensões de encoders próprios):
  - visual_dict (DINOv3, 384) — imagens, não texto
  - face_dict  (EdgeFace, 512)
  - voice_dict (CAM++,   192)

Uso (dentro do container, com o onnxManager no ar):
    docker exec ava python3 /app/scripts/rebuild_indices.py --confirm
    docker exec ava python3 /app/scripts/rebuild_indices.py \
        --confirm --drop-old

  --confirm    aplica de fato (sem ele, roda em dry-run)
  --drop-old   apaga a tabela antiga do LanceDB se sobrar vazia (só a
               porção de TEXTO é removida; visual_dict na mesma tabela
               384 continua intacta)
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
import urllib.request
import json
from pathlib import Path

import numpy as np
import pyarrow as pa

# ── Paths (mesmos de memory_config.py) ────────────────────────────────────────
DB_DIR    = Path(os.getenv("REBUILD_DB_DIR", "/app/Modules/memory/db"))
LANCE_DIR = DB_DIR / "ava_lance"
TABLE_PFX = "vectors"

ONNX_URL      = os.getenv("ONNX_SERVING_URL", "http://localhost:2002")
OLD_DIM       = int(os.getenv("REBUILD_OLD_DIM", "384"))
PASSAGE_PFX   = "passage: "
BATCH         = 64

# (memory_type, sqlite_path, tabela, coluna de texto)
SOURCES = [
    ("long_term",    "ava_memory.db",        "memories",            "text"),
    ("short_term",   "ava_short_term.db",    "turn_groups",         "embed_text"),
    ("tool",         "ava_tools.db",         "tool_vectors",        "text"),
    ("indexed_file", "ava_indexed_files.db", "indexed_file_chunks", "chunk_text"),
]


def embed(texts: list[str]) -> np.ndarray:
    """Chama /v1/embed (o servidor aplica o prefixo e-normalização)."""
    vecs: list[list[float]] = []
    for i in range(0, len(texts), BATCH):
        payload = json.dumps({"texts": texts[i:i + BATCH], "prefix": PASSAGE_PFX}).encode()
        req = urllib.request.Request(
            f"{ONNX_URL}/v1/embed", data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            vecs.extend(json.loads(r.read())["embeddings"])
    arr = np.asarray(vecs, dtype=np.float32)
    return arr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", action="store_true", help="aplica (default: dry-run)")
    ap.add_argument("--drop-old", action="store_true", help="remove a tabela antiga se ficar vazia")
    args = ap.parse_args()

    import lancedb

    db = lancedb.connect(str(LANCE_DIR))

    # Dimensão nova: lê da resposta do servidor (fonte da verdade)
    with urllib.request.urlopen(f"{ONNX_URL}/health", timeout=10) as r:
        health = json.loads(r.read())
    new_dim = int(os.getenv("REBUILD_NEW_DIM", "0")) or None

    # Dimensão real do modelo atual: 1 embed de teste
    probe = embed(["probe"])[0]
    new_dim = len(probe)
    print(f"onnxManager: {health['embed_model']} — dim nova={new_dim} (antiga={OLD_DIM})")

    if new_dim == OLD_DIM:
        print("Dimensão inalterada — nada a reconstruir. Ajuste REBUILD_OLD_DIM se necessário.")
        return 0

    if args.confirm:
        # Cria/atualiza a tabela nova com o schema correto
        schema = pa.schema([
            pa.field("memory_type", pa.string()),
            pa.field("id",          pa.int64()),
            pa.field("vector",      pa.list_(pa.float32(), new_dim)),
        ])
        db.create_table(f"{TABLE_PFX}_{new_dim}", schema=schema, exist_ok=True)

    table_old = db.open_table(f"{TABLE_PFX}_{OLD_DIM}")
    rows_moved = 0

    for mt, db_file, tbl, col in SOURCES:
        path = DB_DIR / db_file
        if not path.exists():
            print(f"[{mt}] {db_file} não existe — pulando")
            continue
        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(f"SELECT id, {col} AS text FROM {tbl}").fetchall()
        except sqlite3.OperationalError as e:
            print(f"[{mt}] tabela {tbl} indisponível ({e}) — pulando")
            continue
        finally:
            conn.close()

        # Só os ids que de fato têm vetor antigo (não re-embedar nada órfão).
        # Volume de vetores é pequeno (dezenas/centenas) — varredura é OK.
        ids_with_vec = {
            r["id"] for r in table_old.to_arrow().to_pylist()
            if r["memory_type"] == mt
        }
        todo = [(r["id"], r["text"]) for r in rows if r["id"] in ids_with_vec and r["text"]]

        if not todo:
            print(f"[{mt}] 0 vetores para reconstruir")
            continue

        print(f"[{mt}] re-embedando {len(todo)} registro(s)...")
        if not args.confirm:
            continue
        ids   = [i for i, _ in todo]
        texts = [t for _, t in todo]
        vecs  = embed(texts)
        data  = pa.table(
            {
                "memory_type": pa.array([mt] * len(ids), pa.string()),
                "id":          pa.array(ids, pa.int64()),
                "vector":      pa.FixedSizeListArray.from_arrays(
                                   pa.array(vecs.ravel(), pa.float32()), new_dim),
            },
            schema=schema,
        )
        db.open_table(f"{TABLE_PFX}_{new_dim}").add(data)

        # Remove a porção de texto da tabela antiga (visual_dict intacta)
        id_list = ",".join(str(i) for i in ids)
        table_old.delete(f"memory_type = '{mt}' AND id IN ({id_list})")
        rows_moved += len(ids)
        print(f"[{mt}] {len(ids)} vetor(es) re-embedados na dim {new_dim}")

    print(f"Total reconstruído: {rows_moved}")

    if args.confirm and args.drop_old:
        leftover = table_old.to_arrow().num_rows
        if leftover == 0:
            db.drop_table(f"{TABLE_PFX}_{OLD_DIM}")
            print(f"Tabela antiga vectors_{OLD_DIM} removida (vazia).")
        else:
            print(f"Tabela antiga mantida: ainda tem {leftover} linha(s) "
                  f"(visual_dict e outros encoders de imagem/áudio).")

    if not args.confirm:
        print("\nDRY-RUN — rode com --confirm para aplicar.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
