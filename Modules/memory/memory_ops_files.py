"""
memory_ops_files.py — Handlers de arquivos indexados (local-scraping).

/indexed-file/write (chunking estrutural + embed em batches, reindexação
incremental por chunk), /read (3 modos: path exato, path+query por chunks,
busca global — híbrida vetor+BM25 com rerank cross-encoder opcional), /check,
/get, /chunk, /delete, /delete-by-path, /delete-many, /list e /reconcile.

Só usa os modelos já disponíveis no endpoint ONNX (embedding e cross-encoder);
BM25 vem do FTS5 do próprio SQLite.

Pontos de projeto:
  • Escrita "embed primeiro, troca depois": nada é apagado antes de o novo
    conteúdo estar chunkado e embedado. Falha após mutação → compensação
    (rollback best-effort + hashes zerados para forçar reindexação).
  • Lock por path (escritas/deleções do mesmo arquivo são serializadas).
  • Todo acesso ao SQLite passa por `_db()` (thread única dedicada; cai para
    execução inline se a conexão não aceitar uso multi-thread).
  • update_access em lote e sem perder exceções.
"""
from __future__ import annotations

import asyncio
import functools
import hashlib
import os
import re
import sqlite3
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from memory_config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBED_DIM,
    GENERAL_SCOPE,
    IF_EMBED_BATCH_SIZE,
    IF_MAX_CONTENT_SIZE,
    encode_ref,
    is_general_ref,
    log,
    ref_real_id,
)
from memory_models import (
    IndexedFileCheckResponse,
    IndexedFileEntry,
    IndexedFileReadRequest,
    IndexedFileReadResponse,
    IndexedFileWriteRequest,
    IndexedFileWriteResponse,
    MemoryToolError,
)
from memory_state import state
from memory_vector import _chunk_text

# ══════════════════════════════════════════════════════════════════════════════
# Configuração local
# ══════════════════════════════════════════════════════════════════════════════

INCREMENTAL_REINDEX = True     # reembeda só chunks que mudaram
STRUCTURED_CHUNKING = True     # corta em funções/classes/headings/parágrafos
HYBRID_SEARCH = True           # vetor + BM25 (FTS5) com fusão RRF
RERANK_ENABLED = True          # cross-encoder (/v1/score) sobre o pool final
RERANK_POOL = 40               # nº de chunks enviados ao cross-encoder
RERANK_TIMEOUT = 8.0           # s; estourou → mantém a ordem da fusão
RRF_K = 60
EMBED_CACHE_MAX = 20000        # ~30 MB (384 float32)
EMBED_CONCURRENCY = 3          # batches de embedding em paralelo
SQL_IN_BATCH = 500             # limite seguro p/ "IN (?,?,...)"
OVERLAP_DEDUP_RATIO = 0.5      # chunks do mesmo arquivo sobrepostos > 50% → descarta
MAX_CONTEXT_CHUNKS = 3

_FTS_TABLE = "indexed_file_chunks_fts"


# ══════════════════════════════════════════════════════════════════════════════
# Infra: executor de DB, locks, tarefas em background
# ══════════════════════════════════════════════════════════════════════════════

_DB_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="if-db")
_DB_MODE: dict[int, bool] = {}


async def _db(if_db, fn, *args, **kw):
    """Executa `fn` (acesso síncrono ao SQLite) na thread única de DB.

    Na primeira chamada por `if_db` faz uma sonda: se a conexão rejeitar uso
    em outra thread (check_same_thread=True), passa a executar inline no loop.
    """
    loop = asyncio.get_running_loop()
    call = functools.partial(fn, *args, **kw) if (args or kw) else fn

    mode = getattr(if_db, "_ava_db_thread_ok", None)
    if mode is None:
        mode = _DB_MODE.get(id(if_db))
    if mode is None:
        try:
            await loop.run_in_executor(_DB_EXECUTOR, if_db._conn.execute, "SELECT 1")
            mode = True
        except Exception as e:
            mode = False
            log.warning(f"IF-DB: conexão não aceita thread dedicada ({e}); usando execução inline")
        try:
            setattr(if_db, "_ava_db_thread_ok", mode)
        except Exception:
            _DB_MODE[id(if_db)] = mode

    if mode:
        return await loop.run_in_executor(_DB_EXECUTOR, call)
    return call()


class _KeyedLocks:
    """Locks por chave, removidos quando ninguém mais os usa."""

    def __init__(self):
        self._locks: dict[Any, list] = {}

    @asynccontextmanager
    async def hold(self, key):
        ent = self._locks.get(key)
        if ent is None:
            ent = [asyncio.Lock(), 0]
            self._locks[key] = ent
        ent[1] += 1
        try:
            async with ent[0]:
                yield
        finally:
            ent[1] -= 1
            if ent[1] == 0:
                self._locks.pop(key, None)


_PATH_LOCKS = _KeyedLocks()
_BG_TASKS: set[asyncio.Task] = set()


def _spawn(coro, what: str):
    """Fire-and-forget que não perde exceções nem deixa a task ser coletada."""
    task = asyncio.get_running_loop().create_task(coro)
    _BG_TASKS.add(task)

    def _done(t: asyncio.Task):
        _BG_TASKS.discard(t)
        if not t.cancelled() and t.exception() is not None:
            log.warning(f"IF background '{what}' falhou: {t.exception()}")

    task.add_done_callback(_done)


def _touch_sync(if_db, ids):
    for i in ids:
        if_db.update_access(i)


def _touch(if_db, ids):
    ids = list(dict.fromkeys(ids))
    if ids:
        _spawn(_db(if_db, _touch_sync, if_db, ids), "update_access")


# ══════════════════════════════════════════════════════════════════════════════
# Helpers de id / escopo
# ══════════════════════════════════════════════════════════════════════════════

def _out_ref(scope_tag: Optional[str], real_id: int) -> int:
    """Id de resposta: composta para o geral (scope_tag), crua para o ativo."""
    return encode_ref(scope_tag, real_id) if scope_tag is not None else real_id


def _general():
    return state.scopes.get(GENERAL_SCOPE)


def _scope_tag_of(if_db) -> Optional[str]:
    return GENERAL_SCOPE if if_db is not state.if_db else None


def _if_backend_for_file_id(file_id: int, *, allow_general_fallback: bool = True):
    """Resolve (if_db, if_index, row, out_id) a partir de uma id de arquivo.

    Id composta (offset +10^12) → escopo geral. Id crua → escopo ativo; só se
    `allow_general_fallback` e não achar, sonda o geral. Operações destrutivas
    devem passar allow_general_fallback=False (id crua nunca pode apagar um
    arquivo do geral por colisão de sequência)."""
    gen = _general()
    if is_general_ref(file_id):
        real = ref_real_id(file_id)
        if gen is None:
            return None
        row = gen.if_db.get_by_id(real)
        if row is None:
            return None
        return gen.if_db, gen.if_index, row, file_id
    row = state.if_db.get_by_id(file_id)
    if row is not None:
        return state.if_db, state.if_index, row, file_id
    if allow_general_fallback and gen is not None and gen.if_db is not state.if_db:
        row = gen.if_db.get_by_id(file_id)
        if row is not None:
            return gen.if_db, gen.if_index, row, encode_ref(GENERAL_SCOPE, file_id)
    return None


def _if_backend_for_chunk_id(chunk_id: int):
    """Mesma resolução para ids de CHUNK (sequência própria de cada escopo).
    Retorna (if_db, real_chunk_id) — contrato original preservado."""
    gen = _general()
    if is_general_ref(chunk_id):
        if gen is None:
            return None
        return gen.if_db, ref_real_id(chunk_id)
    return state.if_db, chunk_id


def _if_backend_for_path(file_path: str):
    """Resolve (if_db, if_index, row) por caminho — ativo primeiro, geral depois."""
    row = state.if_db.get_by_path(file_path)
    if row is not None:
        return state.if_db, state.if_index, row
    gen = _general()
    if gen is not None and gen.if_db is not state.if_db:
        row = gen.if_db.get_by_path(file_path)
        if row is not None:
            return gen.if_db, gen.if_index, row
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Helpers de SQL (centralizados; usam método do if_db quando existir)
# ══════════════════════════════════════════════════════════════════════════════

def _batched(seq, n=SQL_IN_BATCH):
    seq = list(seq)
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _chunks_by_ids(if_db, ids) -> list:
    out = []
    for part in _batched(ids):
        out.extend(if_db.get_chunks_by_ids(part))
    return out


def _files_by_ids(if_db, ids) -> list:
    fn = getattr(if_db, "get_files_by_ids", None)
    out = []
    for part in _batched(ids):
        if fn is not None:
            out.extend(fn(part))
        else:
            out.extend(if_db._conn.execute(
                f"SELECT * FROM indexed_files WHERE id IN ({','.join('?' * len(part))})", part
            ).fetchall())
    return out


def _delete_chunks_by_ids(if_db, ids):
    fn = getattr(if_db, "delete_chunks_by_ids", None)
    if fn is not None:
        fn(list(ids))
        return
    conn = if_db._conn
    for part in _batched(ids):
        conn.execute(f"DELETE FROM indexed_file_chunks WHERE id IN ({','.join('?' * len(part))})", part)
    conn.commit()


def _update_chunk_positions(if_db, updates):
    """updates: [(chunk_index, char_start, char_end, chunk_id)]"""
    fn = getattr(if_db, "update_chunk_positions", None)
    if fn is not None:
        fn(updates)
        return
    conn = if_db._conn
    conn.executemany(
        "UPDATE indexed_file_chunks SET chunk_index=?, char_start=?, char_end=? WHERE id=?", updates
    )
    conn.commit()


def _chunk_with_file(if_db, real_id: int):
    fn = getattr(if_db, "get_chunk_with_file", None)
    if fn is not None:
        return fn(real_id)
    return if_db._conn.execute(
        "SELECT c.*, f.file_path, f.file_hash, f.content "
        "FROM indexed_file_chunks c JOIN indexed_files f ON c.file_id = f.id WHERE c.id = ?",
        (real_id,),
    ).fetchone()


def _count_all_chunks(if_db) -> int:
    return int(if_db._conn.execute("SELECT COUNT(*) FROM indexed_file_chunks").fetchone()[0])


def _all_chunk_ids(if_db) -> set:
    return {r[0] for r in if_db._conn.execute("SELECT id FROM indexed_file_chunks")}


# ══════════════════════════════════════════════════════════════════════════════
# FTS5 (BM25) — tabela externa sincronizada por triggers
# ══════════════════════════════════════════════════════════════════════════════

_FTS_TRIGGERS = f"""
CREATE TRIGGER IF NOT EXISTS ifc_fts_ai AFTER INSERT ON indexed_file_chunks BEGIN
  INSERT INTO {_FTS_TABLE}(rowid, chunk_text) VALUES (new.id, new.chunk_text);
END;
CREATE TRIGGER IF NOT EXISTS ifc_fts_ad AFTER DELETE ON indexed_file_chunks BEGIN
  INSERT INTO {_FTS_TABLE}({_FTS_TABLE}, rowid, chunk_text) VALUES ('delete', old.id, old.chunk_text);
END;
CREATE TRIGGER IF NOT EXISTS ifc_fts_au AFTER UPDATE OF chunk_text ON indexed_file_chunks BEGIN
  INSERT INTO {_FTS_TABLE}({_FTS_TABLE}, rowid, chunk_text) VALUES ('delete', old.id, old.chunk_text);
  INSERT INTO {_FTS_TABLE}(rowid, chunk_text) VALUES (new.id, new.chunk_text);
END;
"""


def _ensure_fts_sync(if_db) -> bool:
    """Cria (uma vez) a tabela FTS5 + triggers. False → sem busca lexical."""
    flag = getattr(if_db, "_ava_fts_ok", None)
    if flag is not None:
        return flag
    ok = False
    try:
        conn = if_db._conn
        existed = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name = ?", (_FTS_TABLE,)
        ).fetchone() is not None
        if not existed:
            conn.execute(
                f"CREATE VIRTUAL TABLE {_FTS_TABLE} USING fts5("
                "chunk_text, content='indexed_file_chunks', content_rowid='id', "
                "tokenize='unicode61 remove_diacritics 2')"
            )
        conn.executescript(_FTS_TRIGGERS)
        if not existed:
            conn.execute(f"INSERT INTO {_FTS_TABLE}({_FTS_TABLE}) VALUES('rebuild')")
        conn.commit()
        ok = True
    except Exception as e:
        log.warning(f"FTS5 indisponível — busca híbrida desligada neste escopo: {e}")
    try:
        setattr(if_db, "_ava_fts_ok", ok)
    except Exception:
        pass
    return ok


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _identifier_like(tok: str) -> bool:
    return (
        "_" in tok
        or any(ch.isdigit() for ch in tok)
        or re.search(r"[a-z][A-Z]", tok) is not None
        or (tok.isupper() and len(tok) >= 3)
    )


def _fts_exprs(query: str) -> tuple[str, str]:
    """(expr_geral, expr_forte). Forte = só tokens que parecem identificadores
    (snake_case, camelCase, com dígito, SIGLA) — esses furam o gate de cosseno."""
    toks = []
    for t in _TOKEN_RE.findall(query):
        if len(t) >= 2 and t not in toks:
            toks.append(t)
    toks = toks[:16]
    general = " OR ".join(f'"{t}"' for t in toks)
    strong = " OR ".join(f'"{t}"' for t in toks if _identifier_like(t))
    return general, strong


def _fts_search_sync(if_db, expr: str, limit: int, file_id: Optional[int] = None) -> list:
    if not expr:
        return []
    T = _FTS_TABLE
    try:
        if file_id is None:
            rows = if_db._conn.execute(
                f"SELECT {T}.rowid, bm25({T}) AS s FROM {T} WHERE {T} MATCH ? ORDER BY s LIMIT ?",
                (expr, limit),
            ).fetchall()
        else:
            rows = if_db._conn.execute(
                f"SELECT {T}.rowid, bm25({T}) AS s FROM {T} "
                f"JOIN indexed_file_chunks c ON c.id = {T}.rowid "
                f"WHERE {T} MATCH ? AND c.file_id = ? ORDER BY s LIMIT ?",
                (expr, file_id, limit),
            ).fetchall()
        return [(int(r[0]), float(r[1])) for r in rows]
    except sqlite3.Error as e:
        log.warning(f"FTS query falhou ({e}) expr={expr[:80]!r}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# Chunking estrutural (delega o corte "fino" ao _chunk_text original)
# ══════════════════════════════════════════════════════════════════════════════

_CHUNK_KEYS: Optional[dict] = None

_PY_EXT = {"py", "pyi", "pyw"}
_MD_EXT = {"md", "markdown", "mdx"}
_PY_TOP = re.compile(r"^(?:async\s+def|def|class)\s")
_PY_INNER = re.compile(r"^(?: {4}|\t)(?:async\s+def|def|class|@)")
_C_TOP = re.compile(
    r"^(?:export\s+|default\s+|public\s+|private\s+|protected\s+|static\s+|abstract\s+|"
    r"async\s+|final\s+|unsafe\s+|pub(?:\([^)]*\))?\s+)*"
    r"(?:function|class|interface|enum|struct|impl|trait|fn|func|def|type|namespace|module|object)\b"
)
_C_EXT = {"js", "jsx", "ts", "tsx", "mjs", "cjs", "java", "go", "rs", "c", "h", "cpp", "hpp", "cc",
          "cs", "kt", "swift", "php", "scala", "dart"}
_PARA_RE = re.compile(r"\n[ \t]*\n+")


def _chunk_keys() -> dict:
    """Descobre (uma vez) quais chaves o _chunk_text usa p/ início/fim/índice."""
    global _CHUNK_KEYS
    if _CHUNK_KEYS is not None:
        return _CHUNK_KEYS
    keys: dict = {}
    try:
        probe = "palavra " * (CHUNK_SIZE // 4 + 80)
        chunks = _chunk_text(probe, CHUNK_SIZE, CHUNK_OVERLAP)
        if len(chunks) >= 2:
            c0 = chunks[0]
            for k in c0:
                kl = str(k).lower()
                if kl == "text":
                    continue
                if "start" in kl:
                    keys["start"] = k
                elif "end" in kl:
                    keys["end"] = k
                elif "index" in kl or kl in ("idx", "i", "n"):
                    keys["index"] = k
            if "start" in keys and "end" in keys:
                good = all(
                    probe[c[keys["start"]]:c[keys["end"]]].strip() == c["text"].strip() for c in chunks
                )
                if not good:
                    keys = {}
            else:
                keys = {}
    except Exception as e:
        log.warning(f"Não foi possível inferir o formato dos chunks ({e}); chunking estrutural/incremental desligados")
        keys = {}
    _CHUNK_KEYS = keys
    if not keys:
        log.info("Chunking estrutural e reindexação incremental desativados (formato de chunk não reconhecido)")
    return keys


def _line_offsets(lines: list[str]) -> list[int]:
    offs, p = [], 0
    for ln in lines:
        offs.append(p)
        p += len(ln)
    return offs


def _py_bounds(lines, offs, top_re) -> list[int]:
    out = []
    for i, ln in enumerate(lines):
        if not top_re.match(ln) and not (top_re is _PY_TOP and ln.startswith("@")):
            continue
        if i > 0 and lines[i - 1].startswith("@"):
            continue                      # decorador já abriu o segmento
        j = i
        while j > 0 and lines[j - 1].startswith("#"):
            j -= 1                        # comentários colados ao def pertencem a ele
        out.append(offs[j])
    return out


def _md_bounds(lines, offs) -> list[int]:
    out, in_fence = [], False
    for i, ln in enumerate(lines):
        s = ln.lstrip()
        if s.startswith("```") or s.startswith("~~~"):
            in_fence = not in_fence
            continue
        if not in_fence and re.match(r"#{1,6}\s", ln):
            out.append(offs[i])
    return out


def _boundaries(content: str, ext: str) -> list[int]:
    lines = content.splitlines(keepends=True)
    offs = _line_offsets(lines)
    b: set[int] = {0}
    if ext in _PY_EXT:
        b.update(_py_bounds(lines, offs, _PY_TOP))
    elif ext in _MD_EXT:
        b.update(_md_bounds(lines, offs))
    elif ext in _C_EXT:
        b.update(offs[i] for i, ln in enumerate(lines) if _C_TOP.match(ln))

    def segs():
        s = sorted(b)
        return list(zip(s, s[1:] + [len(content)]))

    # 2º nível: métodos dentro de classes Python gigantes
    if ext in _PY_EXT:
        for s, e in segs():
            if e - s > CHUNK_SIZE:
                for i, ln in enumerate(lines):
                    if s < offs[i] < e and _PY_INNER.match(ln) and not (i > 0 and lines[i - 1].lstrip().startswith("@")):
                        b.add(offs[i])
    # 3º nível: parágrafos dentro de qualquer segmento ainda grande
    for s, e in segs():
        if e - s > CHUNK_SIZE:
            for m in _PARA_RE.finditer(content, s, e):
                if s < m.end() < e:
                    b.add(m.end())
    return sorted(b)


def _structured_chunks(content: str, extension: Optional[str]):
    keys = _chunk_keys()
    if not keys or len(content) <= CHUNK_SIZE:
        return None
    ext = (extension or "").lstrip(".").lower()
    bounds = _boundaries(content, ext)
    if len(bounds) <= 1:
        return None
    atoms = list(zip(bounds, bounds[1:] + [len(content)]))

    groups, cur_s, cur_e = [], None, None
    for s, e in atoms:
        if cur_s is None:
            cur_s, cur_e = s, e
        elif e - cur_s <= CHUNK_SIZE:
            cur_e = e
        else:
            groups.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    groups.append((cur_s, cur_e))

    ks, ke, ki = keys["start"], keys["end"], keys.get("index")
    out = []
    for gs, ge in groups:
        text = content[gs:ge]
        if not text.strip():
            continue
        if len(text) > CHUNK_SIZE:
            sub = _chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        else:
            sub = _chunk_text(text, len(text) + 1, 0)
        for c in sub:
            c = dict(c)
            c[ks] += gs
            c[ke] += gs
            out.append(c)
    if not out:
        return None
    if ki is not None:
        for i, c in enumerate(out):
            c[ki] = i
    for c in out:   # validação: posições têm que reproduzir o texto
        if content[c[ks]:c[ke]].strip() != c["text"].strip():
            return None
    return out


def _chunk_content(content: str, extension: Optional[str]) -> list:
    if STRUCTURED_CHUNKING:
        try:
            out = _structured_chunks(content, extension)
            if out:
                return out
        except Exception as e:
            log.warning(f"Chunking estrutural falhou ({e}); usando chunking padrão")
    return _chunk_text(content, CHUNK_SIZE, CHUNK_OVERLAP)


# ══════════════════════════════════════════════════════════════════════════════
# Embeddings: cache por hash de chunk + batches concorrentes
# ══════════════════════════════════════════════════════════════════════════════

_EMB_CACHE: "OrderedDict[str, np.ndarray]" = OrderedDict()


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", "replace")).hexdigest()


def _cache_get(h: str):
    v = _EMB_CACHE.get(h)
    if v is not None:
        _EMB_CACHE.move_to_end(h)
    return v


def _cache_put(h: str, v: np.ndarray):
    _EMB_CACHE[h] = v
    _EMB_CACHE.move_to_end(h)
    while len(_EMB_CACHE) > EMBED_CACHE_MAX:
        _EMB_CACHE.popitem(last=False)


async def _embed_passages_cached(texts: list[str]) -> np.ndarray:
    """Embeda passagens reaproveitando o cache; levanta exceção se o backend falhar."""
    if not texts:
        return np.empty((0, EMBED_DIM), dtype=np.float32)
    hashes = [_sha1(t) for t in texts]
    found: dict[str, np.ndarray] = {}
    missing: dict[str, str] = {}
    for h, t in zip(hashes, texts):
        v = _cache_get(h)
        if v is not None:
            found[h] = v
        else:
            missing.setdefault(h, t)

    if missing:
        items = list(missing.items())
        sem = asyncio.Semaphore(EMBED_CONCURRENCY)

        async def one(batch):
            async with sem:
                return await state.embed_engine.embed_passages([t for _, t in batch])

        batches = [items[i:i + IF_EMBED_BATCH_SIZE] for i in range(0, len(items), IF_EMBED_BATCH_SIZE)]
        results = await asyncio.gather(*(one(b) for b in batches))
        for batch, embs in zip(batches, results):
            for (h, _), e in zip(batch, embs):
                v = np.asarray(e, dtype=np.float32)
                found[h] = v
                _cache_put(h, v)
    return np.vstack([found[h] for h in hashes]).astype(np.float32, copy=False)


# ══════════════════════════════════════════════════════════════════════════════
# Escrita
# ══════════════════════════════════════════════════════════════════════════════

class _CommitError(Exception):
    """Falha durante a mutação (já compensada)."""


@dataclass
class _Meta:
    file_path: str
    file_name: Optional[str]
    extension: Optional[str]
    file_hash: Optional[str]
    size: int
    modified: Any
    source: Any = None
    confidence: Any = None


def _meta_from_req(req, content) -> _Meta:
    return _Meta(
        file_path=req.file_path, file_name=req.file_name, extension=req.extension,
        file_hash=req.file_hash, size=req.size or len(content), modified=req.modified,
        source=req.source, confidence=req.confidence,
    )


def _meta_from_row(row, content, file_hash=None) -> _Meta:
    return _Meta(
        file_path=row["file_path"], file_name=row["file_name"], extension=row["extension"],
        file_hash=row["file_hash"] if file_hash is None else file_hash,
        size=row["size"] or len(content), modified=row["modified"],
        source=row["source"], confidence=row["confidence"],
    )


def _plan_incremental(old_rows, chunks):
    """Casa chunks novos com antigos por texto idêntico. None → não vale a pena."""
    keys = _chunk_keys()
    if not keys or not old_rows:
        return None
    pool: dict[str, list] = defaultdict(list)
    for r in sorted(old_rows, key=lambda r: (r["chunk_index"], r["id"])):
        pool[_sha1(r["chunk_text"])].append(r)
    kept, new_idx = [], []
    for i, c in enumerate(chunks):
        lst = pool.get(_sha1(c["text"]))
        if lst:
            kept.append((i, lst.pop(0)))
        else:
            new_idx.append(i)
    if not kept:
        return None
    removed = [r for lst in pool.values() for r in lst]
    return {"kept": kept, "new_idx": new_idx, "removed": removed}


async def _remove_vectors(if_index, ids):
    ids = set(ids)
    if ids:
        await asyncio.get_running_loop().run_in_executor(None, if_index.remove_ids, ids)


async def _rollback_file(if_db, if_index, fid, chunk_ids, is_new, content, content_hash_ok, meta):
    """Compensação best-effort após falha pós-mutação."""
    try:
        await _remove_vectors(if_index, chunk_ids)
    except Exception as e:
        log.warning(f"rollback: remover vetores falhou: {e}")
    if fid is None:
        return
    try:
        if is_new:
            await _db(if_db, if_db.delete_file, fid)
        else:
            await _db(if_db, if_db.delete_chunks_by_file, fid)
            # hashes zerados → próximo write nunca devolve "unchanged"
            await _db(if_db, if_db.update_file, file_id=fid, content=content, content_hash="",
                      file_hash="", size=meta.size, modified=meta.modified)
    except Exception as e:
        log.error(f"rollback: não foi possível restaurar consistência de file_id={fid}: {e}")


async def _commit_full(if_db, if_index, existing, meta: _Meta, content, content_hash, chunks, embs) -> int:
    loop = asyncio.get_running_loop()
    fid = existing["id"] if existing is not None else None
    is_new = existing is None

    # 1. vetores antigos (estado atual do DB, mesmo que parcial)
    if not is_new:
        old_ids = await _db(if_db, if_db.get_chunk_ids_by_file, fid)
        if old_ids:
            try:
                await _remove_vectors(if_index, old_ids)
            except Exception as e:
                raise _CommitError(f"remove_vectors_failed:{e}")

    # 2. DB (arquivo + chunks)
    def _db_part():
        f = fid
        if f is not None:
            if_db.delete_chunks_by_file(f)
            if_db.update_file(file_id=f, content=content, content_hash=content_hash,
                              file_hash=meta.file_hash, size=meta.size, modified=meta.modified)
        else:
            f = if_db.insert_file(
                file_path=meta.file_path, file_name=meta.file_name, extension=meta.extension,
                content=content, content_hash=content_hash, file_hash=meta.file_hash,
                size=meta.size, modified=meta.modified, source=meta.source, confidence=meta.confidence,
            )
        if_db.insert_chunks(f, chunks)
        rows = sorted(if_db.get_chunks_by_file(f), key=lambda r: (r["chunk_index"], r["id"]))
        return f, [r["id"] for r in rows]

    try:
        fid, chunk_ids = await _db(if_db, _db_part)
    except Exception as e:
        await _rollback_file(if_db, if_index, fid, [], is_new, content, False, meta)
        raise _CommitError(f"db_failed:{e}")

    if len(chunk_ids) != embs.shape[0]:
        await _rollback_file(if_db, if_index, fid, chunk_ids, is_new, content, False, meta)
        raise _CommitError(f"chunk_embedding_mismatch:{len(chunk_ids)}!={embs.shape[0]}")

    # 3. vetores
    try:
        await loop.run_in_executor(None, if_index.add_batch, embs, chunk_ids)
    except Exception as e:
        await _rollback_file(if_db, if_index, fid, chunk_ids, is_new, content, False, meta)
        raise _CommitError(f"index_failed:{e}")
    return fid


async def _commit_incremental(if_db, if_index, existing, meta: _Meta, content, content_hash,
                              chunks, plan, new_embs) -> int:
    """Mantém chunks/vetores inalterados; troca só o que mudou. Qualquer falha
    levanta exceção — o chamador cai para _commit_full (que lida com estado parcial)."""
    loop = asyncio.get_running_loop()
    fid = existing["id"]
    keys = _chunk_keys()
    ks, ke = keys["start"], keys["end"]
    kept, new_idx, removed = plan["kept"], plan["new_idx"], plan["removed"]
    removed_ids = [r["id"] for r in removed]
    kept_ids = {r["id"] for _, r in kept}

    await _remove_vectors(if_index, removed_ids)

    def _db_part():
        if_db.update_file(file_id=fid, content=content, content_hash=content_hash,
                          file_hash=meta.file_hash, size=meta.size, modified=meta.modified)
        if removed_ids:
            _delete_chunks_by_ids(if_db, removed_ids)
        if new_idx:
            if_db.insert_chunks(fid, [chunks[i] for i in new_idx])
        rows = if_db.get_chunks_by_file(fid)
        by_id = {r["id"]: r for r in rows}
        new_ids = sorted(i for i in by_id if i not in kept_ids)
        if len(new_ids) != len(new_idx):
            raise RuntimeError(f"id mismatch: {len(new_ids)} != {len(new_idx)}")
        for i, cid in zip(new_idx, new_ids):           # sanidade: ordem de inserção
            if by_id[cid]["chunk_text"] != chunks[i]["text"]:
                raise RuntimeError("chunk text mismatch after insert")
        id_new = dict(zip(new_idx, new_ids))
        id_kept = {ni: r["id"] for ni, r in kept}
        updates = [(i, c[ks], c[ke], id_kept.get(i, id_new.get(i))) for i, c in enumerate(chunks)]
        _update_chunk_positions(if_db, updates)
        return new_ids

    new_ids = await _db(if_db, _db_part)
    if new_idx:
        if new_embs.shape[0] != len(new_ids):
            raise RuntimeError("embedding count mismatch")
        await loop.run_in_executor(None, if_index.add_batch, new_embs, new_ids)
    return fid


async def indexed_file_write(req: IndexedFileWriteRequest):
    """
    Armazena o conteúdo COMPLETO de um arquivo indexado.

    Fluxo (nada é apagado antes de o novo conteúdo estar pronto):
      1. Localiza o dono pelo path (ativo → geral), sob lock do path.
      2. file_hash/content_hash iguais (e chunks presentes) → "unchanged".
      3. Chunking (estrutural quando possível) + embeddings (cache por chunk).
         Falha aqui NÃO altera nada.
      4. Reindexação incremental (só chunks novos/removidos) ou completa.
         Falha pós-mutação → rollback best-effort e hashes zerados.
    """
    content = req.content
    if not content or not content.strip():
        return IndexedFileWriteResponse(stored=False, reason="content_empty")
    if len(content) > IF_MAX_CONTENT_SIZE:
        return IndexedFileWriteResponse(
            stored=False, reason=f"content_too_large:{len(content)}>{IF_MAX_CONTENT_SIZE}",
        )
    content_hash = hashlib.sha256(content.encode("utf-8", "replace")).hexdigest()
    async with _PATH_LOCKS.hold(req.file_path):
        return await _write_locked(req, content, content_hash)


async def _write_locked(req, content: str, content_hash: str):
    loop = asyncio.get_running_loop()

    resolved = await _db(state.if_db, _if_backend_for_path, req.file_path)
    if resolved is not None:
        if_db, if_index, existing = resolved
    else:
        if_db, if_index, existing = state.if_db, state.if_index, None
    tag = _scope_tag_of(if_db)
    where = "geral" if tag is not None else "ativo"
    await _db(if_db, _ensure_fts_sync, if_db)

    meta = _meta_from_req(req, content)

    if existing is not None and not req.force_reindex:
        if existing["file_hash"] == req.file_hash and existing["content_hash"] == content_hash:
            n = await _db(if_db, if_db.count_chunks, existing["id"])
            if n > 0:
                log.info(f"Indexed file unchanged: {req.file_path} (hash_match=True)")
                return IndexedFileWriteResponse(
                    stored=False, reason="unchanged", file_id=_out_ref(tag, existing["id"]),
                    chunks_created=0, was_reindexed=False, hash_match=True,
                )
            log.warning(f"{req.file_path}: hashes iguais mas sem chunks — reindexando")

    # ── Chunking (CPU, fora do loop). Nada mutado até aqui. ──
    chunks = await loop.run_in_executor(None, _chunk_content, content, req.extension)
    if not chunks:
        return IndexedFileWriteResponse(stored=False, reason="chunking_failed")

    # ── Plano incremental ──
    plan = None
    if existing is not None and INCREMENTAL_REINDEX and not req.force_reindex:
        old_rows = await _db(if_db, if_db.get_chunks_by_file, existing["id"])
        plan = _plan_incremental(old_rows, chunks)

    texts_needed = [chunks[i]["text"] for i in plan["new_idx"]] if plan else [c["text"] for c in chunks]
    try:
        embs = await _embed_passages_cached(texts_needed)
    except Exception as e:
        log.error(f"Embedding failed for {req.file_path}: {e}")
        return IndexedFileWriteResponse(stored=False, reason=f"embedding_failed:{e}")

    # ── Mutação ──
    was_reindexed = existing is not None
    mode = "full"
    try:
        if plan:
            try:
                file_id = await _commit_incremental(if_db, if_index, existing, meta, content,
                                                    content_hash, chunks, plan, embs)
                mode = "incremental"
            except Exception as e:
                log.warning(f"{req.file_path}: reindex incremental falhou ({e}); refazendo por completo")
                try:
                    embs_all = await _embed_passages_cached([c["text"] for c in chunks])
                except Exception as e2:
                    await _rollback_file(if_db, if_index, existing["id"], [], False, content, False, meta)
                    return IndexedFileWriteResponse(stored=False, reason=f"embedding_failed:{e2}")
                file_id = await _commit_full(if_db, if_index, existing, meta, content,
                                             content_hash, chunks, embs_all)
        else:
            file_id = await _commit_full(if_db, if_index, existing, meta, content,
                                         content_hash, chunks, embs)
    except _CommitError as e:
        log.error(f"Write failed for {req.file_path}: {e}")
        return IndexedFileWriteResponse(stored=False, reason=str(e))

    hash_match = (
        existing is not None
        and existing["file_hash"] == req.file_hash
        and existing["content_hash"] == content_hash
    )
    extra = ""
    if mode == "incremental":
        extra = f" kept={len(plan['kept'])} new={len(plan['new_idx'])} removed={len(plan['removed'])}"
    log.info(
        f"Indexed file: {req.file_path} → file_id={_out_ref(tag, file_id)} [{where}] "
        f"chunks={len(chunks)} mode={mode}{extra} reindexed={was_reindexed} "
        f"hash_match={hash_match} content_size={len(content)}"
    )
    return IndexedFileWriteResponse(
        stored=True,
        reason="ok" if not was_reindexed else "reindexed",
        file_id=_out_ref(tag, file_id),
        chunks_created=len(chunks),
        was_reindexed=was_reindexed,
        hash_match=hash_match,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Leitura
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class _Cand:
    tag: Optional[str]
    cid: int
    cos: float = 0.0
    in_vec: bool = False
    bm_gen: Optional[float] = None
    bm_strong: Optional[float] = None
    rrf: float = 0.0

    @property
    def key(self):
        return (self.tag, self.cid)

    @property
    def match_type(self) -> str:
        lex = self.bm_gen is not None or self.bm_strong is not None
        if self.in_vec and lex:
            return "hybrid"
        return "semantic" if self.in_vec else "lexical"


_RERANKER = None
_RERANKER_FAILED = False


def _get_reranker():
    """state.reranker se existir; senão RerankerClient (onnx_client) criado sob demanda."""
    global _RERANKER, _RERANKER_FAILED
    r = getattr(state, "reranker", None)
    if r is not None:
        return r
    if _RERANKER is not None:
        return _RERANKER
    if _RERANKER_FAILED:
        return None
    try:
        try:
            from onnx_client import RerankerClient
        except ImportError:
            from Modules.onnx_client import RerankerClient
        url = os.environ.get("AVA_ONNX_BASE_URL")
        _RERANKER = RerankerClient(base_url=url) if url else RerankerClient()
    except Exception as e:
        _RERANKER_FAILED = True
        log.warning(f"Reranker indisponível ({e}); busca segue sem cross-encoder")
    return _RERANKER


async def _rerank_scores(query: str, texts: list[str]):
    rr = _get_reranker()
    if rr is None:
        return None
    try:
        scores = await asyncio.wait_for(rr.score(query, [t[:2500] for t in texts]), timeout=RERANK_TIMEOUT)
        arr = np.asarray(scores, dtype=np.float32)
        return arr if arr.shape == (len(texts),) else None
    except Exception as e:
        log.warning(f"Rerank falhou/expirou ({e}); mantendo ordem da fusão")
        return None


async def _gather(query, emb, backends, *, K, min_score, hybrid, restrict=None, file_id=None):
    """Candidatos em nível de chunk: vetor + (BM25) fundidos por RRF.

    • vetor respeita min_score; BM25 "geral" também (cosseno calculado via
      search_subset); BM25 "forte" (identificadores) fura o gate.
    Retorna (candidatos ordenados, saturado)."""
    loop = asyncio.get_running_loop()
    gen_expr, strong_expr = _fts_exprs(query) if hybrid else ("", "")
    cands: dict[tuple, _Cand] = {}
    saturated = False

    for if_db, idx, tag in backends:
        if restrict is not None:
            raw = await loop.run_in_executor(None, idx.search_subset, emb, restrict, K)
            saturated |= len(raw) >= K and K < len(restrict)
        else:
            raw = await loop.run_in_executor(None, idx.search, emb, K)
            saturated |= len(raw) >= K
        cos_map = {cid: float(s) for cid, s in raw}
        for cid, s in raw:
            if s >= min_score:
                cands[(tag, cid)] = _Cand(tag, cid, cos=float(s), in_vec=True)

        if not (hybrid and gen_expr and await _db(if_db, _ensure_fts_sync, if_db)):
            continue
        lex_gen = await _db(if_db, _fts_search_sync, if_db, gen_expr, K, file_id)
        lex_strong = await _db(if_db, _fts_search_sync, if_db, strong_expr, K, file_id) if strong_expr else []
        need = list({cid for cid, _ in lex_gen + lex_strong if cid not in cos_map})
        if need:
            extra = await loop.run_in_executor(None, idx.search_subset, emb, need, len(need))
            cos_map.update({cid: float(s) for cid, s in extra})

        for cid, bm in lex_gen:
            cos = cos_map.get(cid)
            if cos is None or cos < min_score:
                continue
            c = cands.setdefault((tag, cid), _Cand(tag, cid))
            c.cos, c.bm_gen = cos, bm
        for cid, bm in lex_strong:
            c = cands.setdefault((tag, cid), _Cand(tag, cid))
            c.cos, c.bm_strong = cos_map.get(cid, 0.0), bm

    allc = list(cands.values())
    lists = (
        sorted((c for c in allc if c.in_vec), key=lambda c: -c.cos),
        sorted((c for c in allc if c.bm_gen is not None), key=lambda c: c.bm_gen),
        sorted((c for c in allc if c.bm_strong is not None), key=lambda c: c.bm_strong),
    )
    for lst in lists:
        for r, c in enumerate(lst):
            c.rrf += 1.0 / (RRF_K + r + 1)
    allc.sort(key=lambda c: (-c.rrf, -c.cos))
    return allc, saturated


async def _rows_for(cands, dbs: dict):
    by_tag = defaultdict(list)
    for c in cands:
        by_tag[c.tag].append(c.cid)
    out = {}
    for tag, ids in by_tag.items():
        rows = await _db(dbs[tag], _chunks_by_ids, dbs[tag], ids)
        for r in rows:
            out[(tag, r["id"])] = r
    return out


async def _order_pool(query, pool, rows, rerank: bool):
    """Reordena o pool com o cross-encoder (se disponível). → [(cand, score)]"""
    pool = [c for c in pool if c.key in rows]
    if rerank and len(pool) > 1:
        scores = await _rerank_scores(query, [rows[c.key]["chunk_text"] for c in pool])
        if scores is not None:
            order = sorted(range(len(pool)), key=lambda i: (-float(scores[i]), -pool[i].rrf))
            return [(pool[i], float(scores[i])) for i in order]
    return [(c, c.cos) for c in pool]


def _wants_full_content(req, default_when_unset: bool) -> bool:
    """True só se include_full_content foi pedido; se o campo não veio explícito
    na requisição, usa `default_when_unset` (modo 3 → False)."""
    val = getattr(req, "include_full_content", default_when_unset)
    try:
        fields_set = req.model_fields_set      # pydantic v2
    except AttributeError:
        try:
            fields_set = req.__fields_set__    # pydantic v1
        except AttributeError:
            return bool(val)
    return bool(val) if "include_full_content" in fields_set else default_when_unset


def _entry(file_row, scope_tag, *, score, content="", chunk_row=None, chunk_text=None,
           char_start=None, char_end=None, chunk_id=None, match_type=None):
    kw = dict(
        file_id=_out_ref(scope_tag, file_row["id"]),
        file_path=file_row["file_path"], file_name=file_row["file_name"],
        extension=file_row["extension"], content=content,
        file_hash=file_row["file_hash"], content_hash=file_row["content_hash"],
        size=file_row["size"], modified=file_row["modified"],
        score=round(float(score), 4), confidence=round(file_row["confidence"], 4),
        created_at=file_row["created_at"], access_count=file_row["access_count"],
        source=file_row["source"], scope=scope_tag, chunk_text=None,
    )
    if chunk_row is not None:
        kw.update(
            chunk_text=chunk_text if chunk_text is not None else chunk_row["chunk_text"],
            chunk_index=chunk_row["chunk_index"],
            char_start=char_start if char_start is not None else chunk_row["char_start"],
            char_end=char_end if char_end is not None else chunk_row["char_end"],
        )
    if chunk_id is not None:
        kw["chunk_id"] = chunk_id
    if match_type:
        kw["match_type"] = match_type
    try:
        return IndexedFileEntry(**kw)
    except Exception:
        kw.pop("match_type", None)
        return IndexedFileEntry(**kw)


def _expand(content: str, chunks_sorted: list, row, n: int):
    """Expande o chunk com N vizinhos de cada lado → (texto, start, end)."""
    pos = next((i for i, c in enumerate(chunks_sorted) if c["id"] == row["id"]), None)
    if pos is None or n <= 0:
        return row["chunk_text"], row["char_start"], row["char_end"]
    lo, hi = max(0, pos - n), min(len(chunks_sorted) - 1, pos + n)
    s, e = chunks_sorted[lo]["char_start"], chunks_sorted[hi]["char_end"]
    if s is not None and e is not None and 0 <= s < e <= len(content):
        return content[s:e], s, e
    return "\n".join(c["chunk_text"] for c in chunks_sorted[lo:hi + 1]), s, e


def _overlaps(ranges, s, e) -> bool:
    for r0, r1 in ranges:
        inter = max(0, min(e, r1) - max(s, r0))
        smaller = min(e - s, r1 - r0)
        if smaller > 0 and inter / smaller > OVERLAP_DEDUP_RATIO:
            return True
    return False


async def indexed_file_read(req: IndexedFileReadRequest):
    """
    Lê arquivos indexados. Três modos, mutuamente exclusivos:

      1. `file_path` sozinho → lookup EXATO por caminho; devolve o arquivo inteiro.
      2. `file_path` + `query` → ranqueia só os chunks DESSE arquivo (vetor + BM25,
         rerank opcional), descartando chunks sobrepostos; até `top_k` chunks.
      3. só `query` → busca global (escopo ativo + geral): vetor + BM25 fundidos
         por RRF, rerank cross-encoder do pool, melhor chunk por arquivo.
         O conteúdo completo só vem se `include_full_content` for pedido
         explicitamente.

    Campos opcionais lidos por getattr (adicione-os ao IndexedFileReadRequest
    quando quiser expô-los): `hybrid`, `rerank`, `context_chunks`.
    """
    file_path = (req.file_path or "").strip()
    query = (req.query or "").strip()
    top_k = req.top_k
    min_score = req.min_score
    hybrid = bool(getattr(req, "hybrid", HYBRID_SEARCH))
    rerank = bool(getattr(req, "rerank", RERANK_ENABLED))
    ctx_n = max(0, min(int(getattr(req, "context_chunks", 0) or 0), MAX_CONTEXT_CHUNKS))

    # ── Modo 1/2: caminho absoluto informado ──
    if file_path:
        resolved = await _db(state.if_db, _if_backend_for_path, file_path)
        if resolved is None:
            return IndexedFileReadResponse(results=[], file_path=file_path, query=query or None)
        if_db, if_index, row = resolved
        tag = _scope_tag_of(if_db)
        _touch(if_db, [row["id"]])

        if not query:
            entry = _entry(row, tag, score=1.0, content=row["content"], match_type="exact_path")
            return IndexedFileReadResponse(results=[entry], file_path=file_path)

        # ── Modo 2: query dentro do arquivo ──
        chunk_rows = await _db(if_db, if_db.get_chunks_by_file, row["id"])
        if not chunk_rows:
            return IndexedFileReadResponse(results=[], file_path=file_path, query=query)
        chunk_rows = sorted(chunk_rows, key=lambda c: (c["chunk_index"], c["id"]))
        chunk_ids = [c["id"] for c in chunk_rows]
        by_id = {(tag, c["id"]): c for c in chunk_rows}

        emb = await state.embed_engine.embed_query_one(query)
        K = min(len(chunk_ids), max(top_k * 4, RERANK_POOL))
        ranked, _ = await _gather(query, emb, [(if_db, if_index, tag)], K=K, min_score=min_score,
                                  hybrid=hybrid, restrict=chunk_ids, file_id=row["id"])
        if not ranked:
            return IndexedFileReadResponse(results=[], file_path=file_path, query=query)
        ordered = await _order_pool(query, ranked[:max(RERANK_POOL, top_k * 4)], by_id, rerank)

        full_content = row["content"] if _wants_full_content(req, req.include_full_content) else ""
        results, ranges = [], []
        for c, score in ordered:
            cr = by_id[c.key]
            text, s, e = _expand(row["content"], chunk_rows, cr, ctx_n)
            if s is not None and e is not None and _overlaps(ranges, s, e):
                continue
            if s is not None and e is not None:
                ranges.append((s, e))
            results.append(_entry(
                row, tag, score=score, content=full_content, chunk_row=cr, chunk_text=text,
                char_start=s, char_end=e, chunk_id=_out_ref(tag, cr["id"]),
                match_type="exact_path_chunks",
            ))
            if len(results) >= top_k:
                break
        return IndexedFileReadResponse(results=results, file_path=file_path, query=query)

    # ── Modo 3: busca global (ativo + geral) ──
    if not query:
        raise MemoryToolError("informe 'file_path' (leitura exata) ou 'query' (busca semântica)")

    emb = await state.embed_engine.embed_query_one(query)
    gen = _general()
    backends = [(state.if_db, state.if_index, None)]
    if state.active_scope != GENERAL_SCOPE and gen is not None and gen.if_db is not state.if_db:
        backends.append((gen.if_db, gen.if_index, GENERAL_SCOPE))
    backends = [b for b in backends if b[1].total > 0]
    if not backends:
        return IndexedFileReadResponse(results=[], query=query)
    dbs = {tag: db for db, _, tag in backends}

    cap = max(sum(int(b[1].total) for b in backends), 1)
    K = min(max(top_k * 3, 20), cap)
    pool_size = max(RERANK_POOL, top_k * 4)
    for _ in range(4):                       # over-fetch adaptativo até ter top_k arquivos distintos
        ranked, saturated = await _gather(query, emb, backends, K=K, min_score=min_score, hybrid=hybrid)
        pool = ranked[:pool_size]
        rows = await _rows_for(pool, dbs)
        distinct = {(c.tag, rows[c.key]["file_id"]) for c in pool if c.key in rows}
        if len(distinct) >= top_k or not saturated or K >= cap:
            break
        K = min(K * 2, cap)

    if not rows:
        return IndexedFileReadResponse(results=[], query=query)
    ordered = await _order_pool(query, pool, rows, rerank)

    picked, seen = [], set()                 # melhor chunk por arquivo, já cortado em top_k
    for c, score in ordered:
        fkey = (c.tag, rows[c.key]["file_id"])
        if fkey in seen:
            continue
        seen.add(fkey)
        picked.append((c, score, rows[c.key]))
        if len(picked) >= top_k:
            break

    files = {}
    for tag in {c.tag for c, _, _ in picked}:
        ids = [r["file_id"] for c, _, r in picked if c.tag == tag]
        for fr in await _db(dbs[tag], _files_by_ids, dbs[tag], ids):
            files[(tag, fr["id"])] = fr

    want_full = _wants_full_content(req, False)
    results, touched = [], defaultdict(list)
    for c, score, cr in picked:
        fr = files.get((c.tag, cr["file_id"]))
        if fr is None:
            continue
        text, s, e = cr["chunk_text"], cr["char_start"], cr["char_end"]
        if ctx_n:
            sib = await _db(dbs[c.tag], dbs[c.tag].get_chunks_by_file, fr["id"])
            sib = sorted(sib, key=lambda x: (x["chunk_index"], x["id"]))
            text, s, e = _expand(fr["content"], sib, cr, ctx_n)
        results.append(_entry(
            fr, c.tag, score=score, content=fr["content"] if want_full else "", chunk_row=cr,
            chunk_text=text, char_start=s, char_end=e, match_type=c.match_type,
        ))
        touched[c.tag].append(fr["id"])
    for tag, ids in touched.items():
        _touch(dbs[tag], ids)
    return IndexedFileReadResponse(results=results, query=query)


# ══════════════════════════════════════════════════════════════════════════════
# check / get / chunk / list
# ══════════════════════════════════════════════════════════════════════════════

async def indexed_file_check(file_path: str, file_hash: Optional[str] = None,
                             content_hash: Optional[str] = None):
    """
    Verifica se um arquivo está indexado e se o hash bate.

      - indexed=False  → nunca indexado
      - hash_match=True → hashes informados batem E há chunks (nada a fazer)
      - hash_match=False → hashes divergem ou arquivo sem chunks → reindexar
      - hash_match=None → nenhum hash foi informado (o chamador compara)

    `file_hash` / `content_hash` são opcionais (compatível com chamadas antigas).
    """
    if not file_path:
        raise MemoryToolError("file_path vazio")
    resolved = await _db(state.if_db, _if_backend_for_path, file_path)
    if resolved is None:
        return IndexedFileCheckResponse(indexed=False)
    if_db, _, row = resolved
    tag = _scope_tag_of(if_db)
    chunks_count = await _db(if_db, if_db.count_chunks, row["id"])

    hash_match = None
    if file_hash is not None or content_hash is not None:
        ok = chunks_count > 0
        if file_hash is not None:
            ok = ok and row["file_hash"] == file_hash
        if content_hash is not None:
            ok = ok and row["content_hash"] == content_hash
        hash_match = ok

    return IndexedFileCheckResponse(
        indexed=True,
        file_id=_out_ref(tag, row["id"]),
        stored_file_hash=row["file_hash"],
        stored_content_hash=row["content_hash"],
        stored_modified=row["modified"],
        chunks_count=chunks_count,
        hash_match=hash_match,
    )


async def indexed_file_get(file_id: int):
    """Conteúdo completo de um arquivo indexado pelo ID (aceita id composta)."""
    resolved = await _db(state.if_db, _if_backend_for_file_id, file_id)
    if resolved is None:
        raise MemoryToolError(f"Arquivo indexado #{file_id} não encontrado")
    if_db, _, row, out_id = resolved
    chunks_count = await _db(if_db, if_db.count_chunks, row["id"])
    return {
        "file_id": out_id, "file_path": row["file_path"], "file_name": row["file_name"],
        "extension": row["extension"], "content": row["content"],
        "content_hash": row["content_hash"], "file_hash": row["file_hash"],
        "size": row["size"], "modified": row["modified"], "source": row["source"],
        "confidence": row["confidence"], "created_at": row["created_at"],
        "access_count": row["access_count"], "chunks_count": chunks_count,
    }


async def get_chunk_by_id(chunk_id: int, include_file_content: bool = True):
    """Chunk por id (crua → ativo, com fallback ao geral; composta → geral)."""
    gen = _general()
    candidates = []
    if is_general_ref(chunk_id):
        if gen is not None:
            candidates.append((gen.if_db, ref_real_id(chunk_id)))
    else:
        candidates.append((state.if_db, chunk_id))
        if gen is not None and gen.if_db is not state.if_db:
            candidates.append((gen.if_db, chunk_id))
    for if_db, real_id in candidates:
        row = await _db(if_db, _chunk_with_file, if_db, real_id)
        if row is None:
            continue
        out_id = chunk_id if if_db is state.if_db or is_general_ref(chunk_id) else encode_ref(GENERAL_SCOPE, real_id)
        return {
            "chunk_id": out_id,
            "file_path": row["file_path"], "file_hash": row["file_hash"],
            "char_start": row["char_start"], "char_end": row["char_end"],
            "chunk_text": row["chunk_text"],
            "file_content": row["content"] if include_file_content else None,
        }
    raise MemoryToolError(f"Chunk #{chunk_id} não encontrado")


async def indexed_file_list():
    """Lista mesclada: escopo ativo (ids cruas) + geral (ids compostas) com projeto ativo."""
    gen = _general()
    backends = [(state.if_db, None)]
    if state.active_scope != GENERAL_SCOPE and gen is not None and gen.if_db is not state.if_db:
        backends.append((gen.if_db, GENERAL_SCOPE))
    files = []
    for if_db, tag in backends:
        for row in await _db(if_db, if_db.list_files):
            files.append({
                "file_id": _out_ref(tag, row["id"]),
                "file_path": row["file_path"], "file_name": row["file_name"],
                "extension": row["extension"], "content_hash": row["content_hash"],
                "file_hash": row["file_hash"], "size": row["size"], "modified": row["modified"],
                "created_at": row["created_at"], "access_count": row["access_count"],
                "scope": tag,
            })
    return {"total": len(files), "files": files}


# ══════════════════════════════════════════════════════════════════════════════
# Deleção (DB primeiro; vetores órfãos são inofensivos — chunk_map os descarta)
# ══════════════════════════════════════════════════════════════════════════════

async def _delete_row(if_db, if_index, row, out_id, *, defer_vectors: Optional[set] = None):
    real_id = row["id"]
    chunk_ids = await _db(if_db, if_db.get_chunk_ids_by_file, real_id)
    deleted = await _db(if_db, if_db.delete_file, real_id)        # cascata nos chunks
    if defer_vectors is not None:
        defer_vectors.update(chunk_ids)
    elif chunk_ids:
        try:
            await _remove_vectors(if_index, chunk_ids)
        except Exception as e:
            log.error(f"Vetores órfãos após deletar #{real_id}: {e}")
    log.info(f"Indexed file deleted: #{real_id} ({len(chunk_ids)} chunks removed)")
    return {"deleted": deleted, "file_id": out_id, "chunks_removed": len(chunk_ids)}


async def indexed_file_delete(file_id: int):
    """Remove um arquivo e seus chunks. Id crua só alcança o escopo ATIVO; para o
    geral (com projeto ativo) use a id composta devolvida por /list."""
    resolved = await _db(state.if_db, _if_backend_for_file_id, file_id, allow_general_fallback=False)
    if resolved is None:
        raise MemoryToolError(f"Arquivo indexado #{file_id} não encontrado")
    if_db, if_index, row, out_id = resolved
    async with _PATH_LOCKS.hold(row["file_path"]):
        fresh = await _db(if_db, if_db.get_by_id, row["id"])
        if fresh is None:
            return {"deleted": 0, "file_id": out_id, "chunks_removed": 0}
        return await _delete_row(if_db, if_index, fresh, out_id)


async def indexed_file_delete_by_path(file_path: str):
    """Remove um arquivo indexado pelo caminho (ativo → geral)."""
    if not file_path:
        raise MemoryToolError("file_path vazio")
    async with _PATH_LOCKS.hold(file_path):
        resolved = await _db(state.if_db, _if_backend_for_path, file_path)
        if resolved is None:
            return {"deleted": 0, "file_path": file_path, "message": "not indexed"}
        if_db, if_index, row = resolved
        return await _delete_row(if_db, if_index, row, _out_ref(_scope_tag_of(if_db), row["id"]))


async def indexed_file_delete_many(file_ids: Optional[list] = None, file_paths: Optional[list] = None):
    """Remove vários arquivos agrupando a remoção de vetores (1 remove_ids por índice)."""
    pending: dict[int, tuple] = {}
    results, total_chunks, total_deleted = [], 0, 0

    async def handle(resolved):
        nonlocal total_chunks, total_deleted
        if_db, if_index, row, out_id = resolved
        async with _PATH_LOCKS.hold(row["file_path"]):
            fresh = await _db(if_db, if_db.get_by_id, row["id"])
            if fresh is None:
                results.append({"file_id": out_id, "deleted": 0})
                return
            bucket = pending.setdefault(id(if_index), (if_index, set()))
            r = await _delete_row(if_db, if_index, fresh, out_id, defer_vectors=bucket[1])
            results.append(r)
            total_chunks += r["chunks_removed"]
            total_deleted += int(bool(r["deleted"]))

    for fid in file_ids or []:
        resolved = await _db(state.if_db, _if_backend_for_file_id, fid, allow_general_fallback=False)
        if resolved is None:
            results.append({"file_id": fid, "deleted": 0, "message": "not found"})
        else:
            await handle(resolved)
    for path in file_paths or []:
        res = await _db(state.if_db, _if_backend_for_path, path)
        if res is None:
            results.append({"file_path": path, "deleted": 0, "message": "not indexed"})
        else:
            if_db, if_index, row = res
            await handle((if_db, if_index, row, _out_ref(_scope_tag_of(if_db), row["id"])))

    for if_index, ids in pending.values():
        try:
            await _remove_vectors(if_index, ids)
        except Exception as e:
            log.error(f"Vetores órfãos após delete_many ({len(ids)}): {e}")
    return {"deleted": total_deleted, "chunks_removed": total_chunks, "results": results}


# ══════════════════════════════════════════════════════════════════════════════
# Reconciliação (rodar na inicialização ou sob demanda)
# ══════════════════════════════════════════════════════════════════════════════

def _index_id_set(if_index):
    """Tenta obter o conjunto de ids do índice; None se o índice não expõe."""
    for name in ("ids", "id_set", "_ids", "all_ids"):
        v = getattr(if_index, name, None)
        if v is None:
            continue
        try:
            v = v() if callable(v) else v
            return set(int(x) for x in v)
        except Exception:
            continue
    return None


async def indexed_file_reconcile():
    """Conserta inconsistências SQLite × FAISS:
      • arquivos sem chunks (ex.: falha de uma versão antiga) → reindexa do conteúdo salvo;
      • chunks sem vetor / vetores órfãos → reembeda / remove (se o índice expõe seus ids;
        caso contrário só compara contagens e avisa)."""
    gen = _general()
    backends = [(state.if_db, state.if_index, None)]
    if gen is not None and gen.if_db is not state.if_db:
        backends.append((gen.if_db, gen.if_index, GENERAL_SCOPE))
    report = []
    loop = asyncio.get_running_loop()

    for if_db, if_index, tag in backends:
        rep = {"scope": tag or "active", "reindexed": 0, "failed": 0}
        await _db(if_db, _ensure_fts_sync, if_db)
        for r in await _db(if_db, if_db.list_files):
            if await _db(if_db, if_db.count_chunks, r["id"]) > 0:
                continue
            async with _PATH_LOCKS.hold(r["file_path"]):
                full = await _db(if_db, if_db.get_by_id, r["id"])
                if full is None or not (full["content"] or "").strip():
                    continue
                content = full["content"]
                try:
                    chunks = await loop.run_in_executor(None, _chunk_content, content, full["extension"])
                    embs = await _embed_passages_cached([c["text"] for c in chunks])
                    ch = hashlib.sha256(content.encode("utf-8", "replace")).hexdigest()
                    await _commit_full(if_db, if_index, full, _meta_from_row(full, content), content, ch, chunks, embs)
                    rep["reindexed"] += 1
                except Exception as e:
                    rep["failed"] += 1
                    log.error(f"reconcile: falha ao reindexar {full['file_path']}: {e}")

        db_ids = await _db(if_db, _all_chunk_ids, if_db)
        rep["db_chunks"], rep["index_vectors"] = len(db_ids), getattr(if_index, "total", None)
        idx_ids = _index_id_set(if_index)
        if idx_ids is not None:
            orphans, missing = idx_ids - db_ids, db_ids - idx_ids
            if orphans:
                await _remove_vectors(if_index, orphans)
            if missing:
                rows = await _db(if_db, _chunks_by_ids, if_db, list(missing))
                embs = await _embed_passages_cached([r["chunk_text"] for r in rows])
                await loop.run_in_executor(None, if_index.add_batch, embs, [r["id"] for r in rows])
            rep.update(orphan_vectors_removed=len(orphans), vectors_restored=len(missing))
        elif rep["index_vectors"] is not None and rep["index_vectors"] != rep["db_chunks"]:
            log.warning(f"reconcile [{rep['scope']}]: {rep['db_chunks']} chunks no DB × "
                        f"{rep['index_vectors']} vetores — índice não expõe ids para reparo automático")
        report.append(rep)
    return {"scopes": report}


__all__ = [
    '_if_backend_for_chunk_id',
    '_if_backend_for_file_id',
    '_if_backend_for_path',
    'get_chunk_by_id',
    'indexed_file_check',
    'indexed_file_delete',
    'indexed_file_delete_by_path',
    'indexed_file_delete_many',
    'indexed_file_get',
    'indexed_file_list',
    'indexed_file_read',
    'indexed_file_reconcile',
    'indexed_file_write',
]