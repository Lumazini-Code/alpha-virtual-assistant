# Alpha AI (AVA) — Testes unitários da pipeline do alpha-mcp — sem rede, só as partes puras
# Copyright (C) 2026  Felipe lumazini
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Testes unitários da pipeline do alpha-mcp — sem rede, só as partes puras.

Rodar: python test_pipeline.py   (precisa de rank-bm25/trafilatura instalados)
O servidor em si precisa dos serviços de rede (SearXNG etc.) — para teste
end-to-end use o cliente MCP: python test_mcp_client.py http://127.9.12.22:4010/mcp
"""

import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "ws_server", Path(__file__).resolve().parent / "server.py")
ws = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ws)

fails = []


def check(name, cond):
    print(("✓" if cond else "✗"), name)
    if not cond:
        fails.append(name)


# ── _tokenize / _bm25_rank ──
docs = [
    "Python é uma linguagem de programação interpretada e muito usada em ciência de dados",
    "Receita de bolo de chocolate com cobertura de brigadeiro",
    "Python possui muitas bibliotecas para ciência de dados e machine learning",
    "",
]
scores = ws._bm25_rank("python ciência de dados bibliotecas", docs)
check("bm25: docs de python rankeiam acima do bolo", scores[0] > scores[1])
check("bm25: doc vazio não quebra", scores[3] <= max(scores[:3]))

# ── filter_results ──
results = [
    {"url": "https://www.python.org/docs/", "title": "Python Docs", "content": "python docs", "score": 10},
    {"url": "https://pinterest.com/python", "title": "pin", "content": "python", "score": 9},
    {"url": "https://python.org/other", "title": "dup", "content": "python", "score": 8},  # mesmo domínio
    {"url": "https://realpython.com/tutorials/", "title": "Real", "content": "python tutorials", "score": 7},
    {"url": "não-url", "title": "bad", "content": "", "score": 6},
]
kept = ws.filter_results(results, max_urls=5)
doms = [k["url"] for k in kept]
check("filtro: blocklist aplicada", all("pinterest" not in u for u in doms))
check("filtro: dedup por domínio (1 python.org)", sum("python.org" in u for u in doms) == 1)
check("filtro: url inválida descartada", all(u.startswith("http") for u in doms))
check("filtro: realpython presente", any("realpython" in u for u in doms))

# ── chunk_markdown ──
md = """# Intro
Parágrafo introdutório com bastante texto relevante sobre a query buscada e contexto adicional necessário para passar do tamanho mínimo exigido pelo chunker do sistema.

## História da linguagem Python
Criada por Guido van Rossum em 1991. Muito texto aqui. """ + ("Detalhe histórico relevante. " * 40) + """

## Receita de bolo
Ingredientes: farinha, ovos, açúcar e chocolate em pó. Este parágrafo da receita de bolo não tem nenhuma relação com a query técnica de busca e serve para verificar que seções curtas ficam isoladas nos seus próprios chunks sem se misturar com as seções anteriores de história da linguagem.
"""
chunks = ws.chunk_markdown(md, "Python", "https://exemplo.com/python")
check("chunk: quebrou por headers", len(chunks) >= 3)
check("chunk: seção bolo isolada", any("bolo" in c["section"].lower() for c in chunks))
check("chunk: carrega url+title", all(c["url"] == "https://exemplo.com/python" and c["title"] == "Python" for c in chunks))
check("chunk: respeita max_chunk_chars", all(len(c["text"]) <= ws.CFG["chunks"]["max_chunk_chars"] + 50 for c in chunks))
check("chunk: descarta < min_chunk_chars", not any(len(c["text"]) < ws.CFG["chunks"]["min_chunk_chars"] for c in chunks))

# ── select_chunks ──
picked = ws.select_chunks(chunks, "história da linguagem Python Guido")
check("select: ≥1 chunk por página", picked and all(c["url"] == "https://exemplo.com/python" for c in picked))
check("select: melhor chunk é o de história", "História" in picked[0]["section"] or "história" in picked[0]["text"].lower())
per_page: dict[str, int] = {}
for c in picked:
    per_page[c["url"]] = per_page.get(c["url"], 0) + 1
check("select: respeita max_per_page", all(v <= ws.CFG["chunks"]["max_per_page"] for v in per_page.values()))
tok = sum(len(c["text"]) // 4 + 1 for c in picked)
check("select: dentro do orçamento de tokens", tok <= ws.CFG["chunks"]["max_tokens_total"])

# ── _format ──
out = ws._format(picked, "python")
check("format: citações com [n] e url", "[1]" in out and "https://exemplo.com/python" in out)

print()
print("FALHAS:", fails if fails else "nenhuma")
sys.exit(1 if fails else 0)