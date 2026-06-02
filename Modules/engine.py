"""
AVA KG-RAG — Pipeline Principal
Orquestra as 7 etapas do blueprint, adaptadas ao stack do AVA:

  1. Planner        → LLM (llama-server REST) decompõe o objetivo em sub-tópicos
  2. Web Researcher → DuckDuckGo search + httpx fetch + BeautifulSoup parse
  3. Distiller      → LLM extrai triplas (sujeito, relação, objeto) de cada página
  4. Chunker+Graph  → Semantic chunking + construção do KG (NetworkX + SQLite)
  5. Embeddings     → ONNX Serving API (multilingual-e5-small, mean pooling correto)
  6. Memória        → FAISS + KG SQLite persistidos entre sessões
  7. Retrieval      → Vetor + Grafo + Cross-Encoder → LLM Reasoning final

MODIFIED: All ONNX model calls now go through the unified onnx_serving API.
"""

import logging
import asyncio
from pathlib import Path

from modules.llm_client        import LLMClient
from modules.embedding_engine  import EmbeddingEngine
from modules.reranker          import CrossEncoderReranker
from modules.knowledge_graph   import KnowledgeGraph
from modules.semantic_chunker  import SemanticChunker
from modules.vector_store      import VectorStore
from modules.web_researcher    import WebResearcher
from config import RETRIEVAL, LLM

logger = logging.getLogger(__name__)


class AVAKnowledgeEngine:
    """
    Motor de KG-RAG do AVA.

    Uso principal (dentro de contexto async):
      answer = await engine.query("Como funciona fotossíntese em algas verdes?")

    O método query() executa o pipeline completo:
      Planner → WebResearch → Distill → Chunk+Graph → Embed → Store → Retrieve → Reason

    MODIFIED: All ONNX inference is delegated to the unified onnx_serving API
    via EmbeddingEngine and CrossEncoderReranker (which use onnx_client).
    """

    def __init__(self, lazy_models: bool = False):
        """
        lazy_models=True: não inicializa os clientes API imediatamente.
        Útil para testar sem o onnx_serving disponível.
        """
        logger.info("Iniciando AVAKnowledgeEngine...")

        self._llm        = LLMClient()
        self._kg         = KnowledgeGraph()
        self._store      = VectorStore()
        self._researcher = WebResearcher()
        self._embed      = None
        self._reranker   = None
        self._chunker    = SemanticChunker()   # Sem embed até _load_models()

        if not lazy_models:
            self._load_models()

        logger.info("AVAKnowledgeEngine pronto. Stats: %s", self.stats)

    def _load_models(self):
        if self._embed is not None:
            return
        try:
            # ── MODIFIED: These now create API clients, not local ONNX sessions ─
            self._embed    = EmbeddingEngine()
            self._reranker = CrossEncoderReranker()
            self._chunker  = SemanticChunker(embedding_engine=self._embed)
            logger.info("ONNX API clients initialized (onnx_serving).")
        except Exception as e:
            logger.warning("Failed to initialize ONNX API clients: %s", e)
            logger.warning("System will run without embeddings (degraded mode).")

    # ─── PIPELINE COMPLETO ────────────────────────────────────────────────────

    async def query(self, user_goal: str, verbose: bool = True) -> str:
        """
        Executa o pipeline completo de 7 etapas para responder ao objetivo.
        Método async — compatível com FastAPI sem conflito de event loop.

        Sempre busca na internet — não depende de documentos pré-indexados.
        Resultados são armazenados no FAISS+KG para queries futuras
        relacionadas (memória de longo prazo cresce com o uso).
        """
        self._load_models()

        if verbose:
            print(f"\n🎯 Objetivo: {user_goal}\n{'─'*60}")

        # ── Etapa 1: Planner ─────────────────────────────────────────────────
        if verbose:
            print("  [1/7] Planejando domínio de pesquisa...")
        sub_topics = self._plan(user_goal, verbose)

        # ── Etapa 2: Web Research ─────────────────────────────────────────────
        if verbose:
            print(f"\n  [2/7] Pesquisando na internet ({len(sub_topics)} sub-tópicos)...")

        pages = await self._researcher.research_async(sub_topics, verbose=verbose)

        if not pages:
            logger.error(
                "WebResearcher retornou 0 páginas para os sub-tópicos: %s", sub_topics
            )
            return (
                "❌ Não foi possível recuperar conteúdo da internet. "
                "Verifique a conexão e tente novamente."
            )

        if verbose:
            print(f"\n       Total: {len(pages)} páginas coletadas")

        # ── Etapas 3-6: Processar cada página → KG + FAISS ───────────────────
        total_chunks, total_triples = await self._process_pages(pages, verbose)

        if verbose:
            print(
                f"\n  [3-6] Processamento concluído\n"
                f"        Triplas extraídas : {total_triples}\n"
                f"        Chunks indexados  : {total_chunks}\n"
                f"        KG               : {self._kg.stats['nodes']} nós / "
                f"{self._kg.stats['edges']} arestas\n"
                f"        FAISS total      : {self._store.total} vetores"
            )

        # ── Etapa 7: Retrieval + Reasoning ────────────────────────────────────
        if not self._embed:
            return "⚠️ Modelos ONNX não carregados. Instale os modelos e reinicie."

        answer = await self._retrieve_and_reason(user_goal, verbose)

        if verbose:
            print(f"\n{'─'*60}\n")

        return answer

    # ─── Etapa 1: Planner ────────────────────────────────────────────────────

    def _plan(self, goal: str, verbose: bool) -> list[str]:
        try:
            plan       = self._llm.plan_domain(goal)
            sub_topics = plan.get("sub_topics", [])
            if not sub_topics:
                raise ValueError("Planner retornou lista vazia")
            if verbose:
                for i, t in enumerate(sub_topics, 1):
                    print(f"       {i}. {t}")
            return sub_topics
        except Exception as e:
            logger.warning("Planner falhou (%s), usando objetivo como query direta.", e)
            return [goal]

    # ─── Etapas 3-6: Distill + Chunk + Embed + Store ─────────────────────────

    async def _process_pages(
        self,
        pages: list[tuple[str, dict]],
        verbose: bool,
    ) -> tuple[int, int]:
        total_chunks  = 0
        total_triples = 0

        for i, (text, meta) in enumerate(pages, 1):
            source = meta.get("source", f"page_{i}")
            title  = meta.get("title", source)

            if verbose:
                print(f"\n  [3/7] Distilando: {title[:60]}...")

            # Etapa 3: Extrai triplas semânticas com o LLM
            triples = self._llm.distill_triples(text)
            if triples:
                added = self._kg.add_triples(triples, source_doc=source)
                total_triples += added

            # Etapa 4: Semantic chunking
            if self._embed:
                chunks = await self._chunker.chunk(text)    # ← await adicionado
            else:
                chunks = self._chunker.chunk_simple(text)

            if not chunks:
                continue

            # Etapa 5: Embeddings — MODIFIED: now uses async embed_passages
            if self._embed:
                chunk_texts = [c.text for c in chunks]
                # ── MODIFIED: embed_passages is now async via API ──────────────
                embeddings = await self._embed.embed_passages(chunk_texts)

                # Liga chunks → nós do KG por match de substring
                node_ids_per_chunk = []
                for chunk in chunks:
                    matched = [
                        nid
                        for nid, data in self._kg._graph.nodes(data=True)
                        if data.get("label", "").lower() in chunk.text.lower()
                    ]
                    node_ids_per_chunk.append(matched)

                # Etapa 6: Indexação FAISS (dedup SHA-256 automático)
                inserted = self._store.add(
                    texts=chunk_texts,
                    embeddings=embeddings,
                    source=source,
                    node_ids=node_ids_per_chunk,
                )
                total_chunks += inserted

        return total_chunks, total_triples

    # ─── Etapa 7: Retrieval + Reranking + Reasoning ──────────────────────────

    async def _retrieve_and_reason(self, goal: str, verbose: bool) -> str:
        if verbose:
            print("\n  [7a] Recuperação vetorial densa...")

        # ── MODIFIED: embed_query is now async via API ────────────────────────
        query_emb      = await self._embed.embed_query(goal)
        vector_results = self._store.search(query_emb, top_k=RETRIEVAL.top_k_dense)

        if not vector_results:
            return "❌ Nenhum vetor encontrado no índice após a pesquisa."

        # 7b: Graph neighborhood
        if verbose:
            print("  [7b] Enriquecimento pelo Knowledge Graph...")

        all_node_ids = []
        for entry, _ in vector_results:
            all_node_ids.extend(entry.node_ids)

        kg_matches = self._kg.search_nodes(goal)
        all_node_ids.extend(self._kg._node_id(m) for m in kg_matches)

        graph_facts = self._kg.get_neighborhood(
            list(set(all_node_ids)),
            hops=RETRIEVAL.graph_hops,
        )
        if verbose:
            print(f"       {len(graph_facts)} relações estruturais do KG")

        # 7c: Cross-encoder reranking — MODIFIED: now uses async rerank via API
        if verbose:
            print("  [7c] Reranking cross-encoder...")

        vector_texts   = [entry.text for entry, _ in vector_results]
        graph_texts    = graph_facts[:20]
        all_candidates = vector_texts + graph_texts

        # ── MODIFIED: rerank is now async via API ─────────────────────────────
        ranked = await self._reranker.rerank_async(
            query=goal,
            candidates=all_candidates,
            top_k=RETRIEVAL.top_k_final,
        )
        if verbose:
            print(f"       {len(all_candidates)} candidatos → {len(ranked)} finais")

        # 7d: Monta contexto enriquecido com fontes
        context_parts = []
        for rank_pos, (orig_idx, score, text) in enumerate(ranked, 1):
            if orig_idx < len(vector_results):
                entry        = vector_results[orig_idx][0]
                source_label = f"Fonte: {entry.source}"
            else:
                source_label = "Knowledge Graph"
            context_parts.append(
                f"[{rank_pos}] ({source_label}, relevância: {score:.3f})\n{text}"
            )

        context = "\n\n---\n\n".join(context_parts)

        if graph_facts:
            context += "\n\n---\n\nRelações do Knowledge Graph:\n" + "\n".join(graph_facts[:10])

        return f"Especialização completa: {context}"

    # ─── Utilitários ─────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        return {
            "vectors": self._store.total,
            "kg":      self._kg.stats,
        }

    def close(self):
        self._kg.close()
