"""
AVA — Alpha-code (porta 4006)
============================
Agente ReAct para edição e geração de código, integrado ao framework AVA.

Reuso de serviços:
  - LLM.py:4003      → inferência Groq com tool use nativo (POST /chat/tools)
  - scraping_client:3005 → files (read/write/list/str_replace) + terminal (execute)
  - onnxManager:2002 → embeddings (e5-small) + reranker (MiniLM)
  - memory.py:3001   → memória de sessão (opcional, pós-MVP)
  - CoT:3000         → planner (opcional, pós-MVP)

Próprio módulo:
  - ReAct loop com state machine
  - Tool registry (schemas OpenAI function-calling)
  - search_code (ripgrep local)
  - apply_patch (SEARCH/REPLACE robusto)
  - symbol_lookup (tree-sitter)
  - git tools (subprocess)
  - context budget tracker + summarization
  - model router (policy function)
  - SSE streaming de steps
"""

__version__ = "0.1.0"
