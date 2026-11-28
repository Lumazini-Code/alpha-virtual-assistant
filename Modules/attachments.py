# Alpha AI (AVA) — Anexos multimídia (imagens e documentos)
# Copyright (C) 2026  Felipe lumazini
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regras puras sobre anexos de um turno — sem FastAPI, sem httpx, sem DB.

Tanto a WebUI (seleção múltipla/pasta) quanto a TUI (um anexo) chegam ao
orchestrator pelo ``/execute``. Aqui ficam a normalização dos dois formatos,
a detecção de MIME por assinatura binária e a conversão do marcador devolvido
pela tool ``view_image`` em blocos multimodais do OpenAI.

O módulo é intencionalmente stdlib-only para poder ser testado fora do
container, onde as dependências do orchestrator não estão instaladas.
"""
from __future__ import annotations

import base64
from typing import Any

IMAGE_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff",      "image/jpeg"),
    (b"GIF87a",            "image/gif"),
    (b"GIF89a",            "image/gif"),
    (b"BM",                "image/bmp"),
)

# Marcador devolvido pela tool `view_image`: o orchestrator converte o base64
# em bloco image_url na mensagem role="tool", em vez de mandar o base64 como
# texto para o modelo.
IMAGE_RESULT_KEY = "__ava_image__"


def detect_image_mime(b64: str) -> str:
    """MIME de um base64 de imagem (prefixo data: opcional e ignorado)."""
    payload = b64.split(",", 1)[-1]
    try:
        head = base64.b64decode(payload[:64], validate=False)[:12]
    except Exception:
        return "image/jpeg"
    for magic, mime in IMAGE_MAGIC:
        if head.startswith(magic):
            return mime
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def normalize_request_attachments(req: Any) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Unifica anexos novos (listas) e antigos (singleton) sem duplicação.

    As listas ganham precedência; os campos legados só entram quando a lista
    correspondente veio vazia. Dados repetidos são descartados para o mesmo
    base64 não virar dois blocos image_url no contexto.
    """
    images: list[dict[str, str]] = []
    seen_images: set[str] = set()
    for att in getattr(req, "images", None) or []:
        data = getattr(att, "data", None)
        if isinstance(data, str) and data.startswith("data:") and "," in data:
            # Cliente que mandou data URL completa em vez de base64 puro.
            data = data.split(",", 1)[-1]
        if not isinstance(data, str) or not data or data in seen_images:
            continue
        seen_images.add(data)
        images.append({"name": str(getattr(att, "name", None) or ""), "data": data})
    if not images and getattr(req, "image_base64", None):
        legacy = req.image_base64
        if legacy.startswith("data:") and "," in legacy:
            legacy = legacy.split(",", 1)[-1]
        images.append({"name": "", "data": legacy})

    documents: list[dict[str, str]] = []
    seen_docs: set[str] = set()
    for att in getattr(req, "documents", None) or []:
        data = getattr(att, "data", None)
        if not isinstance(data, str) or not data or data in seen_docs:
            continue
        seen_docs.add(data)
        documents.append({"name": str(getattr(att, "name", None) or ""), "data": data})
    if not documents and getattr(req, "document_content", None):
        documents.append({"name": "", "data": req.document_content})
    return images, documents


def redact_image_result(res: Any) -> Any:
    """Remove o base64 de logs e do evento SSE de tool_result (a imagem vai
    embutida na mensagem da tool, não no payload de telemetria)."""
    if isinstance(res, dict) and isinstance(res.get(IMAGE_RESULT_KEY), dict):
        meta = dict(res[IMAGE_RESULT_KEY])
        meta.pop("base64", None)
        return {IMAGE_RESULT_KEY: meta}
    return res


def image_result_content(image: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Conteúdo multi-parte de uma imagem lida do host, ou None se inválido.

    Formato aceito por ``ToolCallMessage``/LLM.py e repassado verbatim ao
    OpenRouter: um bloco de texto (o caption) seguido do bloco image_url.
    """
    data = image.get("base64")
    if not isinstance(data, str) or not data:
        return None
    mime = str(image.get("mime") or detect_image_mime(data))
    caption = str(image.get("caption") or "[Imagem abaixo.]")
    return [
        {"type": "text", "text": caption},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
    ]
