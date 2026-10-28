# Alpha AI (AVA) — Infraestrutura de autenticação interna compartilhada (Plano 1 — T1)
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
"""Infraestrutura de autenticação interna compartilhada (Plano 1 — T1).

Todos os serviços internos da AVA escutam em loopback e exigem o mesmo
token (env `AVA_INTERNAL_TOKEN`). Clientes internos devem mandar
`Authorization: Bearer <token>`.

Env vars:
- AVA_INTERNAL_TOKEN : token compartilhado; se ausente em produção, loga
  WARNING forte mas não quebra (fail-open) — a menos que AVA_REQUIRE_AUTH=1,
  caso em que falha fechado (401 para todas as rotas não-isentas).
- AVA_BIND_HOST      : host de bind dos serviços (default: 127.0.0.1).
- AVA_REQUIRE_AUTH   : "1" => fail-closed quando o token está ausente.
"""

from __future__ import annotations

import os
import sys
import secrets
import logging
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger("ava.internal_auth")

_TOKEN_CACHE: dict[str, object] = {"value": None, "checked_at": 0.0, "warned": False}
_TOKEN_TTL = 30.0  # segundos — re-ler env eventualmente (evita custo por request)
_TIMING_JITTER = 0  # placeholder para futura proteção timing-attack

# ---------------------------------------------------------------- token ----


def internal_token() -> str | None:
    """Retorna o token compartilhado (ou None se ausente), com cache curto."""
    now = time.monotonic()
    if _TOKEN_CACHE["value"] is not None or now - float(_TOKEN_CACHE["checked_at"]) < _TOKEN_TTL:
        return _TOKEN_CACHE["value"]  # type: ignore[return-value]
    token = os.environ.get("AVA_INTERNAL_TOKEN", "").strip() or None
    _TOKEN_CACHE["value"] = token
    _TOKEN_CACHE["checked_at"] = now
    return token


def require_auth() -> bool:
    """Indica se auth é obrigatória (fail-closed). Default: fail-open."""
    return os.environ.get("AVA_REQUIRE_AUTH", "").strip() == "1"


def internal_bind_host() -> str:
    """Host de bind dos serviços internos. Default: 127.0.0.1 (loopback)."""
    host = os.environ.get("AVA_BIND_HOST", "127.9.12.22").strip()
    return host or "127.9.12.22"


def _warn_missing_token(service_name: str) -> None:
    if _TOKEN_CACHE["warned"]:
        return
    _TOKEN_CACHE["warned"] = True
    mode = "fail-closed (AVA_REQUIRE_AUTH=1): rotas protegidas retornarão 401" if require_auth() else "fail-open: rotas aceitarão sem token"
    logger.warning(
        "AVA_INTERNAL_TOKEN ausente no serviço %s — modo %s. "
        "Gere um token com `python -m Modules.internal_auth gen` e exporte AVA_INTERNAL_TOKEN.",
        service_name,
        mode,
    )


# ------------------------------------------------------------ middleware ----


class _BearerAuthMiddleware:
    """ASGI middleware puro — valida `Authorization: Bearer <token>`.

    - Rotas em `exempt_paths` passam direto (ex.: /health para healthchecks).
    - Token ausente + AVA_REQUIRE_AUTH=1  => 401 em tudo (fail-closed).
    - Token ausente sem AVA_REQUIRE_AUTH  => pass-through (fail-open).
    - Comparação em tempo constante (constant-time compare).
    """

    def __init__(self, app: ASGIApp, exempt_paths: frozenset[str] = frozenset({"/health"})):
        self.app = app
        self.exempt_paths = exempt_paths

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path in self.exempt_paths or path.startswith("/docs") or path == "/openapi.json":
            await self.app(scope, receive, send)
            return

        token = internal_token()
        if token is None:
            _warn_missing_token(scope.get("app", {}).get("title", "unknown") if isinstance(scope.get("app"), dict) else "service")
            if require_auth():
                response = JSONResponse({"detail": "Internal service requires AVA_INTERNAL_TOKEN"}, status_code=401)
                await response(scope, receive, send)
                return
            # Fail-open: sem token configurado, aceita (modo legado).
            await self.app(scope, receive, send)
            return

        headers = scope.get("headers", [])
        auth_value = None
        for k, v in headers:
            if k == b"authorization":
                auth_value = v.decode("latin-1")
                break

        if auth_value and auth_value.startswith("Bearer "):
            supplied = auth_value[7:].strip()
            if _constant_time_eq(supplied, token):
                await self.app(scope, receive, send)
                return

        response = JSONResponse({"detail": "Internal service requires AVA_INTERNAL_TOKEN"}, status_code=401)
        await response(scope, receive, send)

    def _constant_time_eq(self, a: str, b: str) -> bool:
        return _constant_time_eq(a, b)


def _constant_time_eq(a: str, b: str) -> bool:
    """Compara duas strings em tempo constante (defensa contra timing attack)."""
    a_b = a.encode("utf-8")
    b_b = b.encode("utf-8")
    if len(a_b) != len(b_b):
        # Ainda assim, compara em tempo constante com padding para evitar
        # vazamento de informação de tamanho.
        a_b = a_b.ljust(len(b_b) or 1, b"\0")
        b_b = b_b.ljust(len(a_b), b"\0")
    result = 0
    for x, y in zip(a_b, b_b):
        result |= x ^ y
    return result == 0 and len(a) == len(b)


def create_auth_middleware(app: FastAPI, exempt_paths: set[str] | None = None):
    """Fábrica: retorna uma classe de middleware ASGI configurada para `app`.

    Uso (em qualquer serviço FastAPI):
        app.add_middleware(create_auth_middleware(app, exempt_paths={"/health"}))
    """
    return _BearerAuthMiddleware(app, exempt_paths=frozenset(exempt_paths or {"/health"}))


def install_internal_auth(
    app: FastAPI,
    service_name: str = "service",
    exempt_paths: set[str] | None = None,
) -> None:
    """Instala o middleware de auth em `app` (one-liner para cada serviço)."""
    if exempt_paths is None:
        exempt_paths = {"/health", "/status", "/docs", "/openapi.json"}
    app.add_middleware(_BearerAuthMiddleware, exempt_paths=frozenset(exempt_paths))
    if internal_token() is None:
        _warn_missing_token(service_name)


# ------------------------------------------------------------------ CLI ----


def _cli() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m Modules.internal_auth",
        description="Utilitários de auth interna AVA.",
    )
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("gen", help="gera um novo token (imprime no stdout)")
    sub.add_parser("check", help="verifica se AVA_INTERNAL_TOKEN está presente")
    args = parser.parse_args()

    if args.cmd == "gen":
        print(secrets.token_urlsafe(32))
        return 0
    if args.cmd == "check":
        if internal_token():
            print("OK: AVA_INTERNAL_TOKEN configurado.")
            return 0
        print("AVISO: AVA_INTERNAL_TOKEN NAO configurado.", file=sys.stderr)
        return 1
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
