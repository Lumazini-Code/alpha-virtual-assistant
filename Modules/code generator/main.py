"""
CodeGen REST API — Entry Point
"""

import logging
import traceback

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from core.orchestrator import Orchestrator
from api.routes import router

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    orchestrator = Orchestrator()
    await orchestrator.startup()
    app.state.orchestrator = orchestrator
    yield
    await orchestrator.shutdown()


app = FastAPI(
    title="CodeGen API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    logger.error(f"Unhandled exception on {request.url}:\n{tb}")
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "traceback": tb},
    )


app.include_router(router, prefix="/api/v1")