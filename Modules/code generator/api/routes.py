"""
REST API Routes
Only POST endpoints — all interactions are submitted as payloads.
"""

from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional
import uuid

router = APIRouter()


# ─── Request Models ───────────────────────────────────────────────────────────

class CodeRequest(BaseModel):
    request_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str = Field(..., description="What the user wants: write, refactor, or fix code.")
    language: Optional[str] = Field(default="python", description="Target programming language.")
    context_files: Optional[list[str]] = Field(default=[], description="Paths or raw code snippets for additional context.")
    preferences: Optional[dict] = Field(default={}, description="User preferences (style, frameworks, etc.).")


class FeedbackRequest(BaseModel):
    request_id: str = Field(..., description="ID of the original generation request.")
    accepted: bool = Field(..., description="Whether the user accepted the generated code.")
    modifications: Optional[str] = Field(default=None, description="If the user changed the code, send the final version here.")
    notes: Optional[str] = Field(default=None, description="Free-text feedback from the user.")


class MemoryRequest(BaseModel):
    key: str = Field(..., description="Preference key (e.g. 'indent_style', 'preferred_framework').")
    value: str = Field(..., description="Preference value.")
    scope: Optional[str] = Field(default="global", description="Scope: 'global' or a specific language/project.")


class ResearchRequest(BaseModel):
    topic: str = Field(..., description="Topic to search for (library docs, patterns, known bugs, etc.).")
    language: Optional[str] = Field(default=None)
    max_results: Optional[int] = Field(default=5, ge=1, le=20)


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/generate", summary="Submit a code generation request")
async def generate_code(payload: CodeRequest, request: Request, background_tasks: BackgroundTasks):
    """
    Main endpoint. Accepts a natural language prompt and optional context.
    The pipeline will:
      1. Semantically classify the intent (write / refactor / fix)
      2. Query memory for user preferences and past bug fixes
      3. Query the research module for relevant docs / patterns
      4. Generate code via llama.cpp GGUF server
      5. Validate in a sandboxed container
      6. Auto-retry in bug-fix mode on failure (up to MAX_RETRIES)
      7. Return the final code with metadata
    """
    orchestrator = request.app.state.orchestrator
    try:
        result = await orchestrator.run(payload.model_dump())
        return {
            "request_id": payload.request_id,
            "status": "success",
            "intent": result["intent"],
            "language": result["language"],
            "code": result["code"],
            "confidence_score": result["confidence_score"],
            "sandbox_passed": result["sandbox_passed"],
            "iterations": result["iterations"],
            "sources_used": result["sources_used"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback", summary="Submit feedback on a generated result")
async def submit_feedback(payload: FeedbackRequest, request: Request):
    """
    Closes the learning loop. If the user modified or rejected the code,
    that signal is stored in memory and used to improve future generations.
    """
    orchestrator = request.app.state.orchestrator
    try:
        await orchestrator.memory.store_feedback(payload.model_dump())
        return {
            "request_id": payload.request_id,
            "status": "feedback_recorded",
            "message": "Thank you. This will improve future generations for your context.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/memory/preference", summary="Store a user preference")
async def store_preference(payload: MemoryRequest, request: Request):
    """
    Explicitly sets a preference in the memory module.
    E.g.: preferred framework, indentation style, naming conventions.
    """
    orchestrator = request.app.state.orchestrator
    try:
        await orchestrator.memory.store_preference(
            key=payload.key,
            value=payload.value,
            scope=payload.scope,
        )
        return {
            "status": "stored",
            "key": payload.key,
            "value": payload.value,
            "scope": payload.scope,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/research", summary="Trigger a standalone research query")
async def research(payload: ResearchRequest, request: Request):
    """
    Runs the research module independently.
    Useful for pre-loading context before a generation request,
    or for inspecting what the pipeline would retrieve for a given topic.
    """
    orchestrator = request.app.state.orchestrator
    try:
        results = await orchestrator.research.query(
            topic=payload.topic,
            language=payload.language,
            max_results=payload.max_results,
        )
        return {
            "topic": payload.topic,
            "results": results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
