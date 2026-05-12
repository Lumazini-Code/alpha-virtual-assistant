"""
Context Ranker
Merges memory, research, and inline context into a single ranked payload
for the code generator. Resolves contradictions by source priority.

Priority order (highest → lowest):
  1. User inline context (code files passed directly)
  2. Memory — user preferences and past bug fixes
  3. Research — documentation and patterns
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Weights for relevance scoring
SOURCE_WEIGHTS = {
    "inline":    1.0,
    "memory":    0.85,
    "research":  0.65,
}

MAX_TOKENS_BUDGET = 6000  # Approximate token budget for context sent to LLM


class ContextRanker:

    def rank(
        self,
        user_input: str,
        memory: dict,
        research: list[dict],
        inline_context: list[str],
        preferences: dict,
    ) -> dict:
        """
        Builds a unified, de-duplicated, priority-ranked context dict
        ready to be serialised into the LLM prompt.
        """
        sections = []

        # ── Inline context (highest priority) ────────────────────────────────
        for snippet in inline_context:
            sections.append({
                "source": "inline",
                "weight": SOURCE_WEIGHTS["inline"],
                "content": snippet,
            })

        # ── Memory entries ────────────────────────────────────────────────────
        for entry in memory.get("entries", []):
            sections.append({
                "source": "memory",
                "weight": SOURCE_WEIGHTS["memory"] * entry.get("relevance", 1.0),
                "content": entry["content"],
                "type": entry.get("type"),  # "preference" | "bug_fix" | "feedback"
            })

        # ── Research results ──────────────────────────────────────────────────
        for result in research:
            sections.append({
                "source": "research",
                "weight": SOURCE_WEIGHTS["research"] * result.get("relevance", 1.0),
                "content": result["content"],
                "url": result.get("url"),
            })

        # ── Sort by weight descending ─────────────────────────────────────────
        sections.sort(key=lambda s: s["weight"], reverse=True)

        # ── Token budget trimming ─────────────────────────────────────────────
        sections = self._apply_budget(sections)

        # ── Detect and log conflicts ──────────────────────────────────────────
        conflicts = self._detect_conflicts(sections)
        if conflicts:
            logger.warning(f"Context conflicts detected: {conflicts}")

        return {
            "sections": sections,
            "preferences": preferences,
            "conflicts": conflicts,
            "total_sections": len(sections),
        }

    # ─── Budget ───────────────────────────────────────────────────────────────

    def _apply_budget(self, sections: list[dict]) -> list[dict]:
        """
        Naively estimates token count (~4 chars per token) and
        drops lowest-weight sections when over budget.
        """
        kept = []
        total_chars = 0
        budget_chars = MAX_TOKENS_BUDGET * 4

        for section in sections:
            size = len(section["content"])
            if total_chars + size <= budget_chars:
                kept.append(section)
                total_chars += size
            else:
                logger.debug(f"Context budget exceeded — dropping {section['source']} entry.")

        return kept

    # ─── Conflict Detection ───────────────────────────────────────────────────

    def _detect_conflicts(self, sections: list[dict]) -> list[str]:
        """
        Checks for contradictory preferences between memory and research.
        Returns human-readable conflict descriptions.
        """
        conflicts = []
        preference_values: dict[str, Any] = {}

        for section in sections:
            if section.get("type") == "preference":
                content = section["content"]
                # Simple key=value conflict detection
                if ":" in content:
                    key, _, val = content.partition(":")
                    key = key.strip()
                    val = val.strip()
                    if key in preference_values and preference_values[key] != val:
                        conflicts.append(
                            f"Preference conflict on '{key}': "
                            f"'{preference_values[key]}' vs '{val}' — using higher-priority value."
                        )
                    else:
                        preference_values[key] = val

        return conflicts
