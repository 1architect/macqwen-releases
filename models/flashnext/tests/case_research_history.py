"""Expose every research heading, bullet, and result row as evidence."""
from __future__ import annotations

from pathlib import Path
import re

from .api import ROOT, TestSpec


RESEARCH = ROOT / "docs" / "flashnext" / "research.md"


def _clean(value: str) -> str:
    value = re.sub(r"[`*_]", "", value)
    return re.sub(r"\s+", " ", value).strip()


def get_tests() -> list[TestSpec]:
    lines = RESEARCH.read_text().splitlines()
    result = []
    heading = "FlashNext research"
    intro = "Historical evidence from the chronological research record."
    for index, raw in enumerate(lines, 1):
        stripped = raw.strip()
        if stripped.startswith(("## ", "### ")):
            heading = _clean(stripped.lstrip("# "))
            paragraphs = []
            for candidate in lines[index:]:
                candidate = candidate.strip()
                if candidate.startswith("#"):
                    break
                if candidate and not candidate.startswith(("|", "```", "![")):
                    paragraphs.append(_clean(candidate))
                if len(" ".join(paragraphs)) > 280:
                    break
            intro = " ".join(paragraphs) or "Chronological research record."
            result.append(TestSpec(
                id=f"history-L{index}", title=heading, category="history",
                explanation=intro[:420],
                why="It preserves the measured proposal, controls, result, and decision from the research record.",
                source=f"docs/flashnext/research.md:{index}", status="historical",
            ))
            continue
        is_bullet = stripped.startswith("- ")
        is_row = stripped.startswith("|") and not set(stripped.replace("|", "").strip()) <= {"-", ":"}
        if not (is_bullet or is_row):
            continue
        text = _clean(stripped.lstrip("- ").strip("| "))
        if not text or text.lower().startswith(("condition |", "metric |", "profile |")):
            continue
        result.append(TestSpec(
            id=f"history-L{index}", title=f"{heading}: {text[:72]}", category="history",
            explanation=text[:500],
            why=f"This measured item belongs to the research question: {intro[:240]}",
            source=f"docs/flashnext/research.md:{index}", status="historical",
        ))
    return result
