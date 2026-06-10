"""
src/confidence.py
Retrieval Strength scorer.

Computes mean cosine similarity of the final reranked chunks and maps it
to a labelled badge (HIGH / MEDIUM / LOW).

IMPORTANT: This is a retrieval quality signal — NOT a calibrated probability.
It measures how closely the retrieved text matches the query, not how likely
the answer is to be correct. Documented as "Retrieval Strength" throughout
the UI to avoid implying false precision.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import HIGH_RETRIEVAL, MEDIUM_RETRIEVAL
from typing import List


def compute_retrieval_strength(
    scores: List[float],
) -> tuple[float, str, str]:
    """
    Compute retrieval strength from cosine similarity scores.

    Args:
        scores: List of cosine similarity scores (0–1 range) from the retriever

    Returns:
        (score, label, description)
        - score: mean cosine similarity (0–1)
        - label: "HIGH" | "MEDIUM" | "LOW" | "NONE"
        - description: human-readable explanation for UI tooltip
    """
    if not scores:
        return 0.0, "NONE", "No passages retrieved."

    score = sum(scores) / len(scores)

    if score >= HIGH_RETRIEVAL:
        label = "HIGH"
        desc = (
            f"Score: {score:.2f} — Strong match. Multiple passages closely align "
            f"with your question. Answer is well-supported by the corpus."
        )
    elif score >= MEDIUM_RETRIEVAL:
        label = "MEDIUM"
        desc = (
            f"Score: {score:.2f} — Partial match. The corpus contains related "
            f"material but may not address your exact question. Verify key claims."
        )
    else:
        label = "LOW"
        desc = (
            f"Score: {score:.2f} — Weak match. Retrieved passages may be tangentially "
            f"related. Consider rephrasing your question or checking alternative sources."
        )

    return score, label, desc


def get_badge_color(label: str) -> str:
    """Return a hex color for the Streamlit badge based on the label."""
    return {
        "HIGH":   "#22c55e",  # green
        "MEDIUM": "#f59e0b",  # amber
        "LOW":    "#ef4444",  # red
        "NONE":   "#6b7280",  # grey
    }.get(label, "#6b7280")


def get_badge_emoji(label: str) -> str:
    """Return an emoji icon for the label."""
    return {
        "HIGH":   "🟢",
        "MEDIUM": "🟡",
        "LOW":    "🔴",
        "NONE":   "⚪",
    }.get(label, "⚪")


TOOLTIP_DISCLAIMER = (
    "ℹ️ **Retrieval Strength** is the mean cosine similarity between your query and the "
    "retrieved passages. It is **not** a calibrated confidence probability — it measures "
    "how closely the corpus text matches your question, not the correctness of the answer."
)
