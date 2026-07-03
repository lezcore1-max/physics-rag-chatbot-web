"""
src/domain_guard.py
Domain guard + input sanitiser.
- sanitise_query(): strips control chars, truncates to MAX_QUERY_CHARS
- is_out_of_scope(): embeds the query and checks ChromaDB top-1 cosine similarity
  If score < DOMAIN_THRESHOLD → refuse (OOS) without calling LLM
"""

import re
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DOMAIN_THRESHOLD, MAX_QUERY_CHARS


def sanitise_query(query: str) -> str:
    """Strip control characters and truncate long queries."""
    # Strip leading/trailing whitespace
    query = query.strip()
    # Remove control characters (keep printable + newline/tab)
    query = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", query)
    # Collapse multiple spaces
    query = re.sub(r" {2,}", " ", query)
    # Truncate at MAX_QUERY_CHARS
    if len(query) > MAX_QUERY_CHARS:
        query = query[:MAX_QUERY_CHARS]
    # Validate minimum length
    if len(query.strip()) < 3:
        raise ValueError("Query too short (minimum 3 characters).")
    return query


REFUSAL_MESSAGE = (
    "I specialise in undergraduate physics topics covered by the OpenStax University "
    "Physics series and the Feynman Lectures. Your question doesn't seem to fall within "
    "this scope — I couldn't find closely matching content in my physics corpus.\n\n"
    "If you believe this IS a physics question, try rephrasing it with more physics "
    "terminology (e.g. specific laws, equations, or phenomena)."
)

PHYSICS_KEYWORDS = {
    "force", "energy", "mass", "velocity", "acceleration", "momentum",
    "newton", "einstein", "quantum", "wave", "field", "electric", "magnetic",
    "gravity", "thermodynamics", "entropy", "photon", "electron", "proton",
    "nuclear", "relativity", "optics", "mechanics", "oscillation", "resonance",
    "capacitor", "resistor", "inductance", "coulomb", "faraday", "maxwell",
    "schrodinger", "planck", "boltzmann", "feynman", "bohr", "angular",
    "torque", "friction", "pressure", "temperature", "heat", "potential",
    "kinetic", "electromagnetic", "spectrum", "frequency", "wavelength",
    "amplitude", "interference", "diffraction", "polarization", "radioactive",
    "fission", "fusion", "spin", "orbital", "atom", "molecule", "gas", "fluid",
    "viscosity", "bernoulli", "thermite", "voltage", "current", "circuit",
    "power", "watt", "joule", "scattering", "elastic", "inelastic", "uncertainty"
}

# Topics that are definitively NOT physics — used in Stage 1b blocklist.
# Only fires when NONE of the PHYSICS_KEYWORDS matched.
NON_PHYSICS_KEYWORDS = {
    # Biology / Chemistry (non-physics)
    "dna", "rna", "gene", "genetics", "chromosome", "protein", "enzyme",
    "photosynthesis", "mitosis", "meiosis", "evolution", "natural selection",
    "combustion", "stoichiometry", "oxidation", "reduction", "mole",
    "titration", "acid", "base", "ph", "buffer", "catalyst",
    # Climate / Earth Sciences
    "climate change", "global warming", "greenhouse", "carbon dioxide",
    "ozone", "deforestation", "biodiversity", "ecosystem",
    # Economics / Finance
    "stock", "bond", "inflation", "gdp", "interest rate", "black-scholes",
    "option pricing", "derivative", "portfolio", "hedge fund",
    # History / Social Sciences
    "world war", "revolution", "democracy", "colonialism", "trade route",
    "religion", "philosophy", "sociology", "psychology", "linguistics",
    # Medicine / Nutrition
    "diagnosis", "symptom", "antibiotic", "vaccine", "surgery",
    "calorie", "vitamin", "nutrition", "diet", "metabolism",
    # Cooking
    "recipe", "ingredient", "baking", "cuisine",
    # Computing (non-physics)
    "neural network", "machine learning", "deep learning", "algorithm",
    "database", "software", "programming", "html", "javascript",
}


class DomainGuard:
    """
    Two-stage domain guard:
    1. Fast keyword check — if physics keywords present, pass immediately
    2. Embedding similarity check against ChromaDB — if score < threshold, refuse
    """

    def __init__(self, vectorstore=None):
        """
        Args:
            vectorstore: ChromaDB vectorstore instance (optional — enables embedding check).
                         If None, only keyword check is performed.
        """
        self._vectorstore = vectorstore

    def check(self, query: str) -> tuple[bool, float]:
        """
        Returns (is_oos, score) where:
        - is_oos=True  → refuse this query
        - is_oos=False → allow
        - score        → cosine similarity from ChromaDB (0 if keyword-passed)
        """
        query_lower = query.lower()

        # Stage 1a: Fast physics-keyword pass — if physics terms present, always allow
        if any(re.search(rf"\b{re.escape(kw)}\b", query_lower) for kw in PHYSICS_KEYWORDS):
            return False, 1.0  # Not OOS, high confidence

        # Stage 1b: Non-physics blocklist — explicit OOS topics refused immediately
        # (only reached when NO physics keyword matched above)
        for phrase in NON_PHYSICS_KEYWORDS:
            # Use word-boundary match for single words, substring for multi-word phrases
            if " " in phrase:
                if phrase in query_lower:
                    return True, 0.0  # Hard OOS
            else:
                if re.search(rf"\b{re.escape(phrase)}\b", query_lower):
                    return True, 0.0  # Hard OOS

        # Stage 2: Embedding similarity check
        if self._vectorstore is not None:
            try:
                results = self._vectorstore.similarity_search_with_relevance_scores(
                    query, k=1
                )
                if results:
                    score = results[0][1]
                    return score < DOMAIN_THRESHOLD, score
                else:
                    # Empty corpus — let it through
                    return False, 0.0
            except Exception:
                # If embedding fails, don't block the user
                return False, 0.0

        # No vectorstore and no keyword match — conservative pass
        return False, 0.5


# ── Threshold Calibration ──────────────────────────────────────────────────

# 11 in-domain physics queries + 10 out-of-domain queries used for calibration.
# Keep this list stable so threshold changes can be compared consistently.
TEST_QUERIES = [
    # ─ In-domain (expected: in_domain) ───────────────────────────────
    {"query": "What is Newton's second law?", "expected": "in_domain"},
    {"query": "Explain the photoelectric effect.", "expected": "in_domain"},
    {"query": "Derive the equation for simple harmonic motion.", "expected": "in_domain"},
    {"query": "What is the uncertainty principle?", "expected": "in_domain"},
    {"query": "How does a capacitor store energy?", "expected": "in_domain"},
    {"query": "What is entropy in thermodynamics?", "expected": "in_domain"},
    {"query": "Explain Faraday's law of induction.", "expected": "in_domain"},
    {"query": "What is the Bohr model of the hydrogen atom?", "expected": "in_domain"},
    {"query": "How does nuclear fission release energy?", "expected": "in_domain"},
    # Edge cases — short or ambiguous but still physics
    {"query": "what is energy?", "expected": "in_domain"},
    {"query": "what is force?", "expected": "in_domain"},
    # ─ Out-of-domain (expected: out_of_domain) ──────────────────────
    {"query": "How does DNA replication work?", "expected": "out_of_domain"},
    {"query": "What causes global warming?", "expected": "out_of_domain"},
    {"query": "Explain the Black-Scholes equation.", "expected": "out_of_domain"},
    {"query": "How do deep neural networks learn via backpropagation?", "expected": "out_of_domain"},
    {"query": "What is the chemistry of combustion?", "expected": "out_of_domain"},
    {"query": "How does photosynthesis work?", "expected": "out_of_domain"},
    {"query": "What is GDP?", "expected": "out_of_domain"},
    {"query": "Explain the French Revolution.", "expected": "out_of_domain"},
    # Edge cases — ambiguous short queries that are NOT physics
    {"query": "what is a meme?", "expected": "out_of_domain"},
    {"query": "how do vaccines work?", "expected": "out_of_domain"},
]


def evaluate_threshold(
    threshold: float,
    test_queries: list,
    vectorstore=None,
) -> dict:
    """
    Run the domain guard on test_queries at a given threshold and measure accuracy.

    Args:
        threshold:    Cosine similarity cutoff to evaluate.
        test_queries: List of dicts with keys "query" and "expected" ("in_domain" | "out_of_domain").
        vectorstore:  ChromaDB instance for embedding-based Stage 2 checks.
                      If None, only the keyword stages fire.

    Returns:
        dict with keys:
            threshold         — the evaluated value
            false_refusal_rate — fraction of in-domain queries incorrectly refused
            false_accept_rate  — fraction of out-of-domain queries incorrectly accepted
            accuracy           — overall fraction correctly classified
    """
    from config import DOMAIN_THRESHOLD as _orig_threshold

    guard = DomainGuard(vectorstore=vectorstore)
    # Temporarily monkey-patch the module-level threshold used in check()
    import src.domain_guard as _self_mod
    import config as _cfg
    original = _cfg.DOMAIN_THRESHOLD
    _cfg.DOMAIN_THRESHOLD = threshold

    results = []
    for item in test_queries:
        is_oos, score = guard.check(item["query"])
        predicted = "out_of_domain" if is_oos else "in_domain"
        correct = predicted == item["expected"]
        results.append({
            "query":     item["query"],
            "expected":  item["expected"],
            "predicted": predicted,
            "score":     round(score, 4),
            "correct":   correct,
        })

    # Restore original threshold
    _cfg.DOMAIN_THRESHOLD = original

    in_domain_queries  = [r for r in results if r["expected"] == "in_domain"]
    out_domain_queries = [r for r in results if r["expected"] == "out_of_domain"]

    false_refusals = [r for r in in_domain_queries  if not r["correct"]]
    false_accepts  = [r for r in out_domain_queries if not r["correct"]]

    false_refusal_rate = len(false_refusals) / len(in_domain_queries)  if in_domain_queries  else 0.0
    false_accept_rate  = len(false_accepts)  / len(out_domain_queries) if out_domain_queries else 0.0
    accuracy           = sum(r["correct"] for r in results) / len(results) if results else 0.0

    return {
        "threshold":          threshold,
        "false_refusal_rate": round(false_refusal_rate, 4),
        "false_accept_rate":  round(false_accept_rate,  4),
        "accuracy":           round(accuracy, 4),
        "details":            results,
    }


if __name__ == "__main__":
    from config import CHROMA_DIR

    # Guard: calibration requires a built vectorstore.
    if not os.path.exists(CHROMA_DIR):
        print(
            f"ERROR: Vectorstore not found at '{CHROMA_DIR}'.\n"
            "Run 'python src/ingest.py' first to build the ChromaDB, "
            "then re-run this script."
        )
        sys.exit(1)

    print("Loading vectorstore for calibration...")
    from langchain_chroma import Chroma
    from src.embeddings import LocalSentenceTransformerEmbeddings
    from config import EMBED_MODEL

    embeddings = LocalSentenceTransformerEmbeddings(model_name=EMBED_MODEL)
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

    print(f"Vectorstore loaded. Evaluating {len(TEST_QUERIES)} test queries...\n")
    print(f"{'Threshold':>10}  {'Accuracy':>10}  {'False-Refusal':>15}  {'False-Accept':>14}")
    print("-" * 56)

    for thresh in [0.25, 0.35, 0.42, 0.50]:
        report = evaluate_threshold(thresh, TEST_QUERIES, vectorstore=vectorstore)
        print(
            f"{thresh:>10.2f}  "
            f"{report['accuracy']:>10.1%}  "
            f"{report['false_refusal_rate']:>15.1%}  "
            f"{report['false_accept_rate']:>14.1%}"
        )

    print("\nDetailed results for current DOMAIN_THRESHOLD (0.42):")
    detailed = evaluate_threshold(0.42, TEST_QUERIES, vectorstore=vectorstore)
    for item in detailed["details"]:
        status = "✅" if item["correct"] else "❌"
        print(f"  {status}  [{item['expected']:>12}]  score={item['score']:.3f}  {item['query']!r}")
