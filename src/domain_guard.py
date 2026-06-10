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
