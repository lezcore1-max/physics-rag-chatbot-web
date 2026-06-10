"""
src/llm_chain.py
LLM Chain for Physics RAG Chatbot.
Handles prompting, ChatOllama query execution, and streaming integration.
"""

import os
import sys
import math
from typing import List, Optional, Tuple, Dict, Any, Union

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS, OLLAMA_BASE_URL
)
from src.confidence import compute_retrieval_strength
from src.domain_guard import sanitise_query, REFUSAL_MESSAGE

from langchain_core.documents import Document
from langchain_ollama import ChatOllama

# ── RAG Prompt Template ───────────────────────────────────────────────────────
RAG_PROMPT_TEMPLATE = """You are a physics tutor. Answer the student's question STRICTLY using ONLY the provided context below.

STRICT RULES — follow every rule exactly:
1. Cite every single fact using the EXACT citation tag from the context (e.g. [Source: OpenStax Vol 1, p.24]). Place the citation tag immediately after the claim.
2. If the context does not contain enough information to answer the question, respond EXACTLY with: "I cannot find this in my physics corpus." Do NOT use any external knowledge or memory.
3. NEVER invent, guess, or recall formulas, constants, equations, or derivations. ONLY write formulas that appear word-for-word in the context below.
4. NEVER add information beyond what is in the context, even if you believe it to be correct.
5. Format equations using LaTeX: inline as \\(F = ma\\), block equations as \\[E = mc^2\\].
6. Be concise, pedagogically clear, and professional.

Context (use ONLY this — nothing else):
{context_text}

Student Question: {question}

Answer (strictly from context, with citations):"""


def init_llm() -> Optional[ChatOllama]:
    """
    Initialise the ChatOllama model and verify connectivity.
    Returns None if Ollama is not running or the model is not found.
    """
    try:
        llm = ChatOllama(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            num_predict=LLM_MAX_TOKENS,
            base_url=OLLAMA_BASE_URL,
        )
        # Test connection with a short query
        llm.invoke("ping")
        return llm
    except Exception as e:
        print(f"WARNING: Could not connect to Ollama model '{LLM_MODEL}': {e}")
        return None


def get_citation_tag(doc: Document) -> str:
    """Construct a clean, standard citation tag for the document."""
    meta = doc.metadata
    doc_type = meta.get("type", "openstax")
    
    if doc_type == "feynman":
        vol = meta.get("volume", "Vol I")
        ch = meta.get("chapter", "")
        # Extract Roman numeral from volume if needed
        vol_roman = vol.replace("Feynman Vol ", "")
        return f"[Source: Feynman Vol {vol_roman}, Ch.{ch}]"
    else:
        # OpenStax
        source = meta.get("source", "openstax_vol1")
        vol_num = source[-1] if source[-1].isdigit() else "1"
        page = meta.get("page", 1)
        return f"[Source: OpenStax Vol {vol_num}, p.{page}]"


def format_citations_metadata(docs: List[Document]) -> List[Dict[str, Any]]:
    """Convert LangChain Document list to standard citations metadata dicts."""
    citations = []
    for doc in docs:
        meta = doc.metadata
        tag = get_citation_tag(doc)
        citations.append({
            "tag": tag,
            "content": doc.page_content,
            "source": meta.get("source", ""),
            "type": meta.get("type", "openstax"),
            "topic": meta.get("topic", "General Physics"),
            "equation_quality": meta.get("equation_quality", "clean"),
            "page": meta.get("page"),
            "chapter": meta.get("chapter"),
            "title": meta.get("title", ""),
        })
    return citations


def query_pipeline(
    query: str, 
    retriever, 
    domain_guard=None
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str], Optional[List[Dict[str, Any]]], Optional[Tuple[float, str, str]]]:
    """
    First part of the RAG pipeline:
    - Sanitisation
    - Out-Of-Scope Guard
    - Document retrieval
    - Empty retrieval check
    - Retrieval strength calculation
    
    Returns:
        (is_refused, response_dict_if_refused, prompt, citations, strength_meta)
    """
    # 1. Sanitise
    try:
        sanitised = sanitise_query(query)
    except ValueError as e:
        return True, {
            "answer": str(e),
            "citations": [],
            "retrieval_strength": "NONE",
            "retrieval_score": 0.0,
            "retrieval_description": "Query validation failed.",
            "refused": True,
        }, None, None, None

    # 2. OOS Guard
    if domain_guard is not None:
        is_oos, oos_score = domain_guard.check(sanitised)
        if is_oos:
            return True, {
                "answer": REFUSAL_MESSAGE,
                "citations": [],
                "retrieval_strength": "NONE",
                "retrieval_score": oos_score,
                "retrieval_description": f"Similarity score {oos_score:.2f} below domain threshold.",
                "refused": True,
            }, None, None, None

    # 3. Check retriever
    if retriever is None:
        return True, {
            "answer": "The physics retriever is offline. Please build the database first.",
            "citations": [],
            "retrieval_strength": "NONE",
            "retrieval_score": 0.0,
            "retrieval_description": "Retriever is not initialised.",
            "refused": True,
        }, None, None, None

    # 4. Retrieve documents
    try:
        docs = retriever.invoke(sanitised)
    except Exception as e:
        return True, {
            "answer": f"Retrieval failed: {e}",
            "citations": [],
            "retrieval_strength": "NONE",
            "retrieval_score": 0.0,
            "retrieval_description": "Error during retrieval.",
            "refused": True,
        }, None, None, None

    # 5. Empty retrieval guard
    if not docs:
        return True, {
            "answer": "My physics corpus returned no relevant passages for this query. The database may be empty or this topic is not covered. Please run: python src/ingest.py",
            "citations": [],
            "retrieval_strength": "NONE",
            "retrieval_score": 0.0,
            "retrieval_description": "No passages retrieved.",
            "refused": False,
        }, None, None, None

    # 6. Compute retrieval strength from CrossEncoder relevance scores
    scores = []
    for doc in docs:
        rel_score = doc.metadata.get("relevance_score", 0.0)
        # Map raw ms-marco logit to [0, 1] using sigmoid
        # ms-marco logits are roughly between -10 and 10
        mapped = 1.0 / (1.0 + math.exp(-rel_score)) if rel_score != 0.0 else 0.5
        scores.append(mapped)

    score, label, desc = compute_retrieval_strength(scores)
    citations = format_citations_metadata(docs)

    # 7. Build Prompt
    context_blocks = []
    for doc in docs:
        tag = get_citation_tag(doc)
        context_blocks.append(f"--- Citation Tag: {tag} ---\n{doc.page_content}")
    context_text = "\n\n".join(context_blocks)

    prompt = RAG_PROMPT_TEMPLATE.format(context_text=context_text, question=sanitised)

    return False, None, prompt, citations, (score, label, desc)


def execute_rag(
    query: str, 
    retriever, 
    llm, 
    domain_guard=None
) -> Dict[str, Any]:
    """
    Execute the full RAG pipeline synchronously.
    Used for testing and evaluation.
    """
    is_refused, response_dict, prompt, citations, strength_meta = query_pipeline(
        query, retriever, domain_guard
    )
    if is_refused:
        return response_dict

    if llm is None:
        return {
            "answer": "Ollama LLM is not running or model is not loaded.",
            "citations": citations,
            "retrieval_strength": strength_meta[1],
            "retrieval_score": strength_meta[0],
            "retrieval_description": strength_meta[2],
            "refused": False,
        }

    try:
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)
        return {
            "answer": answer.strip(),
            "citations": citations,
            "retrieval_strength": strength_meta[1],
            "retrieval_score": strength_meta[0],
            "retrieval_description": strength_meta[2],
            "refused": False,
        }
    except Exception as e:
        return {
            "answer": f"LLM generation failed: {e}",
            "citations": citations,
            "retrieval_strength": strength_meta[1],
            "retrieval_score": strength_meta[0],
            "retrieval_description": strength_meta[2],
            "refused": False,
        }
