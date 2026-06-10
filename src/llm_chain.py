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
    LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS, GROQ_API_KEY
)
from src.confidence import compute_retrieval_strength
from src.domain_guard import sanitise_query, REFUSAL_MESSAGE

from langchain_core.documents import Document
from langchain_groq import ChatGroq

# ── RAG Prompt Template ───────────────────────────────────────────────────────
RAG_PROMPT_TEMPLATE = """You are a warm, helpful, and clear physics tutor. Answer the student's question step-by-step using ONLY the provided context below.

STRICT RULES — follow every rule exactly:
1. Explain the concepts in a natural, pedagogical, and tutoring voice (do not just stitch sentences word-for-word). Connect the ideas into a coherent flow.
2. Cite every fact using the EXACT citation tag from the context (e.g. [Source: OpenStax Vol 1, p.24]). Place the citation tag immediately after the claim.
3. If the context does not contain enough information to answer the question, respond EXACTLY with: "I cannot find this in my physics corpus." Do NOT use any external knowledge.
4. NEVER invent, guess, or recall formulas, constants, equations, or derivations. ONLY use formulas and numbers that appear in the context below.
5. Format equations using LaTeX: inline as \\(F = ma\\), block equations as \\[E = mc^2\\].
6. Keep your tutoring clear, concise, and helpful.

Context (use ONLY this — nothing else):
{context_text}

Student Question: {question}

Answer (strictly from context, with citations, written in a warm tutoring voice):"""

CONDENSE_PROMPT_TEMPLATE = """Given the following conversation history and a follow-up question, rephrase the follow-up question to be a standalone question that contains all the necessary context to be searched in a physics textbook. Do NOT answer the question.

Conversation History:
{chat_history}

Follow-up Question: {question}

Standalone Question (physics-focused):"""

GREETING_PROMPT_TEMPLATE = """You are a warm, friendly, and helpful physics tutor. Respond to the student's greeting or polite message in a natural, welcoming, and encouraging tutoring voice. Keep it brief (1-2 sentences) and invite them to ask a physics question.

Student Message: {question}

Tutor Response:"""


GREETINGS_KEYWORDS = {
    "hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening",
    "howdy", "hii", "hi there", "hello there", "yo", "hola", "hiii"
}
POLITE_REPLIES = {
    "thanks", "thank you", "ok thanks", "okay thanks", "perfect", "awesome", "great", "thanks!"
}

def is_greeting_or_smalltalk(query: str) -> Optional[str]:
    """Check if query is a greeting or polite reply. Returns type ('greeting', 'polite', or None)."""
    q = query.strip().lower().rstrip("!?.,")
    if q in GREETINGS_KEYWORDS:
        return "greeting"
    if q in POLITE_REPLIES:
        return "polite"
    return None

def format_chat_history(chat_history: List[Any], max_turns: int = 3) -> str:
    """Format the last few turns of chat history for query condensation."""
    if not chat_history:
        return ""
    turns = []
    # Check if first element is list/tuple
    if isinstance(chat_history[0], (list, tuple)):
        for h, a in chat_history[-max_turns:]:
            if "physics corpus" in str(a) or "specialise in undergraduate" in str(a):
                continue
            turns.append(f"Student: {h}\nTutor: {a}")
    else:
        for msg in chat_history[-max_turns*2:]:
            role = "Student" if msg["role"] == "user" else "Tutor"
            content = msg["content"]
            if "physics corpus" in content or "specialise in undergraduate" in content:
                continue
            turns.append(f"{role}: {content}")
    return "\n".join(turns)


def init_llm() -> Optional[ChatGroq]:
    try:
        import streamlit as st
        api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
    except Exception:
        api_key = os.getenv("GROQ_API_KEY", "")

    try:
        llm = ChatGroq(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            api_key=api_key,
        )
        llm.invoke("ping")
        return llm
    except Exception as e:
        print(f"WARNING: Could not connect to Groq model '{LLM_MODEL}': {e}")
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
    domain_guard=None,
    chat_history: Optional[List[Dict[str, str]]] = None,
    llm = None
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str], Optional[List[Dict[str, Any]]], Optional[Tuple[float, str, str]]]:
    """
    First part of the RAG pipeline:
    - Greetings / Small talk bypass
    - History-aware query condensation
    - Sanitisation
    - Out-Of-Scope Guard
    - Document retrieval
    - Empty retrieval check
    - Retrieval strength calculation
    
    Returns:
        (is_refused, response_dict_if_refused, prompt, citations, strength_meta)
    """
    # 0. Detect greetings or small talk
    g_type = is_greeting_or_smalltalk(query)
    if g_type is not None:
        prompt = GREETING_PROMPT_TEMPLATE.format(question=query)
        # Bypassed - no domain guard, no retrieval, return custom high strength
        return False, None, prompt, [], (1.0, "HIGH", "Tutor greeting bypass")

    # 1. Condense query if chat history exists
    search_query = query
    if chat_history and len(chat_history) >= 2 and llm is not None:
        try:
            formatted_history = format_chat_history(chat_history)
            if formatted_history:
                condense_prompt = CONDENSE_PROMPT_TEMPLATE.format(
                    chat_history=formatted_history,
                    question=query
                )
                condensed = llm.invoke(condense_prompt)
                condensed_text = condensed.content if hasattr(condensed, "content") else str(condensed)
                condensed_text = condensed_text.strip().strip('"\'')
                if len(condensed_text) > 3:
                    search_query = condensed_text
        except Exception as condense_err:
            print(f"WARNING: Query condensation failed: {condense_err}")

    # 2. Sanitise
    try:
        sanitised = sanitise_query(search_query)
    except ValueError as e:
        return True, {
            "answer": str(e),
            "citations": [],
            "retrieval_strength": "NONE",
            "retrieval_score": 0.0,
            "retrieval_description": "Query validation failed.",
            "refused": True,
        }, None, None, None

    # 3. OOS Guard
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

    # 4. Check retriever
    if retriever is None:
        return True, {
            "answer": "The physics retriever is offline. Please build the database first.",
            "citations": [],
            "retrieval_strength": "NONE",
            "retrieval_score": 0.0,
            "retrieval_description": "Retriever is not initialised.",
            "refused": True,
        }, None, None, None

    # 5. Retrieve documents
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

    # 6. Empty retrieval guard
    if not docs:
        return True, {
            "answer": "My physics corpus returned no relevant passages for this query. The database may be empty or this topic is not covered. Please run: python src/ingest.py",
            "citations": [],
            "retrieval_strength": "NONE",
            "retrieval_score": 0.0,
            "retrieval_description": "No passages retrieved.",
            "refused": False,
        }, None, None, None

    # 7. Compute retrieval strength from CrossEncoder relevance scores
    scores = []
    for doc in docs:
        rel_score = doc.metadata.get("relevance_score", 0.0)
        # Map raw ms-marco logit to [0, 1] using sigmoid
        mapped = 1.0 / (1.0 + math.exp(-rel_score)) if rel_score != 0.0 else 0.5
        scores.append(mapped)

    score, label, desc = compute_retrieval_strength(scores)
    citations = format_citations_metadata(docs)

    # 8. Build Prompt
    context_blocks = []
    for doc in docs:
        tag = get_citation_tag(doc)
        context_blocks.append(f"--- Citation Tag: {tag} ---\n{doc.page_content}")
    context_text = "\n\n".join(context_blocks)

    # Pass the original question to the RAG prompt so it answers the student's exact query
    original_sanitised = query.strip()
    prompt = RAG_PROMPT_TEMPLATE.format(context_text=context_text, question=original_sanitised)

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
        query, retriever, domain_guard, chat_history=None, llm=llm
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
