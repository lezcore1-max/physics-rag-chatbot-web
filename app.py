"""
app.py
Streamlit UI for the Physics RAG Chatbot.
Provides a premium, interactive interface with real-time streaming answers,
retrieval strength scoring, citation mapping, and source inspections.
"""

import os
import sys
import time
import math
from typing import Generator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    APP_TITLE, APP_ICON, LLM_MODEL, EMBED_MODEL, TOP_K, FINAL_K, BM25_CACHE
)
from src.retriever import get_ensemble_retriever
from src.llm_chain import init_llm, query_pipeline, get_citation_tag
from src.domain_guard import DomainGuard
from src.confidence import get_badge_color, get_badge_emoji, TOOLTIP_DISCLAIMER

import streamlit as st
import pickle
import re

# ── LaTeX Macro Preamble ─────────────────────────────────────────────────────
# DESIGN NOTE — KaTeX isolation:
# Streamlit uses KaTeX for st.latex() and $$...$$ in st.markdown().
# KaTeX renders each block independently — unlike MathJax there is NO shared
# macro registry across render calls.  A hidden <div>\(macros\)</div> injected
# once at startup would work for MathJax but is silently ignored by KaTeX.
# The only reliable approach is to prepend LATEX_MACROS into EVERY display-math
# and inline-math string before it is passed to st.latex() / st.markdown().
# This is verbose but correct for the KaTeX version Streamlit ships.
LATEX_MACROS = r"""
\gdef\FLPdiv{\boldsymbol{\nabla}\cdot}
\gdef\FLPgrad{\boldsymbol{\nabla}}
\gdef\FLPcurl{\boldsymbol{\nabla}\times}
\gdef\FLPnabla{\boldsymbol{\nabla}}
\gdef\FLPlap{\nabla^2}

\gdef\ddp#1#2{\frac{\partial #1}{\partial #2}}
\gdef\ddpl#1#2{\frac{\partial #1}{\partial #2}}
\gdef\ddt#1#2{\frac{d #1}{d #2}}
\gdef\ddtl#1#2{\frac{d #1}{d #2}}

% Vector single letters
\gdef\FLPA{\mathbf{A}}
\gdef\FLPB{\mathbf{B}}
\gdef\FLPC{\mathbf{C}}
\gdef\FLPD{\mathbf{D}}
\gdef\FLPE{\mathbf{E}}
\gdef\FLPF{\mathbf{F}}
\gdef\FLPH{\mathbf{H}}
\gdef\FLPI{\mathbf{I}}
\gdef\FLPJ{\mathbf{J}}
\gdef\FLPL{\mathbf{L}}
\gdef\FLPM{\mathbf{M}}
\gdef\FLPP{\mathbf{P}}
\gdef\FLPR{\mathbf{R}}
\gdef\FLPS{\mathbf{S}}
\gdef\FLPU{\mathbf{U}}

\gdef\FLPa{\mathbf{a}}
\gdef\FLPb{\mathbf{b}}
\gdef\FLPc{\mathbf{c}}
\gdef\FLPd{\mathbf{d}}
\gdef\FLPe{\mathbf{e}}
\gdef\FLPf{\mathbf{f}}
\gdef\FLPg{\mathbf{g}}
\gdef\FLPh{\mathbf{h}}
\gdef\FLPi{\mathbf{i}}
\gdef\FLPj{\mathbf{j}}
\gdef\FLPk{\mathbf{k}}
\gdef\FLPn{\mathbf{n}}
\gdef\FLPp{\mathbf{p}}
\gdef\FLPr{\mathbf{r}}
\gdef\FLPs{\mathbf{s}}
\gdef\FLPu{\mathbf{u}}
\gdef\FLPv{\mathbf{v}}
\gdef\FLPw{\mathbf{w}}
\gdef\FLPx{\mathbf{x}}

% Numbers
\gdef\FLPzero{\mathbf{0}}
\gdef\FLPzeroi{\mathbf{0}_i}
\gdef\FLPone{\mathbf{1}}
\gdef\FLPtwo{\mathbf{2}}

% Greek letters
\gdef\FLPOmega{\boldsymbol{\Omega}}
\gdef\FLPomega{\boldsymbol{\omega}}
\gdef\FLPdelta{\boldsymbol{\delta}}
\gdef\FLPmu{\boldsymbol{\mu}}
\gdef\FLPsigma{\boldsymbol{\sigma}}
\gdef\FLPsigmae{\boldsymbol{\sigma}_e}
\gdef\FLPsigmaop{\boldsymbol{\sigma}_{\text{op}}}
\gdef\FLPsigmap{\boldsymbol{\sigma}_p}
\gdef\FLPtau{\boldsymbol{\tau}}
\gdef\FLPRe{\mathbf{Re}}
"""

# Compact single-line version of the macros for use inside st.markdown() inline
# math blocks ($...$).  st.markdown() only treats $...$ as inline KaTeX math
# when the content sits on ONE line.  The full LATEX_MACROS has % comment lines
# and blank lines which cause the markdown parser to bail out and emit raw text.
# Stripping comments/blanks and joining to a single line fixes that.
# LATEX_MACROS (multiline) is still used for st.latex() where it works fine.
LATEX_MACROS_INLINE = " ".join(
    line.strip()
    for line in LATEX_MACROS.splitlines()
    if line.strip() and not line.strip().startswith("%")
)

# ── LaTeX Rendering Helper ─────────────────────────────────────────────────────
def render_response_with_latex(text: str):
    """
    Render an LLM response with properly displayed LaTeX equations.
    - Display math \\[...\\], $$...$$, or \\begin{env}...\\end{env} → st.latex() (centered, KaTeX-rendered)
    - Inline  math \\(...\\) → $...$ (with prepended macros) inside st.markdown()
    Falls back to plain st.markdown() if no math is detected.

    LATEX_MACROS (defined at module level above) are prepended into every math
    block individually — this is required because Streamlit's KaTeX renderer
    isolates each render call and does not share \\gdef definitions across blocks.
    """

    def clean_math(m_str: str) -> str:
        # Strip \label{...} as KaTeX doesn't support it natively and it causes issues
        m_str = re.sub(r'\\label\{.*?\}', '', m_str)
        return m_str

    # Pattern for display math:
    DISPLAY_ENVS = r'equation|align|gather|multline'
    pattern_bracket = re.compile(r'\\\[(.*?)\\\]', re.DOTALL)
    pattern_dollars = re.compile(r'\$\$(.*?)\$\$', re.DOTALL)
    pattern_env     = re.compile(
        r'\\begin\{(' + DISPLAY_ENVS + r')\*?\}(.*?)\\end\{\1\*?\}',
        re.DOTALL
    )

    all_matches = []
    for m in pattern_bracket.finditer(text):
        all_matches.append(('bracket', m))
    for m in pattern_dollars.finditer(text):
        all_matches.append(('dollars', m))
    for m in pattern_env.finditer(text):
        all_matches.append(('env', m))
    all_matches.sort(key=lambda x: x[1].start())

    filtered_matches = []
    last_match_end = 0
    for kind, m in all_matches:
        if m.start() >= last_match_end:
            filtered_matches.append((kind, m))
            last_match_end = m.end()

    segments = []
    last_end = 0

    for kind, match in filtered_matches:
        before = text[last_end:match.start()]
        if before.strip():
            segments.append(('text', before))
        
        if kind == 'bracket':
            math_content = clean_math(match.group(1).strip())
        elif kind == 'dollars':
            math_content = clean_math(match.group(1).strip())
        else:
            env_name = match.group(1)
            cleaned_inner = clean_math(match.group(2).strip())
            if 'align' in env_name:
                math_content = f"\\begin{{aligned}}\n{cleaned_inner}\n\\end{{aligned}}"
            else:
                math_content = cleaned_inner
            
        if math_content:
            segments.append(('math', LATEX_MACROS + "\n" + math_content))
            
        last_end = match.end()

    remaining = text[last_end:]
    if remaining.strip():
        segments.append(('text', remaining))

    # Nothing to split — check for inline math in plain markdown
    if not segments:
        if '\\(' in text:
            # Embed macros INSIDE each $...$ expression so KaTeX sees them in
            # the same render call. A separate $LATEX_MACROS$ preamble token
            # does NOT work — KaTeX ignores \gdef across isolated render calls,
            # and a multiline string with % comments never gets parsed as math;
            # it leaks as visible text instead.
            processed_text = re.sub(
                r'\\\((.*?)\\\)',
                lambda m: f"${LATEX_MACROS_INLINE} {clean_math(m.group(1))}$",
                text,
                flags=re.DOTALL
            )
            st.markdown(processed_text)
        else:
            st.markdown(text)
        return

    for seg_type, content in segments:
        if seg_type == 'math':
            st.latex(content)
        else:
            if '\\(' in content:
                # Same fix: macros go inside each $...$ so they are in scope
                # for that specific KaTeX render call.
                processed = re.sub(
                    r'\\\((.*?)\\\)',
                    lambda m: f"${LATEX_MACROS_INLINE} {clean_math(m.group(1))}$",
                    content,
                    flags=re.DOTALL
                )
                if processed.strip():
                    st.markdown(processed)
            else:
                if content.strip():
                    st.markdown(content)


# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load Custom Styling ───────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Premium Header */
    .header-container {
        padding: 20px 0;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
        margin-bottom: 25px;
    }
    
    .sidebar-title {
        font-size: 22px;
        font-weight: 700;
        background: linear-gradient(135deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Styled badges */
    .badge-container {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 12px;
    }
    
    .custom-badge {
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        display: inline-flex;
        align-items: center;
        gap: 4px;
        color: white;
    }
    
    /* Glassmorphism for sources */
    .source-card {
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.15);
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
    }
    
    .source-header {
        font-weight: 600;
        color: #60a5fa;
        font-size: 14px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 4px;
    }
    
    .source-metadata {
        font-size: 11px;
        color: #9ca3af;
        margin-bottom: 8px;
        display: flex;
        gap: 12px;
    }
    
    .degraded-warning {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.3);
        color: #f87171;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 10px;
        font-weight: 600;
        text-transform: uppercase;
    }
    
    .source-text {
        font-size: 13px;
        line-height: 1.6;
        color: #d1d5db;
        word-break: break-word;
        overflow-wrap: anywhere;
        white-space: normal;
        max-width: 100%;
    }

    .source-card .source-text p { margin: 0; }

</style>
""", unsafe_allow_html=True)


# ── Load Resources (Cached) ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Initializing Physics Models & Retriever on GPU...")
def load_resources():
    """Initialise and cache LLM, Retriever, and Domain Guard."""
    llm = init_llm()
    retriever = get_ensemble_retriever()
    
    vectorstore = None
    if retriever is not None:
        try:
            # Safely extract Chroma DB vector store for the DomainGuard
            base_ret = getattr(retriever, "base_retriever", retriever)
            if hasattr(base_ret, "retrievers") and len(base_ret.retrievers) > 1:
                vectorstore = getattr(base_ret.retrievers[1], "vectorstore", None)
        except Exception as e:
            print(f"Error extracting vectorstore: {e}")
            
    domain_guard = DomainGuard(vectorstore=vectorstore)
    return llm, retriever, domain_guard


# ── Helper functions ──────────────────────────────────────────────────────────
def get_corpus_chunks_count() -> int:
    """Read the number of documents cached in the BM25 index."""
    if os.path.exists(BM25_CACHE):
        try:
            with open(BM25_CACHE, "rb") as f:
                bm25_data = pickle.load(f)
            return len(bm25_data.get("texts", []))
        except Exception:
            return 0
    return 0


# Load system components
with st.spinner("Initialising RAG engine & loading database..."):
    llm, retriever, domain_guard = load_resources()

# Calculate stats
corpus_chunks = get_corpus_chunks_count()
is_llm_running = llm is not None
is_retriever_online = retriever is not None

# Update sidebar stats in session state
if "queries_count" not in st.session_state:
    st.session_state.queries_count = 0
if "refused_count" not in st.session_state:
    st.session_state.refused_count = 0
if "avg_strength" not in st.session_state:
    st.session_state.avg_strength = 0.0
if "strength_scores" not in st.session_state:
    st.session_state.strength_scores = []
if "chat_history" not in st.session_state:
    # chat_history is the SINGLE source of truth for both UI display and LLM context.
    # It is a list of (user_query: str, assistant_response) tuples.
    # st.session_state.messages is NOT used; messages are derived on the fly below.
    st.session_state.chat_history = []


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">🔭 Physics RAG Chatbot</div>', unsafe_allow_html=True)
    
    # 1. System Status
    st.subheader("System Status")
    col1, col2 = st.columns(2)
    with col1:
        if is_llm_running:
            st.success("LLM: Running ✅")
        else:
            st.error("LLM: Offline ⚠️")
    with col2:
        if is_retriever_online:
            st.success("DB: Online ✅")
        else:
            st.warning("DB: Empty ⚠️")
            
    if not is_llm_running:
        st.info("💡 💡 LLM powered by Groq API. Check that GROQ_API_KEY is set in Streamlit secrets.")

    # 2. Corpus Details
    st.subheader("Physics Corpus")
    st.markdown(f"""
    - **OpenStax University Physics**: Volumes 1–3
    - **Feynman Lectures on Physics**: Volumes I–III
    - **Ingested Database**: `{corpus_chunks:,}` chunks
    """)
    
    # 3. Settings Summary
    st.subheader("Parameters")
    st.markdown(f"""
    - **LLM Model**: `{LLM_MODEL}` (temp=0)
    - **Embed Model**: `{EMBED_MODEL}`
    - **Retriever**: Hybrid (BM25 + MMR Semantic)
    - **Reranker**: ms-marco CrossEncoder
    - **Top-K Chunks**: `{TOP_K} → {FINAL_K} (Reranked)`
    """)

    # 4. Session Statistics (wrapped in a placeholder so it updates in-place
    #    after query completion without requiring a full page rerun)
    st.subheader("Session Stats")
    stats_placeholder = st.sidebar.empty()
    with stats_placeholder.container():
        st.markdown(f"""
    - **Queries Processed**: `{st.session_state.queries_count}`
    - **OOS Refusals**: `{st.session_state.refused_count}`
    - **Avg Retrieval Strength**: `{st.session_state.avg_strength:.2%}`
        """)
    
    st.divider()
    
    # 5. Actions
    col_clear, col_index = st.columns(2)
    with col_clear:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            # Reset all sidebar stats so they reflect the new fresh session.
            st.session_state.queries_count = 0
            st.session_state.refused_count = 0
            st.session_state.avg_strength = 0.0
            st.session_state.strength_scores = []
            # NOTE: No LLM memory object to clear.
            # query_pipeline() in src/llm_chain.py accepts chat_history as a plain
            # list argument each call — there is no ConversationBufferMemory.
            # Clearing chat_history is the single, complete reset for LLM context.
            st.rerun()
            
    with col_index:
        if st.button("🔄 Re-index", use_container_width=True):
            st.warning("To re-ingest corpus, please run the ingestion script in your terminal:")
            st.code("python src/ingest.py --reset", language="bash")


# ── Main Chat Area ───────────────────────────────────────────────────────────
st.title("Physics Tutor Chatbot")
st.caption("Ask questions on classical mechanics, electromagnetism, optics, thermodynamics, waves, and quantum/nuclear physics.")

# Derive the display message list from chat_history (single source of truth).
# Each tuple in chat_history is (user_query, assistant_response).
# assistant_response is either a plain str (refusals/greetings) or a dict
# carrying {"content", "citations", "retrieval_strength", "refused"} for full answers.
import html as html_lib
for human, assistant in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(human)
    with st.chat_message("assistant"):
        if isinstance(assistant, dict):
            refused = assistant.get("refused", False)
            if not refused:
                # Re-render badge
                score, label, desc = assistant["retrieval_strength"]
                badge_color = get_badge_color(label)
                emoji = get_badge_emoji(label)
                st.markdown(f"""
                <div class="badge-container">
                    <span class="custom-badge" style="background-color: {badge_color};" title="{desc}">
                        {emoji} Retrieval Strength: {label} ({score:.2f})
                    </span>
                </div>
                """, unsafe_allow_html=True)
            render_response_with_latex(assistant["content"])
            citations = assistant.get("citations", [])
            if citations and not refused:
                with st.expander("📚 View Source Snippets"):
                    for cite in citations:
                        warn_html = (
                            '<span class="degraded-warning">⚠️ Degraded Equations</span>'
                            if cite.get("equation_quality") == "degraded" else ""
                        )
                        tag     = html_lib.escape(str(cite.get('tag', '')))
                        topic   = html_lib.escape(str(cite.get('topic', '')))
                        fmt     = html_lib.escape(str(cite.get('type', '')).capitalize())
                        content = html_lib.escape(str(cite.get('content', '')))
                        st.markdown(f"""
<div class="source-card">
  <div class="source-header">
    <span>{tag}</span>
    {warn_html}
  </div>
  <div class="source-metadata">
    <span><b>Topic:</b> {topic}</span>
    <span><b>Format:</b> {fmt}</span>
  </div>
  <div class="source-text">{content}</div>
</div>
""", unsafe_allow_html=True)
        else:
            st.markdown(str(assistant))


# ── Query Submission ───────────────────────────────────────────────────────────
# Disable chat input if models are offline
input_placeholder = "Ask a physics question..." if is_retriever_online else "Database is offline. Run ingestion first."
user_query = st.chat_input(placeholder=input_placeholder, disabled=not is_retriever_online)

if user_query:
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(user_query)

    st.session_state.queries_count += 1

    # Run the pipeline
    with st.spinner("Searching corpus & generating answer..."):
        is_refused, response_dict, prompt, citations, strength_meta = query_pipeline(
            user_query, retriever, domain_guard,
            chat_history=st.session_state.chat_history,
            llm=llm
        )
        
        # Inject chat history into prompt for memory
        if not is_refused and prompt and st.session_state.chat_history:
            history_text = "\n".join([
                f"Student: {h}\nTutor: {a['content'] if isinstance(a, dict) else a}"
                for h, a in st.session_state.chat_history[-5:]  # last 5 turns
            ])
            prompt = f"Previous conversation:\n{history_text}\n\n{prompt}"
        
    # Check if refused or empty retrieval
    if is_refused:
        st.session_state.refused_count += 1
        with st.chat_message("assistant"):
            st.markdown(response_dict["answer"])
        # Record in chat_history (plain string for refusals)
        st.session_state.chat_history.append((user_query, response_dict["answer"]))
        # Refresh sidebar stats in-place
        with stats_placeholder.container():
            st.markdown(f"""
    - **Queries Processed**: `{st.session_state.queries_count}`
    - **OOS Refusals**: `{st.session_state.refused_count}`
    - **Avg Retrieval Strength**: `{st.session_state.avg_strength:.2%}`
        """)
    else:
        # Valid physics query: execute streaming LLM response
        with st.chat_message("assistant"):
            # 1. Render Retrieval Strength Badge
            score, label, desc = strength_meta
            badge_color = get_badge_color(label)
            emoji = get_badge_emoji(label)
            
            st.markdown(f"""
            <div class="badge-container">
                <span class="custom-badge" style="background-color: {badge_color};" title="{desc}">
                    {emoji} Retrieval Strength: {label} ({score:.2f})
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            # Update average strength
            st.session_state.strength_scores.append(score)
            st.session_state.avg_strength = sum(st.session_state.strength_scores) / len(st.session_state.strength_scores)
            
            # 2. Stream answer content
            answer_placeholder = st.empty()
            
            # Define response token generator
            if is_llm_running:
                try:
                    response_generator = llm.stream(prompt)
                    full_response = ""
                    for chunk in response_generator:
                        token = chunk.content if hasattr(chunk, "content") else str(chunk)
                        full_response += token
                        answer_placeholder.markdown(full_response + " ▌")
                    # Clear streaming placeholder then render with proper LaTeX
                    answer_placeholder.empty()
                    render_response_with_latex(full_response)
                except Exception as e:
                    full_response = f"LLM streaming failed: {e}"
                    answer_placeholder.error(full_response)
            else:
                full_response = "⚠️  LLM is offline. Could not generate the final response. Please read the retrieved sources below."
                answer_placeholder.warning(full_response)
                
            # 3. Render Citations expander
            if citations:
                import html as html_lib
                with st.expander("📚 View Source Snippets"):
                    for cite in citations:
                        warn_html = (
                            '<span class="degraded-warning">⚠️ Degraded Equations</span>'
                            if cite.get("equation_quality") == "degraded" else ""
                        )
                        # Escape all dynamic values to prevent raw HTML leaking
                        tag      = html_lib.escape(str(cite.get('tag', '')))
                        topic    = html_lib.escape(str(cite.get('topic', '')))
                        fmt      = html_lib.escape(str(cite.get('type', '')).capitalize())
                        content  = html_lib.escape(str(cite.get('content', '')))

                        st.markdown(f"""
<div class="source-card">
  <div class="source-header">
    <span>{tag}</span>
    {warn_html}
  </div>
  <div class="source-metadata">
    <span><b>Topic:</b> {topic}</span>
    <span><b>Format:</b> {fmt}</span>
  </div>
  <div class="source-text">{content}</div>
</div>
""", unsafe_allow_html=True)
                        
            # 4. Save to chat_history — dict payload so history renderer can
            #    re-display badge + citations on subsequent renders.
            #    (st.session_state.messages is NOT used — chat_history is SSoT)
            st.session_state.chat_history.append((
                user_query,
                {
                    "content":            full_response,
                    "citations":          citations,
                    "retrieval_strength": strength_meta,
                    "refused":            False,
                }
            ))
            if len(st.session_state.chat_history) > 5:
                st.session_state.chat_history.pop(0)

            # 5. Refresh sidebar stats in-place — no st.rerun() needed.
            with stats_placeholder.container():
                st.markdown(f"""
    - **Queries Processed**: `{st.session_state.queries_count}`
    - **OOS Refusals**: `{st.session_state.refused_count}`
    - **Avg Retrieval Strength**: `{st.session_state.avg_strength:.2%}`
                """)
