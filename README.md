# 🔭 Local Physics RAG Chatbot

An offline/hybrid Retrieval-Augmented Generation (RAG) system acting as an undergraduate-level physics tutor. Powered by **ChatGroq (`llama-3.1-8b-instant`)**, **LangChain**, local **ChromaDB** vector store, and **Streamlit**.

The corpus consists of **OpenStax University Physics Volumes 1–3** (PDFs) and **The Feynman Lectures on Physics Volumes I–III** (HTML scraped with preserved LaTeX math).

---

## 🏗️ Architecture

```mermaid
flowchart TD
    A[User Query] --> B[Input Sanitiser\ntruncate 500 chars]
    B --> C{Domain Guard\nscore < 0.25?}
    C -- OOS --> D[Polite Refusal\nno LLM called]
    C -- Physics --> E[MMR Ensemble Retriever\nBM25 0.5 + Semantic 0.5]
    E --> F[CrossEncoder Reranker\n7 → top 5 chunks]
    F --> G{Empty Guard\nchunks == 0?}
    G -- Empty --> H[Corpus Error Message]
    G -- OK --> I[Retrieval Strength Score\nmean cosine similarity]
    I --> J[RAG Prompt Builder\nchunks + query]
    J --> K[llama-3.1-8b-instant via Groq\ntemp=0 · max 512 tokens]
    K --> L[Citation Parser]
    L --> M[Streamlit UI\nAnswer + Sources + Strength Badge]

    subgraph Ingestion [Offline Ingestion Pipeline]
        N[OpenStax Vol 1-3 PDFs] --> P[PyMuPDF + pdfplumber\nMangle Detector]
        O[Feynman Vol I-III HTML] --> Q[BeautifulSoup\nLaTeX Preserved]
        P --> R[Sidebar Noise Filter\nRemove Check Your Understanding]
        R --> S[RecursiveCharTextSplitter\n1400/300 · 1200/250]
        Q --> S
        S --> T[all-mpnet-base-v2\nLocal Embeddings]
        T --> U[(ChromaDB\nPersisted · Backed up)]
        U --> E
    end
```

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- **Python 3.10 or 3.11**
- **Groq API Key**: Obtain a free API key from the [Groq Console](https://console.groq.com/) and define it in your environment.
- **NVIDIA GPU (Optional)**: If available, local embedding generation and reranking will automatically run on the GPU via CUDA.

### 2. Setup Virtual Environment
Clone this repository and create a virtual environment:
```bash
python -m venv myvenv
.\myvenv\Scripts\Activate.ps1   # On Windows
pip install -r requirements.txt
```

### 3. Set up Environment Variables
Create a `.env` file in the root directory and add your Groq API key:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
```

---

## 📥 Ingestion & Scraping

To populate the local database, run the following steps in sequence:

### 1. Download OpenStax PDFs
Downloads University Physics Volumes 1–3 dynamically by crawling the latest high-resolution PDF links:
```bash
python scripts/download_corpus.py
```

### 2. Scrape Feynman Lectures
Extracts all 115 chapters of The Feynman Lectures on Physics with mathematical LaTeX inline equations preserved:
```bash
python scripts/feynman_scraper.py
```

### 3. Ingest Into ChromaDB
Parses, splits, filters sidebar noise, hashes document IDs (for idempotency), generates embeddings, and saves the vector store + BM25 indices:
```bash
python src/ingest.py
```
*Note: Ingestion automatically runs on your GPU if a CUDA-enabled device is detected, making it extremely fast. Progress checkpoints are saved every 500 chunks.*

---

## 🚀 Running the Web Application

Launch the Streamlit UI dashboard:
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 🧪 Evaluation & Hallucination Test Suite

The test suite evaluates the RAG system parameters (Retrieval Precision@5, Recall@5, Citation Accuracy, OOS Refusal Rate) and compares them against a bare LLM baseline.

### 1. Run Baseline (Bare LLM)
Queries the 18 physics questions directly on `llama-3.1-8b-instant` without context:
```bash
python tests/baseline_runner.py
```

### 2. Run Hallucination & Retrieval Suite
Evaluates the RAG system and compares metrics side-by-side:
```bash
python tests/hallucination_suite.py --report --baseline
```
Detailed execution reports (both JSON and compiled PDF reports) are saved to `tests/results/`.

---

## 🐳 Containerised Deployment

To deploy the app using Docker:
```bash
docker-compose up -d
```
Access the application at `http://localhost:8501`. Ensure your `.env` file contains your `GROQ_API_KEY` before building.

---

## ⚠️ Important Considerations & Design Decisions

### 1. Retrieval Strength vs. Confidence
- The **Retrieval Strength** badge shown in the UI (calculated as the mean cosine similarity of top-K reranked chunks) is a **proxy metric of retrieval relevance**.
- It is **not** a calibrated probability of the answer's factual correctness. The UI clearly states this in tooltips to maintain transparency.

### 2. GPU Acceleration
- Local embeddings (`sentence-transformers/all-mpnet-base-v2`) and the reranker model automatically leverage NVIDIA CUDA acceleration if PyTorch detects a GPU.

### 3. LaTeX Equation Degradation
- PDFs can contain complex vector formatting that mangles equations when extracted. The ingestion pipeline uses a **three-layer parsing strategy** (PyMuPDF $\rightarrow$ mangle detection $\rightarrow$ pdfplumber fallback).
- Chunks that fail the mangle test are still indexed, but their citations in the UI are flagged with a `⚠️ Degraded Equations` badge to notify the user.
