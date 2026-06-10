"""
src/retriever.py
MMR Ensemble Retriever combining BM25 keyword search and vectorstore search,
followed by CrossEncoder reranking.
"""

import os
import sys
import pickle
from typing import List, Optional, Union

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    CHROMA_DIR, BM25_CACHE, TOP_K, FINAL_K, BM25_WEIGHT, SEMANTIC_WEIGHT,
    MMR_LAMBDA, FETCH_K, RERANKER_MODEL, EMBED_MODEL
)

from langchain_core.documents import Document
from langchain_chroma import Chroma
from src.embeddings import LocalSentenceTransformerEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents.compressor import BaseDocumentCompressor
from pydantic import ConfigDict
from typing import Sequence, Any, Optional

class CustomCrossEncoderReranker(BaseDocumentCompressor):
    model: HuggingFaceCrossEncoder
    top_n: int = 3

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Any] = None,
    ) -> Sequence[Document]:
        if not documents:
            return []
        scores = self.model.score([(query, doc.page_content) for doc in documents])
        docs_with_scores = list(zip(documents, scores, strict=False))
        result = sorted(docs_with_scores, key=lambda x: x[1], reverse=True)
        
        final_docs = []
        for doc, score in result[: self.top_n]:
            doc.metadata["relevance_score"] = float(score)
            final_docs.append(doc)
        return final_docs

def get_ensemble_retriever() -> Optional[Union[ContextualCompressionRetriever, EnsembleRetriever]]:
    """
    Initialises and returns the ensemble retriever.
    Combines:
    1. BM25Retriever from cached corpus
    2. Chroma vectorstore retriever with MMR search
    Reranks using CrossEncoderReranker.
    """
    # 1. Initialise Embeddings
    embeddings = LocalSentenceTransformerEmbeddings()

    # 2. Check ChromaDB vectorstore
    if not os.path.exists(CHROMA_DIR) or not os.listdir(CHROMA_DIR):
        print(f"WARNING: ChromaDB directory '{CHROMA_DIR}' is empty or does not exist.")
        return None

    try:
        vectorstore = Chroma(
            collection_name="physics_rag",
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR,
        )
    except Exception as e:
        print(f"ERROR: Could not initialise ChromaDB: {e}")
        return None

    # 3. Check and load BM25 index cache
    if not os.path.exists(BM25_CACHE):
        print(f"WARNING: BM25 index cache '{BM25_CACHE}' does not exist.")
        return None

    try:
        with open(BM25_CACHE, "rb") as f:
            bm25_data = pickle.load(f)
        
        texts = bm25_data.get("texts", [])
        metadatas = bm25_data.get("metadatas", [])
        ids = bm25_data.get("ids", [])

        if not texts:
            print("WARNING: BM25 cache is empty.")
            return None

        # Recreate Documents from texts and metadatas for BM25Retriever
        bm25_docs = []
        for i, text in enumerate(texts):
            meta = metadatas[i] if i < len(metadatas) else {}
            # Ensure doc_id is in metadata
            if i < len(ids):
                meta["doc_id"] = ids[i]
            bm25_docs.append(Document(page_content=text, metadata=meta))

        bm25_retriever = BM25Retriever.from_documents(bm25_docs)
        bm25_retriever.k = TOP_K

    except Exception as e:
        print(f"ERROR: Could not load BM25 index: {e}")
        return None

    # 4. Initialise Semantic Retriever (MMR)
    semantic_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": TOP_K,
            "fetch_k": FETCH_K,
            "lambda_mult": MMR_LAMBDA,
        }
    )

    # 5. Build Ensemble Retriever
    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=[BM25_WEIGHT, SEMANTIC_WEIGHT],
    )

    # 6. Reranker setup
    try:
        # Load CrossEncoder (will download and cache locally on first run)
        print(f"[RE-RANK] Initialising CrossEncoder: {RERANKER_MODEL}...")
        model = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
        reranker = CustomCrossEncoderReranker(model=model, top_n=FINAL_K)
        
        # Wrapped compression retriever
        retriever = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=ensemble,
        )
        return retriever
    except Exception as e:
        print(f"ERROR: Could not load CrossEncoder reranker: {e}")
        # Fallback to pure ensemble without reranking
        print("Falling back to pure ensemble retriever.")
        return ensemble

if __name__ == "__main__":
    # Test retriever loading
    print("Testing retriever loading...")
    r = get_ensemble_retriever()
    if r:
        print("Success! Retriever loaded.")
    else:
        print("Retriever could not be loaded (likely corpus not ingested yet).")
