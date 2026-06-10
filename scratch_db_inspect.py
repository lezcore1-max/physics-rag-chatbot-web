import os
import sys
import pickle
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CHROMA_DIR, BM25_CACHE
from langchain_chroma import Chroma
from src.embeddings import LocalSentenceTransformerEmbeddings

def main():
    print(f"Chroma directory: {CHROMA_DIR}")
    print(f"BM25 cache path: {BM25_CACHE}")
    
    if not os.path.exists(CHROMA_DIR):
        print("Chroma directory does not exist.")
        return
        
    embeddings = LocalSentenceTransformerEmbeddings()
    vectorstore = Chroma(
        collection_name="physics_rag",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    
    all_data = vectorstore.get()
    metadatas = all_data.get("metadatas", [])
    documents = all_data.get("documents", [])
    
    print(f"Total chunks in Chroma: {len(metadatas)}")
    
    sources = Counter()
    types = Counter()
    
    for meta in metadatas:
        sources[meta.get("source")] += 1
        types[meta.get("type")] += 1
        
    print("\nTypes distribution:")
    for t, count in types.items():
        print(f"  {t}: {count}")
        
    print("\nSources distribution:")
    for s, count in sources.items():
        print(f"  {s}: {count}")

    if os.path.exists(BM25_CACHE):
        with open(BM25_CACHE, "rb") as f:
            bm25_data = pickle.load(f)
        bm25_metadatas = bm25_data.get("metadatas", [])
        bm25_types = Counter()
        for meta in bm25_metadatas:
            bm25_types[meta.get("type")] += 1
        print("\nBM25 Types distribution:")
        for t, count in bm25_types.items():
            print(f"  {t}: {count}")

if __name__ == "__main__":
    main()
