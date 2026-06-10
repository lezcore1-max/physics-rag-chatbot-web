import os
import sys
import pickle
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import BM25_CACHE

def main():
    print(f"BM25 cache path: {BM25_CACHE}")
    
    if not os.path.exists(BM25_CACHE):
        print("BM25 cache does not exist.")
        return
        
    with open(BM25_CACHE, "rb") as f:
        bm25_data = pickle.load(f)
        
    metadatas = bm25_data.get("metadatas", [])
    texts = bm25_data.get("texts", [])
    
    print(f"Total chunks in BM25: {len(metadatas)}")
    
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

if __name__ == "__main__":
    main()
