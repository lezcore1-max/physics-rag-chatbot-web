import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.retriever import get_ensemble_retriever
from src.llm_chain import get_citation_tag

def test_query(query):
    print(f"\n==================================================")
    print(f"Testing Query: '{query}'")
    print(f"==================================================")
    
    retriever = get_ensemble_retriever()
    if retriever is None:
        print("Retriever is None!")
        return
        
    docs = retriever.invoke(query)
    print(f"Retrieved {len(docs)} documents.")
    
    for idx, doc in enumerate(docs):
        print(f"\n--- Document {idx+1} ---")
        print(f"Source Tag: {get_citation_tag(doc)}")
        print(f"Metadata: {doc.metadata}")
        # Print first 200 chars of content
        content_preview = doc.page_content.replace("\n", " ")[:200]
        print(f"Content: {content_preview}...")
        
        # Check if there is relevance score
        rel_score = doc.metadata.get("relevance_score")
        print(f"Metadata relevance_score: {rel_score}")
        
        # Check other attributes
        for attr in ['relevance_score', 'score', 'state']:
            if hasattr(doc, attr):
                print(f"Attribute {attr}: {getattr(doc, attr)}")

if __name__ == "__main__":
    # Test a Feynman-related query
    test_query("What is the principle of virtual work?")
