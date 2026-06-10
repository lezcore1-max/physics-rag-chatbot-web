import os
import sys
import fitz

def search_corpus(query_text):
    print(f"Searching corpus for: '{query_text}'")
    query_lower = query_text.lower()
    
    # Search Feynman text files
    feynman_dir = "data/raw/feynman"
    feynman_matches = []
    for root, dirs, files in os.walk(feynman_dir):
        for file in files:
            if file.endswith(".txt"):
                path = os.path.join(root, file)
                try:
                    content = open(path, "r", encoding="utf-8").read()
                    if query_lower in content.lower():
                        feynman_matches.append(path)
                except:
                    pass
                    
    # Search OpenStax PDF files
    openstax_matches = []
    for root, dirs, files in os.walk("data/raw"):
        for file in files:
            if file.endswith(".pdf"):
                path = os.path.join(root, file)
                try:
                    doc = fitz.open(path)
                    for page_num in range(len(doc)):
                        text = doc[page_num].get_text()
                        if query_lower in text.lower():
                            openstax_matches.append((path, page_num + 1))
                except:
                    pass
                    
    print(f"Feynman matches: {feynman_matches}")
    print(f"OpenStax matches: {openstax_matches}")

if __name__ == "__main__":
    search_corpus("mgh")
    print("-" * 50)
    search_corpus("gravitational potential energy")
