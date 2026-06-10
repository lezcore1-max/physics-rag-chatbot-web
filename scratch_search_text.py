import os
import sys
import re
import fitz # PyMuPDF

def search_text_files(directory, query_text):
    print(f"Searching text files in {directory} for '{query_text}'...")
    query_lower = query_text.lower()
    matches = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".txt"):
                path = os.path.join(root, file)
                try:
                    content = open(path, "r", encoding="utf-8").read()
                    if query_lower in content.lower():
                        matches.append(path)
                except Exception as e:
                    pass
    return matches

def search_pdf_files(directory, query_text):
    print(f"Searching PDF files in {directory} for '{query_text}'...")
    query_lower = query_text.lower()
    matches = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".pdf"):
                path = os.path.join(root, file)
                try:
                    doc = fitz.open(path)
                    for page_num in range(len(doc)):
                        text = doc[page_num].get_text()
                        if query_lower in text.lower():
                            matches.append((path, page_num + 1))
                except Exception as e:
                    pass
    return matches

if __name__ == "__main__":
    raw_dir = "c:\\Users\\vansh\\Music\\physics\\data\\raw"
    
    # 1. Search for virtual work
    q1 = "principle of virtual work"
    txt_m1 = search_text_files(raw_dir, q1)
    pdf_m1 = search_pdf_files(raw_dir, q1)
    print(f"Matches for '{q1}':")
    print(f"Text files: {txt_m1}")
    print(f"PDF files: {pdf_m1}")
    
    # 2. Search for the exact beam problem
    q2 = "Initially the beam is horizontal"
    txt_m2 = search_text_files(raw_dir, q2)
    pdf_m2 = search_pdf_files(raw_dir, q2)
    print(f"\nMatches for '{q2}':")
    print(f"Text files: {txt_m2}")
    print(f"PDF files: {pdf_m2}")
