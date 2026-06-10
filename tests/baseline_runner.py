"""
tests/baseline_runner.py
Runs the 18 physics test questions directly on ChatOllama without RAG (no context).
Saves results for comparison against the RAG system to demonstrate reduction in hallucinations.
"""

import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import LLM_MODEL
from src.llm_chain import init_llm

def main():
    print("=" * 60)
    # 1. Load questions
    questions_file = os.path.join("tests", "test_questions.json")
    if not os.path.exists(questions_file):
        print(f"ERROR: {questions_file} not found. Run tests from project root.")
        return

    with open(questions_file, "r") as f:
        all_q = json.load(f)

    # Filter for physics questions only (exclude OOS)
    physics_questions = [q for q in all_q if "Out of Scope" not in q["category"]]
    print(f"Loaded {len(physics_questions)} physics questions for baseline run.")

    # 2. Initialize LLM (No RAG)
    print(f"Initialising bare LLM ({LLM_MODEL})...")
    try:
        llm = init_llm()
        if llm is None:
            raise ValueError("init_llm returned None")
    except Exception as e:
        print(f"ERROR: Could not connect to LLM: {e}")
        return

    # 3. Run evaluation
    results = []
    print("\nRunning baseline evaluation (no RAG)...")
    
    total_keywords_matched = 0
    total_possible_keywords = 0

    for idx, q in enumerate(physics_questions):
        print(f"  [{q['id']}] {q['question'][:50]}...")
        
        # Plain question prompt
        prompt = f"Answer the following question about physics:\n\n{q['question']}\n\nAnswer concisely and accurately:"
        
        start_time = datetime.now()
        try:
            response = llm.invoke(prompt)
            answer = response.content if hasattr(response, "content") else str(response)
            success = True
        except Exception as e:
            answer = f"FAILED: {e}"
            success = False

        duration = (datetime.now() - start_time).total_seconds()
        
        # Check keyword matches
        answer_lower = answer.lower()
        matched = [kw for kw in q["ground_truth_keywords"] if kw.lower() in answer_lower]
        keyword_pct = len(matched) / len(q["ground_truth_keywords"]) if q["ground_truth_keywords"] else 0.0
        
        total_keywords_matched += len(matched)
        total_possible_keywords += len(q["ground_truth_keywords"])

        # Check for citation (should be 0% for baseline)
        has_citation = "[source:" in answer_lower or "source: openstax" in answer_lower or "source: feynman" in answer_lower
        
        results.append({
            "id": q["id"],
            "question": q["question"],
            "category": q["category"],
            "answer": answer,
            "duration_sec": duration,
            "keywords_matched": matched,
            "keyword_match_pct": keyword_pct,
            "has_citation": has_citation,
            "success": success
        })

    avg_keyword_match = total_keywords_matched / total_possible_keywords if total_possible_keywords else 0.0
    citation_rate = sum(1 for r in results if r["has_citation"]) / len(results) if results else 0.0

    print(f"\nBaseline Run Summary:")
    print(f"  Avg Keyword Match Rate: {avg_keyword_match:.2%}")
    print(f"  Citation Rate: {citation_rate:.2%}")

    # 4. Save results
    output_dir = os.path.join("tests", "results")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "baseline_results.json")
    
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "model": LLM_MODEL,
        "avg_keyword_match": avg_keyword_match,
        "citation_rate": citation_rate,
        "results": results
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Baseline results saved to {output_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
