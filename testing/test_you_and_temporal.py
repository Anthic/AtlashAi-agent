import sys
import os
from pathlib import Path
from dotenv import load_dotenv

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
load_dotenv(dotenv_path=root_dir / ".env")

def test_you_search():
    print("--- [TEST 1: You.com Search Tool] ---")
    try:
        from tools.you_search_tool import you_web_search, search_temporal_facts, search_academic_critiques
        print("Testing you_web_search...")
        results = you_web_search("DeepSeek V3 architecture", num_web_results=2)
        print(f"✅ you_web_search returned {len(results)} hits.")
        if results:
            print(f"   Sample title: {results[0].get('title')}")
            print(f"   Sample url: {results[0].get('url')}")

        print("\nTesting search_academic_critiques...")
        critiques = search_academic_critiques(topic="Retrieval-Augmented Generation", methodology="dense embeddings")
        print(f"✅ search_academic_critiques returned {len(critiques)} hits.")
    except Exception as e:
        print(f"❌ Error in You.com search tool: {e}")

def test_defense_simulator():
    print("\n--- [TEST 2: Defense Simulator Live Critiques] ---")
    try:
        from agents.defense_simulator_agent import generate_defense_questions
        res = generate_defense_questions(
            paper_title="Evaluating Hallucinations in Large Language Models using RAG",
            paper_content="We propose an automated pipeline to measure factual hallucinations in retrieval augmented generation models...",
            tier="worker"
        )
        print(f"✅ Questions generated: {len(res.get('questions', []))}")
        print(f"   Provider: {res.get('provider')}")
        print(f"   Web Grounded: {res.get('web_grounded')}")
        print(f"   Citations: {res.get('citations')}")
    except Exception as e:
        print(f"❌ Error in Defense Simulator: {e}")

def test_temporal_fact_check():
    print("\n--- [TEST 3: Temporal Fact-Check Agent] ---")
    try:
        from agents.fact_check_agent import verify_paper_claims_live
        res = verify_paper_claims_live(
            paper_title="BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
            paper_content="BERT achieves state-of-the-art results on eleven natural language processing tasks, outperforming all previous models."
        )
        print(f"✅ Verified claims: {len(res.get('verified_claims', []))}")
        for vc in res.get("verified_claims", []):
            print(f"   Claim: {vc.get('claim')}")
            print(f"   Status: {vc.get('status')}")
            print(f"   Explanation: {vc.get('explanation')}")
            print(f"   Citations count: {len(vc.get('citations', []))}")
        print(f"   Timestamp: {res.get('timestamp')}")
    except Exception as e:
        print(f"❌ Error in Temporal Fact Check: {e}")

if __name__ == "__main__":
    test_you_search()
    test_defense_simulator()
    test_temporal_fact_check()
