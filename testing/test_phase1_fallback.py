"""
testing/test_phase1_fallback.py
────────────────────────────────────────────────────────────
Diagnostic test for Phase 1 Multi-Provider Cascade & Zero Cutoff.
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Set project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

# Load Environment Variables
load_dotenv(dotenv_path=root_dir / ".env")

from pipeline.fallback import execute_with_fallback
from pipeline.chains import run_with_zero_cutoff

def test_cascade():
    print("\n🔍 --- [TEST 1: Fast Worker Cascade Execution] ---")
    prompt = "Explain quantum entanglement in 2 short bullet points for an academic abstract."
    
    try:
        res = execute_with_fallback(prompt, tier="worker")
        print(f"✅ Provider Used: {res.provider_used}")
        print(f"⏱️ Duration: {res.duration_sec}s")
        print(f"📊 Token Usage: {res.token_usage}")
        print(f"📝 Response Preview:\n{res.content[:200]}...\n")
    except Exception as e:
        print(f"❌ Worker Cascade Error: {e}")

    print("\n🔍 --- [TEST 2: Grounded Zero Cut-off Test] ---")
    try:
        complete_text = run_with_zero_cutoff(
            initial_text="CRISPR-Cas9 is a revolutionary gene-editing technology that allows scientists to",
            context="CRISPR-Cas9 allows precise genome editing in living organisms for treating genetic diseases.",
            verified_urls_text="- https://nature.com/articles/crispr",
            topic="CRISPR Gene Editing",
            tier="worker"
        )
        print(f"✅ Clean Complete Output:\n{complete_text}\n")
    except Exception as e:
        print(f"❌ Zero Cut-off Error: {e}")

if __name__ == "__main__":
    test_cascade()
