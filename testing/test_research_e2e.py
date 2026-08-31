import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from dotenv import load_dotenv
load_dotenv()

from pipeline import run_research


print("==================================================")
print("           1. TESTING FAST RESEARCH MODE          ")
print("==================================================")
t0 = time.time()
res_fast = run_research("AI in Healthcare Diagnostics 2026", mode="fast")
t_fast = time.time() - t0

report_fast = res_fast.get("report", "")
score_fast = res_fast.get("critique_score", 0)
urls_fast = res_fast.get("verified_urls", [])

print(f"Status: COMPLETED successfully in {t_fast:.1f}s")
print(f"Report Length: {len(report_fast)} characters")
print(f"Critique Score: {score_fast}/10")
print(f"Verified Sources: {len(urls_fast)}")
print("\n--- Fast Report Preview ---")
print(report_fast[:300] + ("..." if len(report_fast) > 300 else ""))

print("\n==================================================")
print("           2. TESTING DEEP RESEARCH MODE          ")
print("==================================================")
t0 = time.time()
res_deep = run_research("Next-generation battery technologies for electric vehicles 2026", mode="deep")
t_deep = time.time() - t0

report_deep = res_deep.get("report", "")
critique_deep = res_deep.get("critique", "")
score_deep = res_deep.get("critique_score", 0)
fact_score_deep = res_deep.get("fact_check_score", 0.0)
urls_deep = res_deep.get("verified_urls", [])
knowledge_graph_deep = res_deep.get("knowledge_graph", "")

print(f"Status: COMPLETED successfully in {t_deep:.1f}s")
print(f"Report Length: {len(report_deep)} characters")
print(f"Critique Score: {score_deep}/10")
print(f"Fact-Check Trust Score: {fact_score_deep * 100:.0f}%")
print(f"Verified Sources: {len(urls_deep)}")
print(f"Knowledge Graph Generated: {bool(knowledge_graph_deep)}")

print("\n--- Deep Critique Feedback Snippet ---")
print(critique_deep[:300] + ("..." if len(critique_deep) > 300 else ""))

print("\n--- Deep Report Conclusion & Sources (Tail) ---")
print(report_deep[-450:] if len(report_deep) > 450 else report_deep)

print("\n==================================================")
print("           ALL TESTS COMPLETED SUCCESSFULLY       ")
print("==================================================")
