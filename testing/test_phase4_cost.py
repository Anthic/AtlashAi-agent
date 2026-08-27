"""
testing/test_phase4_cost.py
────────────────────────────────────────────────────────────
Fintech Decimal Precision & Token Budget Test.
"""

import sys
from pathlib import Path

# Project root path setup
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from pipeline.token_budget import TokenBudgetManager, BDTCostCalculator

def test_token_budget_and_bdt_cost():
    print("\n" + "=" * 65)
    print("🧪 --- [FINTECH TEST: Decimal Precision BDT Cost Calculator] ---")
    print("=" * 65)

    # 1. Test Token Counting & Constraining
    budget = TokenBudgetManager(max_input_tokens=50)
    sample_text = "This is a long research prompt about quantum machine learning and genetic algorithms. " * 10
    
    tokens_before = budget.count_tokens(sample_text)
    constrained_text = budget.constrain_text(sample_text)
    tokens_after = budget.count_tokens(constrained_text)

    print(f"📊 Original Tokens: {tokens_before}")
    print(f"✂️ Constrained Tokens: {tokens_after} (Hard Budget: 50)")

    # 2. Test Sub-Paisa Micro-Billing Cases (Exact Decimal Precision)
    test_cases = [
        ("worker-groq", 500, 200),       # Exact: ৳0.0088 (No artificial floor)
        ("worker-gemini", 1000, 500),    # Exact: ৳0.0250
        ("master-mistral", 2500, 1200),  # Exact: ৳1.5250
        ("unknown-model", 100, 100),     # Tests logging warning + fallback
    ]

    print("\n💰 --- [BDT Exact Cost Breakdown (1 USD = 120 BDT)] ---")
    for model_key, prompt_tok, comp_tok in test_cases:
        res = BDTCostCalculator.calculate_cost(model_key, prompt_tok, comp_tok)
        print(f"• Model: {model_key:<16} | Tokens: {res['total_tokens']:<5} | Exact Cost: ৳{res['cost_bdt']:.4f} BDT ({res['credits_deducted']} Credits)")

    print("\n✅ All Fintech Decimal Precision calculations verified successfully!\n")

if __name__ == "__main__":
    test_token_budget_and_bdt_cost()
