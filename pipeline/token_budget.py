"""
pipeline/token_budget.py
────────────────────────────────────────────────────────────
Fintech-Grade Token Budgeting & BDT Cost Accounting Engine.
• Financial Calculations powered by `decimal.Decimal` (Zero Float Drift).
• Exact Sub-Paisa Micro-Billing (No Artificial Floor Overcharge).
• Sliding Window Token Constraining.
• 1 Credit = 1.00 BDT (৳).
"""

import logging
from decimal import Decimal, ROUND_HALF_UP
from typing import List, Dict, Any, Optional

log = logging.getLogger(__name__)

# ── 1. BDT Pricing Table (Per 1 Million Tokens in BDT ৳ using Decimal) ────────
# Standardized based on 1 USD = 120 BDT + Local Server Margin

MODEL_PRICING_BDT_PER_1M: Dict[str, Dict[str, Decimal]] = {
    # Fast Worker Tier (Sub-paisa micro rates)
    "worker-groq": {
        "input": Decimal("8.00"),       # ৳0.008 per 1k
        "output": Decimal("24.00"),     # ৳0.024 per 1k
    },
    "worker-gemini": {
        "input": Decimal("10.00"),      # ৳0.010 per 1k
        "output": Decimal("30.00"),     # ৳0.030 per 1k
    },
    "worker-mistral": {
        "input": Decimal("25.00"),      # ৳0.025 per 1k
        "output": Decimal("75.00"),     # ৳0.075 per 1k
    },
    "worker-openrouter": {
        "input": Decimal("12.00"),
        "output": Decimal("36.00"),
    },
    # Smart Master Tier
    "master-mistral": {
        "input": Decimal("250.00"),     # ৳0.250 per 1k
        "output": Decimal("750.00"),    # ৳0.750 per 1k
    },
    "master-gemini": {
        "input": Decimal("150.00"),     # ৳0.150 per 1k
        "output": Decimal("450.00"),    # ৳0.450 per 1k
    },
    "master-openrouter": {
        "input": Decimal("200.00"),
        "output": Decimal("600.00"),
    },
}

# Fallback rate if unknown model encountered
DEFAULT_PRICING: Dict[str, Decimal] = {
    "input": Decimal("20.00"),
    "output": Decimal("60.00"),
}

DECIMAL_ONE_MILLION = Decimal("1000000")
PRECISION_FOUR_PLACES = Decimal("0.0001")
PRECISION_TWO_PLACES = Decimal("0.01")


# ── 2. Token Budget Manager (Sliding Window Enforcer) ──────────────────────────

class TokenBudgetManager:
    """
    Manages and constrains token limits for prompts and message histories.
    """

    def __init__(self, max_input_tokens: int = 4096, max_output_tokens: int = 2048):
        self.max_input = max_input_tokens
        self.max_output = max_output_tokens
        self._tokenizer = None
        self._init_tokenizer()

    def _init_tokenizer(self):
        try:
            import tiktoken
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            log.warning("tiktoken unavailable, falling back to heuristic token counter: %s", e)
            self._tokenizer = None

    def count_tokens(self, text: str) -> int:
        """Accurate token count via tiktoken with heuristic fallback."""
        if not text:
            return 0
        if self._tokenizer:
            return len(self._tokenizer.encode(text))
        return max(1, len(text) // 4)

    def constrain_text(self, text: str, max_tokens: Optional[int] = None) -> str:
        """Truncates an oversized text block to strictly fit the budget."""
        budget = max_tokens or self.max_input
        current_tokens = self.count_tokens(text)
        if current_tokens <= budget:
            return text

        ratio = budget / current_tokens
        safe_chars = int(len(text) * ratio * 0.95)
        log.warning("Text truncated from %d tokens to fit %d budget.", current_tokens, budget)
        return text[:safe_chars] + "\n...[truncated for token budget]"

    def constrain_messages(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Sliding-window message constraining:
        Preserves System Prompt (if present) and retains newest messages.
        """
        budget = max_tokens or self.max_input
        if not messages:
            return []

        system_msg = None
        working_messages = messages[:]
        if working_messages and working_messages[0].get("role") == "system":
            system_msg = working_messages.pop(0)

        system_tokens = self.count_tokens(system_msg.get("content", "")) if system_msg else 0
        remaining_budget = budget - system_tokens

        total_tokens = 0
        constrained = []

        for msg in reversed(working_messages):
            tokens = self.count_tokens(msg.get("content", ""))
            if total_tokens + tokens > remaining_budget:
                log.info("TokenBudget: Dropped older message to preserve %d token limit.", budget)
                break
            constrained.insert(0, msg)
            total_tokens += tokens

        if system_msg:
            constrained.insert(0, system_msg)

        return constrained


# ── 3. BDT Cost & Credit Calculator (Fintech Precision) ────────────────────────

class BDTCostCalculator:
    """
    Fintech-grade cost calculator using `Decimal` arithmetic.
    Calculates exact sub-paisa BDT costs and deductions with zero float drift.
    """

    @staticmethod
    def calculate_cost(
        model_key: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> Dict[str, Any]:
        """
        Calculates exact BDT cost and credits to deduct for an AI generation.
        """
        if model_key not in MODEL_PRICING_BDT_PER_1M:
            log.warning(
                "Unknown model_key '%s' encountered in BDTCostCalculator. Falling back to default pricing.",
                model_key,
            )
            pricing = DEFAULT_PRICING
        else:
            pricing = MODEL_PRICING_BDT_PER_1M[model_key]

        dec_prompt = Decimal(str(prompt_tokens))
        dec_comp = Decimal(str(completion_tokens))

        prompt_cost = (dec_prompt / DECIMAL_ONE_MILLION) * pricing["input"]
        completion_cost = (dec_comp / DECIMAL_ONE_MILLION) * pricing["output"]
        total_bdt = prompt_cost + completion_cost

        # Exact sub-paisa precision (4 decimal places: e.g. ৳0.0088)
        cost_bdt_precise = total_bdt.quantize(PRECISION_FOUR_PLACES, rounding=ROUND_HALF_UP)
        # Credit deduction (2 decimal places: e.g. 0.01 Credits)
        credits_deducted = total_bdt.quantize(PRECISION_TWO_PLACES, rounding=ROUND_HALF_UP)

        return {
            "model_key": model_key,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost_bdt": float(cost_bdt_precise),
            "credits_deducted": float(credits_deducted),
            "currency": "BDT",
        }
