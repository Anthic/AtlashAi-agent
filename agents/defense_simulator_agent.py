import json
import logging
import re
from typing import Dict, Any, List, Optional
from pipeline.fallback import execute_with_fallback, CascadeExecutionResult
log = logging.getLogger(__name__)


COMMITTEE_QUESTIONS_PROMPT = """You are the Chair of an Elite Doctoral & Master's Thesis Defense Committee.
Analyze the submitted research paper and convene your 3-examiner committee:
1. Prof. Vance (Methodology Hawk) — Attacks sample size, control groups, confounding variables, dataset bias, and statistical power.
2. Dr. Evelyn (Novelty & SOTA Challenger) — Challenges incremental claims, asks why prior art was omitted, and doubts theoretical contribution.
3. Dean Hastings (Practical Stress-Tester) — Asks about real-world deployment, computational cost, edge-case failures, and negative societal impact.
TASK:
Formulate exactly ONE deeply rigorous, targeted defense question from each examiner based directly on the paper's claims and weaknesses.
OUTPUT FORMAT:
Return strictly a valid JSON array containing exactly 3 objects (do NOT include markdown code blocks or extra conversational text):
[
  {
    "examiner_id": "vance",
    "examiner_name": "Prof. Vance",
    "examiner_title": "Methodology & Validity Hawk",
    "avatar_badge": "🔬",
    "targeted_section": "Section name or claim in paper",
    "question": "Exact tough question challenging their methodology",
    "what_examiner_looks_for": "Key proof or reasoning needed to survive this question"
  },
  {
    "examiner_id": "evelyn",
    "examiner_name": "Dr. Evelyn",
    "examiner_title": "Novelty & SOTA Challenger",
    "avatar_badge": "⚡",
    "targeted_section": "Prior art or novelty claim",
    "question": "Exact tough question challenging originality",
    "what_examiner_looks_for": "Theoretical justification and baseline comparison"
  },
  {
    "examiner_id": "hastings",
    "examiner_name": "Dean Hastings",
    "examiner_title": "Practical Stress-Tester",
    "avatar_badge": "🏛️",
    "targeted_section": "Real-world scalability or limitations",
    "question": "Exact tough question on failure modes or cost",
    "what_examiner_looks_for": "Pragmatic engineering or clinical constraints acknowledged"
  }
]
"""

DEFENSE_EVALUATION_PROMPT = """You are the Thesis Defense Committee evaluating a student's live defense answer (rebuttal) to an examiner's interrogation.
EXAMINER: {examiner_name} ({examiner_title})
TARGET QUESTION:
{question}
STUDENT'S LIVE REBUTTAL / ANSWER:
{student_answer}
TASK:
Deliberate on the student's answer. Be strict and academic. Did they dodge? Did they provide empirical reasoning?
Return strictly a valid JSON object (no markdown, no preamble):
{{
  "score": 85,
  "verdict": "Passed with Honors | Defense Accepted | Conditional Pass | Defense Cracked",
  "examiner_reaction": "One sentence capturing the examiner's facial and intellectual reaction.",
  "strengths": ["Strong defense point made"],
  "weaknesses": ["Flaw or evasion in student's logic"],
  "closing_advice": "Specific revision or justification to add to the paper before final viva."
}}
"""

def generate_defense_questions (
    paper_title : str,
    paper_content : str,
    custom_cascade : Optional[List[str]] = None
) -> Dict[str, Any] :
    """Generates 3 tough committee questions from the 3 examiners."""
    full_prompt = (
        f"[SYSTEM]\n{COMMITTEE_QUESTIONS_PROMPT}\n\n"
        f"[PAPER TITLE]\n{paper_title}\n\n"
        f"[PAPER BODY / EXCERPT]\n{paper_content[:12000]}\n\n"
        f"[COMMITTEE INTERROGATION QUESTIONS]"        
    )

    result = execute_with_fallback(full_prompt, custom_cascade=custom_cascade)
    raw = result.content.strip() 

    json_match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if json_match : 
        try : 
            questions = json.loads(json_match.group(0))
            return {"title" : paper_title, "questions" : questions, "provider" : result.provider_used}
        except Exception:
            pass
    return {
        "title": paper_title,
        "questions": [
            {
                "examiner_id": "vance",
                "examiner_name": "Prof. Vance",
                "examiner_title": "Methodology Hawk",
                "avatar_badge": "🔬",
                "targeted_section": "Methodology",
                "question": "How did you guarantee that your experimental setup did not suffer from selection bias or data leakage?",
                "what_examiner_looks_for": "Rigorous cross-validation or baseline isolation",
            }
        ],
        "provider": result.provider_used,
    }


def evaluate_defense_rebuttal(
    examiner_name: str,
    examiner_title: str,
    question: str,
    student_answer: str,
    custom_cascade: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Evaluates the student's defense answer to a specific question."""
    prompt = DEFENSE_EVALUATION_PROMPT.format(
        examiner_name=examiner_name,
        examiner_title=examiner_title,
        question=question,
        student_answer=student_answer,
    )
    result = execute_with_fallback(f"[SYSTEM]\n{prompt}", custom_cascade=custom_cascade)
    raw = result.content.strip()
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except Exception:
            pass
    return {
        "score": 75,
        "verdict": "Conditional Pass",
        "examiner_reaction": "The committee accepted the argument with reservations.",
        "strengths": ["Addressed core concern directly"],
        "weaknesses": ["Could provide stronger mathematical grounding"],
        "closing_advice": "Incorporate empirical variance intervals into the final manuscript.",
    }