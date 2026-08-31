import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pipeline.model import MODEL_REGISTRY
from pipeline.chains import _looks_complete, sanitize_final_report



def test_model_registry_keys():
    """Checks whether the required models exist in the configuration"""
    assert "worker-groq" in MODEL_REGISTRY
    assert "master-mistral" in MODEL_REGISTRY
    assert "worker-gemini" in MODEL_REGISTRY


def test_sanitizer_logic():
    """Tests the report sanitizer function"""
    dirty_text = "## Introduction\nSome text here.\n---\n```json\n{}\n```"
    cleaned = sanitize_final_report(dirty_text)
    assert "```json" not in cleaned
    assert "---" not in cleaned


def test_looks_complete_sentence():
    """Tests cut-off detection"""
    assert _looks_complete("This is a complete sentence.") == True
    assert _looks_complete("This sentence was cut off in the") == False


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("Running pipeline unit tests...")
    test_model_registry_keys()
    print("[PASS] test_model_registry_keys")
    test_sanitizer_logic()
    print("[PASS] test_sanitizer_logic")
    test_looks_complete_sentence()
    print("[PASS] test_looks_complete_sentence")
    print("All unit tests passed successfully!")
