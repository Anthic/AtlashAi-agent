import os
import threading
from langchain_mistralai import ChatMistralAI

model_map = {
    "fast": "mistral-small-2603",
    "smart": "mistral-large-2512",
}

_llm_cache : dict = {}
_lock = threading.Lock()

def get_llm(kind : str) -> ChatMistralAI :
    if kind not in model_map:
        raise ValueError(f"Unknown model kind: {kind!r}. Valid options: {list(model_map.keys())}")

    if kind in _llm_cache:
        return _llm_cache[kind]
    with _lock:
        if kind not in _llm_cache:
            _llm_cache[kind] = ChatMistralAI(
                model = model_map[kind],
                temperature=0,
                api_key=os.getenv("MISTRALAI_API_KEY"),
            )
        return _llm_cache[kind]
    



def get_available_models()-> list:
    return list(model_map.keys())
def clear_cache() -> None:
    with _lock :
        _llm_cache.clear()
