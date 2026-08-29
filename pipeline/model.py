
import os
import logging
import threading
from typing import Optional, Dict, Any
from langchain_core.language_models.chat_models import BaseChatModel


log = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "worker-groq": {
        "provider": "groq",
        "model": "qwen/qwen3.8-27b",  
        "temperature": 0.1,
    },
    "worker-gemini": {
        "provider": "google",
        "model": "gemini-3.5-flash-lite",   
        "temperature": 0.1,
    },
    "worker-mistral": {
        "provider": "mistral",
        "model": "mistral-small-latest",
        "temperature": 0.1,
    },
    "worker-openrouter": {
        "provider": "openrouter",
        "model": "qwen/qwen-2.5-72b-instruct:free",
        "temperature": 0.1,
    },
    "master-mistral": {
        "provider": "mistral",
        "model": "mistral-small-latest",
        "temperature": 0.2,
    },
    "master-gemini": {
        "provider": "google",
        "model": "gemini-3.5-flash-lite",   
        "temperature": 0.2,
    },
    "master-openrouter": {
        "provider": "openrouter",
        "model": "deepseek/deepseek-r1:free",
        "temperature": 0.2,
    },
}
_llm_cache : Dict[str, BaseChatModel] = {}
_lock = threading.Lock()


def create_llm_instance(provider: str, model_name: str, temperature: float = 0.0, **kwargs) -> BaseChatModel:
    provider = provider.lower()
    kwargs.setdefault("max_retries", 1)   

    if provider == "groq":
        from langchain_groq import ChatGroq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set in environment variables.")
        return ChatGroq(
            model=model_name,
            temperature=temperature,
            groq_api_key=api_key,
            **kwargs
        )

    elif provider in ("google", "gemini"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in environment variables.")
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=api_key,
            **kwargs
        )

    elif provider == "mistral":
        from langchain_mistralai import ChatMistralAI
        api_key = os.getenv("MISTRALAI_API_KEY")
        if not api_key:
            raise ValueError("MISTRALAI_API_KEY is not set in environment variables.")
        return ChatMistralAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
            **kwargs
        )

    elif provider == "openrouter":
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set in environment variables.")
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            **kwargs
        )

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
def get_llm(key_or_kind : str = "worker-groq", **override_kwargs) -> BaseChatModel :
    """
    Retrive or create a cached LLM instance,
    accepts registry key (e.g., "worker-groq", "master-mistrial") or legacy "fast/smart"
    """
    alias_map  = {
        "fast" : "worker-groq",
        "smart" : "master-mistral",
    }
    resolved_key = alias_map.get(key_or_kind, key_or_kind)

    if resolved_key not in MODEL_REGISTRY :
        raise ValueError(
            f"Unknown model key: {key_or_kind!r}. Valid options: {list(MODEL_REGISTRY.keys())}"
        )

    if not override_kwargs and resolved_key in _llm_cache : 
        return  _llm_cache[resolved_key]
    with _lock : 
        if not override_kwargs and resolved_key in _llm_cache : 
            return _llm_cache[resolved_key]

        conf = MODEL_REGISTRY[resolved_key]
        kwargs = {**conf, **override_kwargs}
        provider = kwargs.pop('provider')
        model_name = kwargs.pop('model')
        temperature = kwargs.pop('temperature', 0.0)

        instance = create_llm_instance(
            provider= provider,
            model_name=model_name,
            temperature= temperature,
            **kwargs
        )
        if not override_kwargs :
            _llm_cache[resolved_key] = instance
        return instance


def clear_model_cache() -> None:
    """Clears internal LLM instance cache."""
    with _lock:
        _llm_cache.clear()
        log.info("LLM Model cache cleared.")
