# rag/llm.py

from langchain_community.llms import LlamaCpp
import os


def get_llm(model_name="phi3", params=None):
    """
    Returns a local LLM.

    Supported:
        - phi3 (Phi-3 Mini GGUF)
    """

    if params is None:
        params = {
            "temperature": 0.5,
            "max_tokens": 512,
            "n_ctx": 4096
        }

    model_name = model_name.lower()

    if model_name == "phi3":
        model_path = os.path.join("models", "Phi-3-mini-4k-instruct-q4.gguf")

        return LlamaCpp(
            model_path=model_path,
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
            n_ctx=params["n_ctx"],
            verbose=False
        )

    else:
        raise ValueError(f"Unsupported LLM: {model_name}")