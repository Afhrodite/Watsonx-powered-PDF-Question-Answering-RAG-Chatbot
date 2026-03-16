# rag/config.py

LLM_OPTIONS = [
    "phi3"
]

EMBEDDING_OPTIONS = [
    "minilm",
    "mpnet"
]

VECTOR_DB = "chroma"

VECTOR_DB_PATH = "vector_store"

SYSTEM_PROMPT = """
You are a helpful AI assistant that answers questions using the provided document context.

Rules:
- Only use the provided context.
- If the answer is not in the context, say "I don't know".
- Be concise.
"""
