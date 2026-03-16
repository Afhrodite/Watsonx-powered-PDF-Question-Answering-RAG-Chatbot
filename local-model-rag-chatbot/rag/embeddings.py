# rag/embeddings.py

from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_model(model_name="all-MiniLM-L6-v2", params=None):
    """
    Returns a local embedding model.

    Options:
        - all-MiniLM-L6-v2: fast, lightweight, free
        - all-mpnet-base-v2: optional higher quality
    """
    if params is None:
        params = {}

    model_name = model_name.lower()

    if model_name in ["all-minilm-l6-v2", "minilm"]:
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs=params)
    
    elif model_name in ["all-mpnet-base-v2", "mpnet"]:
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs=params)

    else:
        raise ValueError(
            f"Embedding '{model_name}' not supported. Available: minilm, mpnet"
        )