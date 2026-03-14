# rag/embeddings.py

from langchain_ibm import WatsonxEmbeddings
from langchain.embeddings import OpenAIEmbeddings 
from langchain_hub import HuggingFaceEmbeddings   


def get_embedding_model(model_name="watsonx", project_id="skills-network", params=None):
    """
    Returns a LangChain embedding model based on user selection.
    
    Supported models like:
        - watsonx: IBM Watsonx Slate embedding (default)
        - openai: OpenAI text-embedding-3-small
        - bge-small: BigScience BGE embedding
    """

    # Default parameters (truncates if its too long/ return additional info)
    if params is None:
        params = {"truncate_input_tokens": 3, "return_options": {"input_text": True}}

    model_name = model_name.lower()

    # IBM Slate 125M English Retrieval model (125M params)
    if model_name == "watsonx":
        return WatsonxEmbeddings(
            model_id="ibm/slate-125m-english-rtrvr",
            url="https://us-south.ml.cloud.ibm.com",
            project_id=project_id,
            params=params,
        )

    # OpenAI text embedding 3 small (1.5B params)
    elif model_name == "openai":
        return OpenAIEmbeddings(model="text-embedding-3-small", model_kwargs=params)
    
    # BigScience BGE Small (1.2B params)
    elif model_name == "bge-small":
        return HuggingFaceEmbeddings(repo_id="bigscience/bge-small", model_kwargs=params)

    else:
        raise ValueError(
            f"The Embedding model '{model_name}' not supported. Available models: watsonx, openai, bge-small"
        )