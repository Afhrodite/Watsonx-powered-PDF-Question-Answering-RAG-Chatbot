# rag/retriever.py

from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rag.document_loaders import load_pdf
from rag.embeddings import get_embedding_model
# from langchain_community.vectorstores import FAISS, Qdrant

def get_retriever(file_path, embedding_model_name="watsonx", vector_db="chroma", project_id="skills-network"):
    """
    Returns a retriever object for a PDF file using a selected embedding model and vector DB.

    Embedding models:
        - watsonx (default - IBM Watsonx Slate embedding)
        - openai (OpenAI text-embedding-3-small)
        - bge-small (BigScience BGE embedding)

    Vector DB options:
        - chroma (default - Easy, simple, persistent in memory/disk)
        - faiss (Very fast, efficient, needs disk saving for persistence)
        - qdrant (Distributed, supports filtering, needs service running)
    """

    # Load PDF
    documents = load_pdf(file_path)

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    # Get embedding model
    embedding_model = get_embedding_model(embedding_model_name, project_id=project_id)

    # Create vector DB
    if vector_db.lower() == "chroma":
        vectordb = Chroma.from_documents(chunks, embedding_model)

    # elif vector_db.lower() == "faiss":
    #     vectordb = FAISS.from_documents(chunks, embedding_model)
    # elif vector_db.lower() == "qdrant":
    #     vectordb = Qdrant.from_documents(chunks, embedding_model)

    else:
        raise ValueError(f"Vector DB '{vector_db}' not supported. Default: chroma")

    # Return retriever
    return vectordb.as_retriever()