# rag/retriever.py

from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loaders.pdf_loader import load_pdf
from rag.embeddings import get_embedding_model
from rag.config import VECTOR_DB_PATH
import os

def get_retriever(file_path, embedding_model_name="minilm", vector_db="chroma"):
    # Load PDF
    documents = load_pdf(file_path)

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    # Local embeddings
    embedding_model = get_embedding_model(embedding_model_name)

    # Chroma vector store
    persist_dir = os.path.join(VECTOR_DB_PATH, os.path.basename(file_path))
    if os.path.isdir(persist_dir):
        vectordb = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
    else:
        vectordb = Chroma.from_documents(chunks, embedding_model, persist_directory=persist_dir)

    return vectordb.as_retriever()