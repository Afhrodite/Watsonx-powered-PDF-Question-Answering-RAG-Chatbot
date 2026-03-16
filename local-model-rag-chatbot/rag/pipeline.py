# rag/pipeline.py

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from rag.config import SYSTEM_PROMPT
from utils.logger import logger
import time
import traceback

def get_rag_chain(llm, retriever):
    """
    Build a modern RAG chain using create_retrieval_chain.
    """
    # Build system prompt for the LLM
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("system", "Context:\n{context}"),
        ("human", "{input}")
    ])
    
    # Build the QA chain (stuff chain)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    # Combine retriever with QA chain
    return create_retrieval_chain(retriever, question_answer_chain)


def query(file_path, question, llm, retriever, return_sources=False):
    """
    Generic RAG pipeline.

    Args:
        - file_path: path to the PDF or document
        - question: user query
        - llm: LLM model (from llm.py)
        - retriever: Retriever object (from retriever.py)
        - return_sources: whether to return source documents

    Returns:
        - dict: {'result': answer, 'sources': list}
    """

    # Start timer to measure total RAG processing time
    start_time = time.time()

    # Log the user's question for debugging and monitoring
    logger.info(f"Starting query for file: {file_path}, question: {question}")

    try:
        # Build the RAG chain
        chain = get_rag_chain(llm, retriever)
        logger.info("RAG chain built successfully.")
        # Run the question through the chain
        response = chain.invoke({"input": question})
        logger.info("RAG chain invoked successfully.")

    except Exception as e:
        logger.error("Error during RAG query:")
        tb = traceback.format_exc()
        logger.error(tb)
        raise RuntimeError(f"RAG query failed: {e}\nFull traceback:\n{tb}")


    # Log total response time for performance monitoring
    end_time = time.time()
    logger.info(f"Total response time: {end_time - start_time:.2f}s")

    # Extract answer
    answer = response["answer"]

    # Extract sources if requested
    sources = []
    if return_sources and "source_documents" in response:
        for doc in response["source_documents"]:
            page = doc.metadata.get("page", "Unknown")
            sources.append(f"Page {page}")

    # Remove duplicates
    sources = list(set(sources))

    return {
        "result": answer,
        "sources": sources
    }