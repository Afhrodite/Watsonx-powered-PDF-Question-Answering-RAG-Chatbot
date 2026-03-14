# rag/pipeline.py

from langchain.chains import RetrievalQA

def query(file_path, question, llm, retriever, chain_type="stuff", return_sources=False):
    """
    Generic RAG pipeline.

    Args:
        - file_path: path to the PDF or document
        - question: user query
        - llm: LLM model (from llm.py)
        - retriever: Retriever object (from retriever.py)
        - chain_type: type of RAG chain
        - return_sources: whether to return source documents

    Returns:
        - dict: {'result': answer}
    """
    
    # Build the RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=return_sources
    )

    # Run the question through the chain
    response = qa_chain.invoke(question)

    # Return the result
    return response