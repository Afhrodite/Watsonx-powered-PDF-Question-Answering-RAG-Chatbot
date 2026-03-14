# app/main.py

import gradio as gr
from rag.config import LLM_OPTIONS, EMBEDDING_OPTIONS, VECTOR_DB
from rag.llm import get_llm
from rag.embeddings import get_embedding_model
from rag.retriever import get_retriever
from rag.pipeline import query


def rag_qa(pdf_file, user_question, llm_choice, embedding_choice):
    """
    Main RAG function for Gradio.

    Inputs:
        - pdf_file: uploaded PDF
        - user_question: the users question
        - llm_choice: selected LLM model
        - embedding_choice: selected embedding model

    Returns:
        - LLM-generated answer
    """

    if pdf_file is None:
        return "Please upload a PDF file."

    if user_question.strip() == "":
        return "Please type a question."

    # Initialize LLM
    llm_model = get_llm(model_name=llm_choice)

    # Create retriever
    retriever_obj = get_retriever(
        file_path=pdf_file.name,
        embedding_model_name=embedding_choice,
        vector_db=VECTOR_DB
    )

    # Run question through RAG pipeline
    response = query(
        file_path=pdf_file.name,
        question=user_question,
        llm=llm_model,
        retriever=retriever_obj,
        return_sources=False
    )

    return response["result"]



# Gradio Interface
rag_ui = gr.Interface(
    # Whenever a user submits an input the function 'rag_qa' gets called
    fn=rag_qa,
    inputs=[
        # File upload button
        gr.File(label="Upload PDF", file_types=[".pdf"], type="file"),
        # Text field for the users question
        gr.Textbox(label="Your Question", placeholder="Type your question here...", lines=2),
        # Dropdown for the user to choose an LLM model
        gr.Dropdown(label="Select LLM", choices=LLM_OPTIONS, value="mixtral"),
        # Dropdown for the user to choose an Embedding model
        gr.Dropdown(label="Select Embedding", choices=EMBEDDING_OPTIONS, value="watsonx")
    ],
    # To display the answer
    outputs=gr.Textbox(label="Answer"),
    title="PDF Question Answering RAG Chatbot",
    description=(
        "Upload a PDF, select an LLM and embedding model, and ask a question. "
        "The chatbot will answer using the uploaded document."
    ),
    # Remove Gradio's report button
    allow_flagging="never",
)

# Launch app
if __name__ == "__main__":
    rag_ui.launch(server_name="0.0.0.0", server_port=7860)