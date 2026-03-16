# app/main.py

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        - user_question: the user's question
        - llm_choice: selected LLM model
        - embedding_choice: selected embedding model

    Returns:
        - LLM-generated answer
    """

    if pdf_file is None:
        return "Please upload a PDF file."

    if user_question.strip() == "":
        return "Please type a question."

    # Set default embedding if none provided
    if not embedding_choice:
        embedding_choice = "minilm"

    # Initialize LLM
    llm_model = get_llm()  # only one model, default Phi-3

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
        return_sources=True
    )

    answer = response["result"]
    sources = response["sources"]

    if sources:
        sources_text = "\n".join(sources)
        return f"{answer}\n\nSources:\n{sources_text}"

    return answer


# Modern Gradio interface using Blocks
with gr.Blocks(
    title="Offline PDF RAG Chatbot",
    theme=gr.themes.Soft(),
) as rag_ui:

    gr.Markdown(
        """
        # Offline PDF RAG Chatbot

        Ask questions about your PDF using a **fully local AI model (Phi-3 Mini)**.

        **Features**
        - Runs completely offline
        - Powered by Phi-3
        - Uses Retrieval-Augmented Generation (RAG)
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(
                label="Upload PDF",
                file_types=[".pdf"],
                type="filepath"
            )

            embedding_dropdown = gr.Dropdown(
                label="Select Embedding",
                choices=EMBEDDING_OPTIONS,
                value=EMBEDDING_OPTIONS[0]  # minilm
            )

        with gr.Column(scale=2):
            question_input = gr.Textbox(
                label="Ask a question about the document",
                placeholder="Example: What is an encoder?",
                lines=2
            )

            submit_button = gr.Button("Ask Question")

            answer_output = gr.Textbox(
                label="Answer",
                lines=10
            )

    gr.Markdown(
        """
        ---
        **Tech Stack**

        - Local LLM: Phi-3 Mini 4K Instruct  
        - Embeddings: MiniLM / MPNet  
        - Vector DB: Chroma  
        - Framework: LangChain  
        """
    )

    submit_button.click(
        fn=rag_qa,
        inputs=[pdf_input, question_input, embedding_dropdown],
        outputs=answer_output
    )

# Launch app
if __name__ == "__main__":
    rag_ui.launch(server_name="0.0.0.0", server_port=7860)