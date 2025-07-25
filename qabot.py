from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from ibm_watsonx_ai import Credentials
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA

import gradio as gr
import warnings


# Suppress warnings
def warn(*args, **kwargs):
    pass

warnings.warn = warn
warnings.filterwarnings('ignore')

# Initialize IBM Watsonx large language model (LLM) for text generation
def get_llm():
    model_id = 'mistralai/mixtral-8x7b-instruct-v01'
    parameters = {
        GenParams.MAX_NEW_TOKENS: 256,
        GenParams.TEMPERATURE: 0.5,
    }
    project_id = "skills-network"
    return WatsonxLLM(
        model_id=model_id,
        url="https://us-south.ml.cloud.ibm.com",
        project_id=project_id,
        params=parameters,
    )

# Load and parse PDF file into documents using LangChain's PDF loader
def document_loader(file):
    loader = PyPDFLoader(file.name)
    return loader.load()

# Split documents into smaller overlapping chunks for better retrieval
def text_splitter(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len,
    )
    return splitter.split_documents(documents)

# Initialize Watsonx embedding model for vector representations
def watsonx_embedding():
    embed_params = {
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
        EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
    }
    return WatsonxEmbeddings(
        model_id="ibm/slate-125m-english-rtrvr",
        url="https://us-south.ml.cloud.ibm.com",
        project_id="skills-network",
        params=embed_params,
    )

# Create Chroma vector database from text chunks using the embedding model
def vector_database(chunks):
    embedding_model = watsonx_embedding()
    return Chroma.from_documents(chunks, embedding_model)

# Construct a retriever object from a given PDF file
def retriever(file):
    documents = document_loader(file)
    chunks = text_splitter(documents)
    vectordb = vector_database(chunks)
    return vectordb.as_retriever()

# Main function: takes PDF and user query, returns LLM-generated answer using RAG
def retriever_qa(file, query):
    llm = get_llm()
    retriever_obj = retriever(file)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever_obj,
        return_source_documents=False,
    )
    response = qa.invoke(query)
    return response['result']

# Define the Gradio user interface for the RAG chatbot
rag_application = gr.Interface(
    fn=retriever_qa,
    allow_flagging="never",
    inputs=[
        gr.File(
            label="Upload PDF File",
            file_count="single",
            file_types=['.pdf'],
            type="filepath"
        ),
        gr.Textbox(
            label="Input Query",
            lines=2,
            placeholder="Type your question here..."
        )
    ],
    outputs=gr.Textbox(label="Output"),
    title="RAG Chatbot",
    description="Upload a PDF document and ask any question. The chatbot will try to answer using the provided document."
)


if __name__ == "__main__":
    rag_application.launch(server_name="0.0.0.0", server_port=7860)