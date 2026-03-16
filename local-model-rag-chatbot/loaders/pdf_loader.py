# loaders/pdf_loader.py

from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.document_loaders import DocxLoader, TextLoader


def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()

# def load_docx(file_path):
#     loader = DocxLoader(file_path)
#     return loader.load()

# def load_txt(file_path):
#     loader = TextLoader(file_path)
#     return loader.load()