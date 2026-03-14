# rag/config.py

LLM_OPTIONS = [
    "mixtral",    
    "gpt4all",     
    "bge-small",  
    "oasst",       
    "mpt-7b",      
    "vicuna-7b",   
]

EMBEDDING_OPTIONS = [
    "watsonx",   
    "openai",    
    "bge-small"  
]

# Set to default
VECTOR_DB = "chroma"