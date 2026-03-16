# PDF Question-Answering RAG Chatbots

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)
![Gradio](https://img.shields.io/badge/Gradio-4.25%2B-orange?logo=gradio)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green?logo=langchain)
![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)

_Created by **Réka Gábosi**_

## Table of Contents

- [Description](#description)
- [Included Chatbots](#included-chatbots)
- [Local PDF RAG Chatbot](#local-pdf-rag-chatbot)
  - [Screenshots](#screenshots-local)
  - [How to run](#how-to-run-local)
  - [Deployment](#deployment-local)
- [Watsonx-powered PDF RAG Chatbot](#watsonx-powered-pdf-rag-chatbot)
  - [Screenshots](#screenshots-watsonx)
  - [How to run](#how-to-run-watsonx)
  - [Deployment](#deployment-watsonx)
- [Future Ideas](#future-ideas)
- [License](#license)
- [Acknowledgements](#acknowledgements)
---

## Description

This repository contains two versions of a **PDF Question-Answering chatbot** using **RAG (Retrieval-Augmented Generation)**:  

1. **Local Model Chatbot** – Runs entirely locally with a GGUF LLM (Phi-3 Mini 4K).  
2. **Watsonx-powered Chatbot** – Uses IBM Watsonx LLM and embeddings via API.  

Both allow users to:

- Upload PDF files  
- Ask natural language questions about the content  
- Get answers grounded in the document chunks  

---

## Included Chatbots

| Chatbot | Model | Deployment | Notes |
|---------|-------|-----------|-------|
| Local PDF RAG Chatbot | Phi-3 Mini 4K GGUF | Docker + Kubernetes | Runs fully offline |
| Watsonx PDF RAG Chatbot | IBM Watsonx Mixtral-8x7B | Docker only | Requires IBM API keys |

---

## Local PDF RAG Chatbot

**Screenshots**

### Launch Screen
![App Launch](/local-model-rag-chatbot/images/1.png)

### Question & Answer
![Q&A Example](/local-model-rag-chatbot/images/2.png)

**Features:**

- Uses `langchain_classic` and `LlamaCpp` to run a **local GGUF LLM**.  
- Supports embeddings via `MiniLM` or `MPNet`.  
- Full **offline RAG pipeline** with Chroma vector store.  
- Simple Gradio interface for PDF Q&A.

### How to run (local)

1. Make sure the model is downloaded to `models/Phi-3-mini-4k-instruct-q4.gguf`.  
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python app/main.py
```

4. Open your browser:

```bash
http://localhost:7860
```

### Deployment (local)

#### Docker

```bash
docker build -t local-rag-chatbot .
docker run -p 7860:7860 local-rag-chatbot
```

#### Kubernetes

A simple k8s/deployment.yaml:

```bash
apiVersion: apps/v1
kind: Deployment
metadata:
  name: local-rag-chatbot
spec:
  replicas: 1
  selector:
    matchLabels:
      app: local-rag-chatbot
  template:
    metadata:
      labels:
        app: local-rag-chatbot
    spec:
      containers:
      - name: rag-chatbot
        image: local-rag-chatbot:latest
        ports:
        - containerPort: 7860
        env:
        - name: GRADIO_SERVER_NAME
          value: "0.0.0.0"
        - name: GRADIO_SERVER_PORT
          value: "7860"
---
apiVersion: v1
kind: Service
metadata:
  name: local-rag-chatbot-service
spec:
  type: NodePort
  selector:
    app: local-rag-chatbot
  ports:
    - protocol: TCP
      port: 7860
      targetPort: 7860
```

This allows you to deploy the local RAG chatbot on any Kubernetes cluster.

---

## Watsonx-powered PDF RAG Chatbot

**Screenshots**

### Launch Screen
![App Launch](/watsonx-rag-chatbot/images_and_pdf/example1.png)

### Question & Answer
![Q&A Example](/watsonx-rag-chatbot/images_and_pdf/example2.png)

**Features:**

- Uses IBM **Watsonx Mixtral-8x7B** LLM via API.
- Embeddings generated with Watsonx embedding model.
- LangChain RAG pipeline with Chroma vector store.
- Gradio interface for PDF Q&A.
- Requires **IBM API credentials** (WATSONX_API_KEY, WATSONX_URL, PROJECT_ID).

### How to run (Watsonx)

1. Set environment variables:

```bash
export WATSONX_API_KEY="your_api_key"
export WATSONX_URL="https://us-south.ml.cloud.ibm.com"
export PROJECT_ID="skills-network"
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python app/watsonx_main.py
```

4. Open your browser:

```bash
http://localhost:7860
```

### Deployment (Watsonx)

#### Docker

```bash
docker build -t watsonx-rag-chatbot -f Dockerfile.watsonx .
docker run -p 7860:7860 \
  -e WATSONX_API_KEY="your_api_key" \
  -e WATSONX_URL="https://us-south.ml.cloud.ibm.com" \
  -e PROJECT_ID="skills-network" \
  watsonx-rag-chatbot
```

Kubernetes deployment is not required since the app relies on API keys and external services.

## Future Ideas

- Support multiple PDFs per query
- Add history/logging of previous Q&A sessions
- Improve UI with semantic search previews
- Benchmark Watsonx vs. local GGUF models
- Deploy local RAG chatbot to Hugging Face Spaces

# Acknowledgements

- **IBM** – for the RAG course and Watsonx APIs
- **LangChain** – for pipelines, loaders, splitters, and RetrievalQA
- **Gradio** – for building interactive interfaces quickly
- **Mixtral / Phi-3** – LLMs used in the two chatbots