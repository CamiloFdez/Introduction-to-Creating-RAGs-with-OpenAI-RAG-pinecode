# Introduction-to-Creating-RAGs-with-OpenAI-RAG-pinecode

## Overview
This project implements a **Retrieval-Augmented Generation (RAG)** system using the **LangChain framework** and **Pinecone** as a vector database.

RAG combines **information retrieval** with **text generation**, allowing a language model to answer questions using relevant external documents instead of relying only on its internal knowledge. This approach improves accuracy and reduces hallucinations.

This repository corresponds to **Repository 2** of the lab assignment: *Introduction to Creating RAGs*.

---

## Architecture Overview

The system follows a standard RAG pipeline:

1. **Document Ingestion**
   - Documents are loaded from local files
   - Text is split into chunks
   - Embeddings are generated
   - Chunks are stored in Pinecone

2. **Retrieval**
   - A user query is embedded
   - Pinecone retrieves the most relevant document chunks

3. **Generation**
   - Retrieved context is injected into a prompt
   - A language model generates an answer grounded in the retrieved context

### Components Used
- **LangChain**: Orchestration framework
- **Pinecone**: Vector database
- **HuggingFace Embeddings**: Sentence-transformers
- **HuggingFace LLM**: GPT-2 (text generation)

---

## Note on OpenAI Usage

Although the lab statement suggests using **OpenAI for embeddings and LLMs**, this implementation uses **open-source HuggingFace models** instead.

This decision was made to:
- Avoid paid API dependencies
- Ensure the project is fully reproducible without external costs
- Focus on understanding the **core RAG architecture**, which is model-agnostic

The architecture and workflow remain identical to an OpenAI-based RAG:
- Dense embeddings
- Vector similarity search
- Context-augmented generation

Although this we made a notebook with the OpenAI implementation, we decided to keep this repository focused on the open-source version for simplicity and accessibility. And as you will see in the notebook this works just as well as the OpenAI version, with the added benefit of being free to use and modify.

---

## Project Structure

```text
├── src/
│   ├── ingest.py       # Document ingestion pipeline
│   ├── rag.py          # RAG pipeline implementation
├── data/
│   └── sample.txt      # Sample document for ingestion
├── .env                # Environment variables (Pinecone API keys)
├── README.md           # Project documentation
└── requirements.txt    # Python dependencies
```

---

## Installation

1. Clone the repository
```bash
git clone https://github.com/your-username/your-rag-repo.git
cd your-rag-repo
```
2. Create a virtual environment and install dependencies
```bash
python -m venv venv
source venv/bin/activate  
pip install -r requirements.txt
```

---

## Environment Variables
Create a `.env` file in the root directory with your Pinecone API credentials:
```env
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_ENVIRONMENT=your-pinecone-environment
PINECONE_INDEX_NAME=your-pinecone-index-name
```

---

## Document Ingestion
Run the ingestion script to load documents into Pinecone:
```bash
python src/ingest.py
```
This will:
- Load `data/sample.txt`
- Split it into chunks
- Generate embeddings

Here is the image of the ingestion process:
![image](https://github.com/CamiloFdez/Introduction-to-Creating-RAGs-with-OpenAI-RAG-pinecode/blob/main/images/ingestPy.PNG)

---

## RAG Pipeline
Run the RAG pipeline to answer a query:
```bash
python src/rag.py
```
Example query: "What is the main topic of the document?"
Expected output: A generated answer based on the retrieved context from Pinecone.

Here is the image of the RAG process:
![image](https://github.com/CamiloFdez/Introduction-to-Creating-RAGs-with-OpenAI-RAG-pinecode/blob/main/images/ragPy.PNG)

---

## Conclusion
This project demonstrates how to build a RAG system using LangChain and Pinecone, with open-source models for embeddings and generation. The architecture is flexible and can be easily adapted to use OpenAI models if desired.