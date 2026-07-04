# Basic RAG with LangChain, Gemini, and ChromaDB

A minimal implementation of a Retrieval-Augmented Generation (RAG) pipeline using LangChain, Google Gemini, and ChromaDB. This project demonstrates the core concepts behind building a RAG application, including document chunking, vector embeddings, semantic retrieval, and response generation using a large language model.

## Overview

This project implements a complete RAG workflow:

- Create a knowledge base
- Split documents into semantic chunks
- Generate vector embeddings with Google Gemini
- Store embeddings in ChromaDB
- Retrieve relevant context using similarity search
- Generate grounded responses using Gemini

## Features

- LangChain-based RAG pipeline
- Google Gemini language model integration
- Gemini Embeddings (`gemini-embedding-2-preview`)
- Chroma vector database
- Recursive text splitting
- Semantic similarity retrieval
- Prompt templating
- Environment variable configuration

## Tech Stack

- Python
- LangChain
- Google Gemini API
- ChromaDB
- python-dotenv

## Project Structure

```text
Basic-Rag/
│
├── main.py
├── .env
├── .gitignore
├── pyproject.toml
├── uv.lock
└── README.md
```

## Installation

### Clone the repository

```bash
git clone https://github.com/PremNarvekar/Basic-Rag.git
cd Basic-Rag
```

### Create a virtual environment

```bash
uv venv
```

### Activate the virtual environment

**Windows**

```powershell
.venv\Scripts\Activate.ps1
```

**macOS/Linux**

```bash
source .venv/bin/activate
```

### Install dependencies

```bash
uv sync
```

or

```bash
uv add langchain langchain-core langchain-google-genai langchain-chroma langchain-text-splitters chromadb python-dotenv
```

## Environment Variables

Create a `.env` file in the project root.

```env
GOOGLE_API_KEY=YOUR_API_KEY
```

## Running the Project

```bash
python main.py
```

## RAG Architecture

```text
Knowledge Base
      │
      ▼
Document
      │
      ▼
Text Splitter
      │
      ▼
Gemini Embeddings
      │
      ▼
ChromaDB
      │
      ▼
Retriever
      │
      ▼
Relevant Context
      │
      ▼
Prompt Template
      │
      ▼
Gemini LLM
      │
      ▼
Generated Response
```

## Concepts Demonstrated

- Retrieval-Augmented Generation (RAG)
- Vector Embeddings
- Semantic Search
- Vector Databases
- Document Chunking
- Prompt Engineering
- Similarity-Based Retrieval
- LangChain Expression Language (LCEL)

## Future Improvements

- PDF document loader
- DOCX and text document support
- Website ingestion
- Persistent vector database
- Metadata filtering
- Source attribution
- Conversational memory
- Multi-document knowledge base
- LangGraph integration
- Production deployment

## License

This project is licensed under the MIT License.

## Author

**Prem Narvekar**

AI Engineer

GitHub: https://github.com/PremNarvekar
