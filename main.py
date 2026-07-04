from dotenv import load_dotenv
from pathlib import Path
import tempfile
import os

from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv(dotenv_path=Path(".env"))

print(os.getenv("GOOGLE_API_KEY"))


# Embedding Model
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview"
)


# Knowledge Base
KNOWLEDGE_BASE = """
LangChain is an open-source framework for building applications powered by large language models (LLMs). It provides modular components that simplify the development of AI applications such as chatbots, Retrieval-Augmented Generation (RAG) systems, AI agents, document question-answering systems, and workflow automation.

The framework includes integrations for multiple LLM providers like Google Gemini, OpenAI, Anthropic Claude, and local models. LangChain offers document loaders to import data from PDFs, Word files, web pages, databases, and text files. Text splitters divide large documents into smaller chunks, making them easier for embedding models to process.

Embeddings convert text into numerical vectors that capture semantic meaning. These vectors are stored in vector databases such as Chroma, FAISS, Pinecone, Weaviate, or Milvus. During retrieval, the user's query is also converted into an embedding, and the most semantically similar document chunks are retrieved using similarity search.

In a Retrieval-Augmented Generation (RAG) pipeline, LangChain retrieves relevant context from the vector database and sends it to the language model along with the user's question. This approach reduces hallucinations and enables the model to answer questions based on custom knowledge instead of relying only on its pre-trained data.

LangChain also provides prompt templates, output parsers, retrievers, chains, memory, tools, agents, and integration with LangGraph for creating stateful AI workflows. Developers can combine these components to build scalable, production-ready AI applications for customer support, enterprise search, legal document analysis, healthcare, education, finance, and internal knowledge management.

Overall, LangChain serves as the orchestration layer that connects language models, external knowledge sources, vector databases, APIs, and application logic into a unified AI system.
"""


def create_kb():
    """Create a vector store from the knowledge base."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    doc = Document(
        page_content=KNOWLEDGE_BASE,
        metadata={"source": "langchain_knowledge_base.md"},
    )

    chunks = splitter.split_documents([doc])

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory=tempfile.mkdtemp(),
    )

    return vector_store


def demo_basic_rag():

    vector_store = create_kb()

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    llm = init_chat_model(
        model="gemini-3.1-flash-lite",
        model_provider='google_genai',
        temperature=0.2,
    )

    prompt = ChatPromptTemplate.from_template(
        """
Answer the question based only on the following context.

Context:
{context}

Question:
{question}

Answer:
Answer concisely.
If you don't know the answer, simply say "I don't know."
"""
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    questions = [
        "What is LangChain?",
        "How does RAG work?",
        "Why do some developers dislike LangChain?",
    ]

    print("========== Basic RAG Demo ==========\n")

    for q in questions:
        answer = rag_chain.invoke(q)
        print(f"Q: {q}")
        print(f"A: {answer}\n")


if __name__ == "__main__":
    demo_basic_rag()