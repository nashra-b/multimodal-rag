"""
multimodal-rag
==============
Production-grade Multimodal RAG pipeline for PDF documents.
 
Packages:
    ingestion   — PDF parsing, table processing, image summarization, chunking
    embeddings  — OpenAI embedding generation with batching and retries
    vectorstore — Pinecone client, namespaced upsert, hybrid retriever
    pipeline    — End-to-end ingestion orchestrator and LangChain RAG chain
    app         — Streamlit chatbot UI
"""
 
__version__ = "1.0.0"
__author__  = "Nashra Babar"