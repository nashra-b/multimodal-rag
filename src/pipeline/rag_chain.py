"""
rag_chain.py
------------
LangChain RAG chain — compatible with LangChain 1.x (modern LCEL approach).
No deprecated langchain.memory or langchain.chains imports.
"""

import os
import logging
from typing import Iterator, Optional

from langchain_core.documents                   import Document
from langchain_core.prompts                     import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai                           import ChatOpenAI
from langchain.chains                           import ConversationalRetrievalChain
from langchain_community.chat_message_histories import ChatMessageHistory

logger = logging.getLogger(__name__)

# ── Prompts ────────────────────────────────────────────────────────────────────

CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Given the conversation history and a follow-up question, "
     "rewrite the follow-up as a fully self-contained question. "
     "If no rewrite is needed, return it unchanged."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

QA_SYSTEM = """You are an expert financial document analyst.
Answer ONLY from the provided context. Do not hallucinate.
For numeric questions, cite the exact figure and source page.
If context is insufficient, say so clearly.
End each fact with: Source: [element_type], Page [page_number]."""

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", QA_SYSTEM),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Context:\n\n{context}\n\nQuestion: {question}"),
])


class RAGChain:
    """
    Conversational RAG chain using ConversationalRetrievalChain
    with manual chat history management (LangChain 1.x compatible).
    """

    def __init__(
        self,
        retriever,
        model:         str   = "gpt-4o",
        temperature:   float = 0.0,
        memory_window: int   = 5,
        streaming:     bool  = False,
    ):
        self.retriever     = retriever
        self.memory_window = memory_window
        self.history       = ChatMessageHistory()

        self.llm = ChatOpenAI(
            model          = model,
            temperature    = temperature,
            streaming      = streaming,
            openai_api_key = os.getenv("OPENAI_API_KEY"),
        )

        self.chain = ConversationalRetrievalChain.from_llm(
            llm                    = self.llm,
            retriever              = self.retriever,
            return_source_documents= True,
            verbose                = False,
        )

        logger.info(f"RAGChain ready | model={model} | temperature={temperature}")

    # ── Public API ─────────────────────────────────────────────────────────────

    def invoke(self, question: str) -> dict:
        # Keep only last N exchanges
        messages = self.history.messages[-(self.memory_window * 2):]

        result = self.chain.invoke({
            "question":    question,
            "chat_history": messages,
        })

        self.history.add_user_message(question)
        self.history.add_ai_message(result.get("answer", ""))

        return {
            "answer":           result.get("answer", ""),
            "source_documents": result.get("source_documents", []),
            "question":         question,
        }

    def stream(self, question: str) -> Iterator[str]:
        """Streaming mode — yields tokens, then stores in history."""
        messages = self.history.messages[-(self.memory_window * 2):]
        full     = ""
        for chunk in self.chain.stream({"question": question, "chat_history": messages}):
            token = chunk.get("answer", "")
            if token:
                full += token
                yield token
        self.history.add_user_message(question)
        self.history.add_ai_message(full)

    def clear_memory(self) -> None:
        self.history.clear()

    def get_chat_history(self) -> list:
        return self.history.messages

    # ── Source helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def format_sources(source_documents: list[Document]) -> list[dict]:
        seen, sources = set(), []
        for doc in source_documents:
            meta = doc.metadata
            key  = (meta.get("source"), meta.get("page_number"), meta.get("chunk_index", 0))
            if key in seen:
                continue
            seen.add(key)
            sources.append({
                "element_type": meta.get("element_type", "text"),
                "page_number":  meta.get("page_number"),
                "source":       meta.get("source", "unknown"),
                "snippet":      doc.page_content[:200].strip() + "…",
                "chunk_index":  meta.get("chunk_index", 0),
            })
        return sources

    @staticmethod
    def element_type_badge(element_type: str) -> str:
        return {"text": "📄 Text", "table": "📊 Table", "image": "🖼️ Image"}.get(
            element_type.lower(), "📄 Text"
        )
