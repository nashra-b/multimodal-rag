"""
rag_chain.py
------------
LangChain RAG chain with:
  - Conversational memory (last N exchanges)
  - Source document citations with element-type badges
  - Namespace-aware routing (text / table / image)
  - Streaming support for Streamlit

Usage:
    chain  = RAGChain(retriever=hybrid_retriever.retriever)
    result = chain.invoke("What was net income in Q3?")
    # result → {"answer": str, "source_documents": [Document]}
"""

import os
import logging
from typing import Iterator, Optional

from langchain_core.documents        import Document
from langchain_core.prompts          import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers   import StrOutputParser
from langchain_core.runnables        import RunnablePassthrough, RunnableLambda
from langchain_openai                import ChatOpenAI
from langchain.memory                import ConversationBufferWindowMemory
from langchain.chains                import ConversationalRetrievalChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm            import LLMChain

logger = logging.getLogger(__name__)


# ── Prompt Templates ───────────────────────────────────────────────────────────

# System prompt tuned for banking / financial document Q&A
SYSTEM_PROMPT = """You are an expert financial document analyst assistant.
You answer questions based ONLY on the provided context retrieved from PDF documents.

Context may include three types of content — treat each carefully:
  • TEXT    : Narrative sections, summaries, management commentary
  • TABLE   : Structured financial data (always prefer tables for numeric questions)
  • IMAGE   : Visual summaries of charts, graphs, or diagrams

Rules:
  1. Answer only from the provided context. Do not hallucinate.
  2. For numeric / financial questions, cite the exact figure and its source page.
  3. If the answer spans multiple sources, synthesise them coherently.
  4. If the context does not contain enough information, say so explicitly — do NOT invent citations.
  5. After each fact, cite its source using the TYPE and PAGE values from the chunk header.
     Example citation format: Source: TEXT, Page 12
  6. Be concise but complete. Use bullet points for multi-part answers.
"""

CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Given the conversation history and a follow-up question, "
     "rewrite the follow-up question to be a fully self-contained question. "
     "Do not answer the question — only rewrite it. "
     "If no rewrite is needed, return the question unchanged."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human",
     "Context from the document:\n\n{context}\n\n"
     "Question: {question}"),
])


# ── Context formatter ──────────────────────────────────────────────────────────

def format_docs(docs: list[Document]) -> str:
    """
    Format retrieved documents into a single context string.
    Each chunk is prefixed with its element type and page number
    so the LLM can cite them accurately.
    """
    parts = []
    for i, doc in enumerate(docs, 1):
        meta         = doc.metadata
        element_type = meta.get("element_type", "text").upper()
        page         = meta.get("page_number", "N/A")
        source       = meta.get("source", "unknown")
        chunk_idx    = meta.get("chunk_index", 0)

        header = (
            f"[CHUNK {i} | TYPE: {element_type} | "
            f"PAGE: {page} | FILE: {source} | CHUNK: {chunk_idx}]"
        )
        parts.append(f"{header}\n{doc.page_content}")

    return "\n\n---\n\n".join(parts)


# ── RAG Chain ─────────────────────────────────────────────────────────────────

class RAGChain:
    """
    Conversational RAG chain wrapping LangChain's ConversationalRetrievalChain.

    Supports:
      - Multi-turn conversation with windowed memory
      - Streaming token output for Streamlit
      - Source document passthrough for citation display
      - Configurable LLM model and temperature

    Args:
        retriever       : LangChain BaseRetriever (hybrid or vector-only)
        model           : OpenAI model name
        temperature     : LLM temperature (0 = deterministic, best for RAG)
        memory_window   : Number of past exchanges to keep in context
        streaming       : Enable token streaming
    """

    def __init__(
        self,
        retriever,
        model:          str   = "gpt-4o",
        temperature:    float = 0.0,
        memory_window:  int   = 5,
        streaming:      bool  = False,
    ):
        self.retriever    = retriever
        self.model        = model
        self.temperature  = temperature
        self.memory_window = memory_window

        # ── LLM ───────────────────────────────────────────────────────────────
        self.llm = ChatOpenAI(
            model       = model,
            temperature = temperature,
            streaming   = streaming,
            openai_api_key = os.getenv("OPENAI_API_KEY"),
        )

        # ── Memory ────────────────────────────────────────────────────────────
        self.memory = ConversationBufferWindowMemory(
            k                    = memory_window,
            memory_key           = "chat_history",
            return_messages      = True,
            output_key           = "answer",    # which chain output to store
        )

        # ── Chain ─────────────────────────────────────────────────────────────
        self.chain = self._build_chain()

        logger.info(
            f"RAGChain ready | model={model} | "
            f"temperature={temperature} | memory_window={memory_window}"
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def invoke(self, question: str) -> dict:
        """
        Ask a question. Returns:
            {
                "answer"           : str,
                "source_documents" : list[Document],
                "question"         : str,
            }
        """
        logger.debug(f"RAGChain.invoke: {question[:80]}")
        result = self.chain.invoke({"question": question})
        return {
            "answer":           result.get("answer", ""),
            "source_documents": result.get("source_documents", []),
            "question":         question,
        }

    def stream(self, question: str) -> Iterator[str]:
        """
        Stream answer tokens. Yields str tokens one at a time.
        Use in Streamlit with st.write_stream().

        Note: source_documents not available in stream mode.
              Call invoke() after streaming for citations.
        """
        logger.debug(f"RAGChain.stream: {question[:80]}")
        for chunk in self.chain.stream({"question": question}):
            token = chunk.get("answer", "")
            if token:
                yield token

    def clear_memory(self) -> None:
        """Reset conversation history."""
        self.memory.clear()
        logger.info("Conversation memory cleared.")

    def get_chat_history(self) -> list:
        """Return the current conversation history messages."""
        return self.memory.chat_memory.messages

    # ── Chain construction ─────────────────────────────────────────────────────

    def _build_chain(self) -> ConversationalRetrievalChain:
        """
        Build a ConversationalRetrievalChain with:
          - Question condensation (turns follow-ups into standalone questions)
          - Document retrieval
          - QA with injected chat history
          - Source document passthrough
        """
        # LLM for condensing follow-up questions into standalone questions
        condense_llm = ChatOpenAI(
            model          = self.model,
            temperature    = 0.0,         # always deterministic for question rewrite
            openai_api_key = os.getenv("OPENAI_API_KEY"),
        )

        chain = ConversationalRetrievalChain.from_llm(
            llm                       = self.llm,
            retriever                 = self.retriever,
            memory                    = self.memory,
            condense_question_llm     = condense_llm,
            condense_question_prompt  = CONDENSE_QUESTION_PROMPT,
            get_chat_history          = lambda h: h,  # keep as BaseMessage list for MessagesPlaceholder
            combine_docs_chain_kwargs = {
                "prompt":           QA_PROMPT,
                "document_variable_name": "context",
                "document_separator": "\n\n---\n\n",
            },
            return_source_documents   = True,
            verbose                   = False,
        )

        return chain

    # ── Source formatting helpers (used by Streamlit UI) ──────────────────────

    @staticmethod
    def format_sources(source_documents: list[Document]) -> list[dict]:
        """
        Convert source Documents into a list of dicts for display in the UI.

        Returns:
            [
                {
                    "element_type": "table",
                    "page_number" : 12,
                    "source"      : "annual_report.pdf",
                    "snippet"     : "First 200 chars of content…",
                    "chunk_index" : 0,
                },
                ...
            ]
        """
        seen    = set()
        sources = []

        for doc in source_documents:
            meta         = doc.metadata
            element_type = meta.get("element_type", "text")
            page         = meta.get("page_number")
            source       = meta.get("source", "unknown")
            chunk_idx    = meta.get("chunk_index", 0)

            # Deduplicate by (source, page, chunk_index)
            key = (source, page, chunk_idx)
            if key in seen:
                continue
            seen.add(key)

            sources.append({
                "element_type": element_type,
                "page_number":  page,
                "source":       source,
                "snippet":      doc.page_content[:200].strip() + "…",
                "chunk_index":  chunk_idx,
            })

        return sources

    @staticmethod
    def element_type_badge(element_type: str) -> str:
        """Return an emoji badge for a given element type."""
        badges = {
            "text":  "📄 Text",
            "table": "📊 Table",
            "image": "🖼️ Image",
        }
        return badges.get(element_type.lower(), "📄 Text")
