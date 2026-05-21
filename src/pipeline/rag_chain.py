"""
rag_chain.py — LangChain 1.x compatible using LCEL
"""

import os
import logging
from typing import Iterator
from langchain_core.documents                   import Document
from langchain_core.prompts                     import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers              import StrOutputParser
from langchain_core.runnables                   import RunnablePassthrough, RunnableLambda
from langchain_openai                           import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory

logger = logging.getLogger(__name__)

QA_SYSTEM = """You are an expert financial document analyst.
Answer ONLY from the provided context. Do not hallucinate.
For numeric questions, cite the exact figure and source page.
If context is insufficient, say so clearly."""

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", QA_SYSTEM),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Context:\n\n{context}\n\nQuestion: {question}"),
])


def format_docs(docs: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        m = doc.metadata
        parts.append(
            f"[CHUNK {i} | TYPE: {m.get('element_type','text').upper()} | "
            f"PAGE: {m.get('page_number','?')}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)


class RAGChain:
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
        self._last_docs    = []

        self.llm = ChatOpenAI(
            model          = model,
            temperature    = temperature,
            streaming      = streaming,
            openai_api_key = os.getenv("OPENAI_API_KEY"),
        )

        # LCEL chain
        self.chain = (
            RunnablePassthrough.assign(
                context = RunnableLambda(
                    lambda x: format_docs(self.retriever.invoke(x["question"]))
                )
            )
            | QA_PROMPT
            | self.llm
            | StrOutputParser()
        )

        logger.info(f"RAGChain ready | model={model}")

    def invoke(self, question: str) -> dict:
        messages = self.history.messages[-(self.memory_window * 2):]

        # Retrieve docs separately so we can return them
        self._last_docs = self.retriever.invoke(question)

        answer = self.chain.invoke({
            "question":    question,
            "chat_history": messages,
        })

        self.history.add_user_message(question)
        self.history.add_ai_message(answer)

        return {
            "answer":           answer,
            "source_documents": self._last_docs,
            "question":         question,
        }

    def stream(self, question: str) -> Iterator[str]:
        messages     = self.history.messages[-(self.memory_window * 2):]
        self._last_docs = self.retriever.invoke(question)
        full         = ""
        for token in self.chain.stream({"question": question, "chat_history": messages}):
            full += token
            yield token
        self.history.add_user_message(question)
        self.history.add_ai_message(full)

    def clear_memory(self) -> None:
        self.history.clear()

    def get_chat_history(self) -> list:
        return self.history.messages

    @staticmethod
    def format_sources(source_documents: list[Document]) -> list[dict]:
        seen, sources = set(), []
        for doc in source_documents:
            m   = doc.metadata
            key = (m.get("source"), m.get("page_number"), m.get("chunk_index", 0))
            if key in seen:
                continue
            seen.add(key)
            sources.append({
                "element_type": m.get("element_type", "text"),
                "page_number":  m.get("page_number"),
                "source":       m.get("source", "unknown"),
                "snippet":      doc.page_content[:200].strip() + "…",
                "chunk_index":  m.get("chunk_index", 0),
            })
        return sources

    @staticmethod
    def element_type_badge(element_type: str) -> str:
        return {"text": "📄 Text", "table": "📊 Table", "image": "🖼️ Image"}.get(
            element_type.lower(), "📄 Text"
        )
