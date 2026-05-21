# retriever.py
from langchain_community.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

NAMESPACES = ["pdf-text", "pdf-tables", "pdf-images"]

class HybridRetriever:
    """
    Combines dense vector search (Pinecone) with sparse BM25.
    Queries all three namespaces (text, tables, images) so no content is missed.
    """
    def __init__(self, pinecone_index, all_docs):
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

        # One retriever per namespace so all ingested content is searchable
        namespace_retrievers = [
            PineconeVectorStore(
                index=pinecone_index,
                embedding=embeddings,
                namespace=ns,
            ).as_retriever(
                search_type="mmr",
                search_kwargs={"k": 4, "fetch_k": 12, "lambda_mult": 0.7}
            )
            for ns in NAMESPACES
        ]

        dense_retriever = EnsembleRetriever(
            retrievers=namespace_retrievers,
            weights=[0.5, 0.35, 0.15],  # text, tables, images
        )

        if all_docs:
            sparse_retriever = BM25Retriever.from_documents(all_docs)
            sparse_retriever.k = 4
            self.retriever = EnsembleRetriever(
                retrievers=[dense_retriever, sparse_retriever],
                weights=[0.7, 0.3]
            )
        else:
            self.retriever = dense_retriever
