PDF Input
    │
    ▼
┌─────────────────────────────────┐
│     unstructured.io Partition   │
│  Text │ Tables │ Images         │
└──────┬────────┬────────┬────────┘
       │        │        │
       ▼        ▼        ▼
   Chunker  Table→Text  GPT-4o
   (semantic) converter  Vision
       │        │        │
       └────────┴────────┘
                │
                ▼
        OpenAI Embeddings
        text-embedding-3-large
                │
                ▼
    ┌──────────────────────┐
    │       Pinecone       │
    │  Namespace: text     │
    │  Namespace: tables   │
    │  Namespace: images   │
    └──────────┬───────────┘
               │
               ▼
     LangChain Hybrid Retriever
     (Dense MMR + Sparse BM25)
               │
               ▼
     GPT-4o with Conversation
     Memory + Source Citations
               │
               ▼
       Streamlit Chat UI