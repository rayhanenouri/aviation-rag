from pathlib import Path
from typing import List, Optional, Dict, Any
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from chromadb.config import Settings


class AviationVectorStore:
    """Manages embeddings and vector storage for aviation maintenance documents."""

    def __init__(self, persist_dir: str = "data/chroma_db"):
        self.persist_dir = Path(persist_dir)
        self.collection_name = "aviation_maintenance_docs"

        # all-MiniLM-L6-v2: fast CPU inference, good semantic understanding
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        self.vector_store = None

    def create_vector_store(self, documents: List[Document]) -> Chroma:
        """Embed documents and store in ChromaDB with persistence."""
        if not documents:
            raise ValueError("cannot create vector store from empty document list")

        # ensure persistence directory exists
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        print(f"Embedding {len(documents)} chunks...")
        print("This takes ~30 seconds on first run, then cached to disk")

        # chromadb handles batching internally for large document sets
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=str(self.persist_dir),
            collection_name=self.collection_name
        )

        print(f"Vector store created: {self.persist_dir}")
        return self.vector_store

    def load_existing(self) -> Chroma:
        """Load pre-built vector store from disk for fast startup."""
        if not self.persist_dir.exists():
            raise FileNotFoundError(
                f"no vector store at {self.persist_dir}. run create_vector_store first"
            )

        # chromadb settings for faster initialization
        chroma_settings = Settings(
            anonymized_telemetry=False,
            allow_reset=False
        )

        self.vector_store = Chroma(
            persist_directory=str(self.persist_dir),
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
            client_settings=chroma_settings
        )

        return self.vector_store

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Search for semantically similar chunks with optional metadata filtering."""
        if self.vector_store is None:
            raise RuntimeError("vector store not initialized. call load_existing or create_vector_store")

        # filter example: {"document_type": "airworthiness_directive"}
        return self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[tuple[Document, float]]:
        """search with similarity scores for visualization purposes"""
        if self.vector_store is None:
            raise RuntimeError("vector store not initialized")

        return self.vector_store.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter
        )

    def get_retriever(self, k: int = 5):
        """Return LangChain retriever interface for RAG chain integration."""
        if self.vector_store is None:
            raise RuntimeError("vector store not initialized")

        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )


if __name__ == "__main__":
    from src.document_loader import AviationDocumentLoader

    # test: load docs and create vector store
    loader = AviationDocumentLoader()
    documents = loader.load_all()

    vector_store = AviationVectorStore()
    vector_store.create_vector_store(documents)

    # test semantic search
    print("\nTesting semantic search:")
    query = "What are the compliance requirements for A320 hydraulic systems?"
    results = vector_store.similarity_search(query, k=3)

    for i, doc in enumerate(results, 1):
        print(f"\n[{i}] {doc.metadata['source_file']}, page {doc.metadata['page_number']}")
        print(f"    {doc.page_content[:150]}...")
