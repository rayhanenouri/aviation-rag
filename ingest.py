from pathlib import Path
from src.document_loader import AviationDocumentLoader
from src.vector_store import AviationVectorStore


def main():
    print("AVIATION DOCUMENT INGESTION")
    print()

    # load aviation maintenance documents
    print("Loading PDFs from docs/...")
    loader = AviationDocumentLoader(docs_dir="docs")
    documents = loader.load_all()
    print()

    # count chunks per document for ingestion report
    doc_chunks = {}
    for doc in documents:
        source = doc.metadata["source_file"]
        doc_chunks[source] = doc_chunks.get(source, 0) + 1

    print("INGESTION REPORT")
    print(f"Documents loaded:  {len(doc_chunks)}")
    for doc_name, chunk_count in sorted(doc_chunks.items()):
        print(f"  {doc_name}: {chunk_count} chunks")
    print(f"Total chunks:      {len(documents)}")
    print()

    # create chromadb vector store with embeddings
    print("Creating ChromaDB vector store...")
    vs = AviationVectorStore(persist_dir="data/chroma_db")
    vs.create_vector_store(documents)
    print()

    # verify indexing with sample retrieval test
    print("VERIFICATION TEST")
    print("Running sample retrieval query...")
    query = "What is the compliance deadline for EASA AD 2024-0038?"
    results = vs.similarity_search_with_score(query, k=3)

    print(f"Query: {query}")
    print(f"Retrieved {len(results)} chunks:")
    print()

    for i, (doc, score) in enumerate(results, 1):
        # chromadb uses cosine distance (0-2 range)
        similarity_pct = max(0, min(100, (1 - score / 2) * 100))
        print(f"[{i}] {doc.metadata['source_file']}, page {doc.metadata['page_number']}")
        print(f"    Similarity: {similarity_pct:.1f}%")
        print(f"    Type: {doc.metadata['document_type']}")
        print(f"    Content: {doc.page_content[:120]}...")
        print()

    print("Vector store ready for RAG queries")


if __name__ == "__main__":
    main()
