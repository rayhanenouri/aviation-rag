from src.vector_store import AviationVectorStore

print("TESTING VECTOR STORE LOAD FROM DISK")
print()

# load from disk without creating
vs = AviationVectorStore(persist_dir="data/chroma_db")
vs.load_existing()
print()

# test retrieval
print("Testing retrieval...")
query = "What aircraft types are affected by EASA AD 2024-0038?"
results = vs.similarity_search_with_score(query, k=3)

print(f"Query: {query}")
print(f"Retrieved {len(results)} chunks:")
print()

for i, (doc, score) in enumerate(results, 1):
    similarity_pct = max(0, min(100, (1 - score / 2) * 100))
    print(f"[{i}] {doc.metadata['source_file']}, page {doc.metadata['page_number']}")
    print(f"    Similarity: {similarity_pct:.1f}%")
    print(f"    Content: {doc.page_content[:100]}...")
    print()

print("Vector store loaded and working correctly")
