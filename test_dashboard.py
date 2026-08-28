from src.vector_store import AviationVectorStore
from src.rag_pipeline import AviationRAG

print("Loading vector store...")
vs = AviationVectorStore()
vs.load_existing()

print("Initializing RAG pipeline...")
rag = AviationRAG(vs)

print("\nGathering system status...")
chunk_count = vs.vector_store._collection.count()
all_data = vs.vector_store.get()
unique_files = set()
for metadata in all_data['metadatas']:
    unique_files.add(metadata['source_file'])

print(f"ChromaDB: Connected")
print(f"Documents: {len(unique_files)}")
print(f"Chunks: {chunk_count}")
print(f"Embedding Model: all-MiniLM-L6-v2")
print(f"LLM Model: llama3.2:3b")

print("\nTesting query...")
result = rag.query("What is the C-MAPS dataset?")
print(f"\nAnswer: {result['answer'][:200]}...")
print(f"\nSources found: {len(result['sources'])}")

print("\nDashboard components validated successfully!")
