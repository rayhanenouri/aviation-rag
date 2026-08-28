from src.document_loader import AviationDocumentLoader
from src.vector_store import AviationVectorStore


print("CHROMADB PERSISTENCE TEST")
print()

# load documents
print("Loading documents...")
loader = AviationDocumentLoader(docs_dir="docs")
documents = loader.load_all()
print()

# create vector store
print("Creating vector store...")
vs = AviationVectorStore(persist_dir="data/chroma_db")
vs.create_vector_store(documents)
print()

# immediately load to test persistence
print("Testing persistence - loading immediately...")
vs2 = AviationVectorStore(persist_dir="data/chroma_db")
vs2.load_existing()
print()

# verify chunk count matches
created_count = vs.vector_store._collection.count()
loaded_count = vs2.vector_store._collection.count()

print("PERSISTENCE VERIFICATION")
print(f"Created:  {created_count} chunks")
print(f"Loaded:   {loaded_count} chunks")
print(f"Match:    {'YES' if created_count == loaded_count else 'NO'}")
print()

if created_count == loaded_count and loaded_count > 0:
    print("Persistence working correctly")
else:
    print("PERSISTENCE FAILED")
    print("ChromaDB may not be persisting data to disk")
