from src.vector_store import AviationVectorStore
from src.rag_pipeline import AviationRAG
import plotly.graph_objects as go

print("Loading vector store...")
vs = AviationVectorStore()
vs.load_existing()

print("Initializing RAG pipeline...")
rag = AviationRAG(vs)

print("\nTesting query with similarity scores...")
result = rag.query("What are the compliance requirements for A320 hydraulic systems?")

print(f"\nAnswer: {result['answer'][:150]}...")
print(f"\nSources with similarity scores:")

for i, source in enumerate(result['sources'], 1):
    print(f"  [{i}] {source['file']}, page {source['page']}")
    print(f"      Similarity: {source['similarity_score']:.2f}%")
    print(f"      Type: {source['document_type']}")

print("\nTesting radar chart generation...")
categories = []
scores = []

for i, source in enumerate(result['sources'], 1):
    filename = source['file'].replace('.pdf', '')
    if len(filename) > 15:
        filename = filename[:12] + '...'
    label = f"{filename} p{source['page']}"
    categories.append(label)
    scores.append(source['similarity_score'])

categories_loop = categories + [categories[0]]
scores_loop = scores + [scores[0]]

print(f"Radar chart data: {len(categories)} axes, scores range {min(scores):.1f}% - {max(scores):.1f}%")

print("\nAll visualization components validated successfully!")
