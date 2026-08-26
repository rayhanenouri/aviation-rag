from typing import Dict, List, Any
from langchain_groq import ChatGroq
from src.vector_store import AviationVectorStore
from dotenv import load_dotenv
import os

load_dotenv()


class AviationRAG:
    """RAG pipeline for aviation maintenance documentation queries."""

    def __init__(self, vector_store: AviationVectorStore):
        self.vector_store = vector_store

        # validate groq api key exists before attempting connection
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError(
                "GROQ_API_KEY not found in environment variables. "
                "Add GROQ_API_KEY to your .env file to enable LLM inference."
            )

        # groq api with gpt-oss-120b for reliable inference
        self.llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0.1,
            max_tokens=512,
            api_key=groq_api_key
        )

    def query(self, question: str, document_type_filter: str = None) -> Dict[str, Any]:
        """Execute RAG query and return answer with source citations and similarity scores."""
        import re
        from langchain_core.messages import SystemMessage, HumanMessage

        # retrieve relevant chunks with similarity scores
        docs_with_scores = self.vector_store.similarity_search_with_score(
            query=question,
            k=5,
            filter={"document_type": document_type_filter} if document_type_filter else None
        )

        # build context from retrieved chunks
        context_parts = []
        for doc, score in docs_with_scores:
            context_parts.append(doc.page_content)
        context = "\n\n".join(context_parts)

        # call llm directly with aviation expert prompt
        messages = [
            SystemMessage(content="""You are an aviation maintenance expert.
Answer questions using ONLY the provided context from official aviation documents.
If the answer is not in the context say: I don't have that information in the provided documents.
Always cite the source document and page number.

IMPORTANT: Provide ONLY the final answer. Do NOT show your thinking process or reasoning steps."""),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}\n\nProvide a direct answer based on the context above:")
        ]

        response = self.llm.invoke(messages)
        answer = response.content if hasattr(response, 'content') else str(response)

        # strip think tags if present - remove everything from <think> to </think> or end
        answer = re.sub(r'<think>.*', '', answer, flags=re.DOTALL).strip()

        # if answer is now empty or very short, return a default message
        if len(answer) < 10:
            answer = "I don't have that information in the provided documents."

        # format source documents with scores for traceability and visualization
        sources = []
        for doc, score in docs_with_scores:
            # chromadb uses cosine distance (0-2 range)
            # convert to similarity percentage: 0 distance = 100%, 2 distance = 0%
            similarity_pct = max(0, min(100, (1 - score / 2) * 100))

            sources.append({
                "file": doc.metadata["source_file"],
                "page": doc.metadata["page_number"],
                "document_type": doc.metadata["document_type"],
                "content": doc.page_content[:300],
                "similarity_score": similarity_pct
            })

        return {
            "answer": answer,
            "sources": sources,
            "question": question
        }


if __name__ == "__main__":
    # test pipeline with loaded vector store
    print("Loading vector store...")
    vs = AviationVectorStore()
    vs.load_existing()

    print("Initializing RAG pipeline...")
    rag = AviationRAG(vs)

    # test queries
    test_questions = [
        "What are the compliance requirements for A320 airworthiness directives?",
        "Explain the NASA C-MAPS turbofan engine degradation dataset"
    ]

    for question in test_questions:
        print(f"\n{'='*80}")
        print(f"Question: {question}")
        print(f"{'='*80}")

        result = rag.query(question)

        print(f"\nAnswer:\n{result['answer']}")

        print(f"\nSources:")
        for i, source in enumerate(result['sources'], 1):
            print(f"  [{i}] {source['file']}, page {source['page']}")
            print(f"      {source['content'][:150]}...")
