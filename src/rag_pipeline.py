from typing import Dict, List, Any
from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate
from vector_store import AviationVectorStore


class AviationRAG:
    """RAG pipeline for aviation maintenance documentation queries."""

    def __init__(self, vector_store: AviationVectorStore):
        self.vector_store = vector_store

        # llama3.2:3b runs locally, no API costs
        self.llm = OllamaLLM(
            model="llama3.2:3b",
            temperature=0.1  # low temperature for factual accuracy
        )

        # aviation-specific system prompt enforces grounding and citations
        self.prompt_template = PromptTemplate(
            template="""You are an aviation maintenance documentation assistant for MRO engineers.

Answer questions using ONLY the provided maintenance documentation context below.
If the answer is not in the context, respond: "I don't have that information in the provided documents."

Always cite sources with document name and page number in your answer.

Use proper aviation terminology:
- Airworthiness Directives (ADs) are mandatory compliance actions
- Refer to aircraft by type (A320, A330) and systems (hydraulic, fuel, engine)
- Compliance deadlines are critical — state them exactly as written
- Maintenance procedures must be followed precisely — do not paraphrase

Maintain technical accuracy. MRO engineers will verify your answers against the original documentation for regulatory compliance.

Context:
{context}

Question: {question}

Answer:""",
            input_variables=["context", "question"]
        )

        # retrieval chain with return_source_documents for traceability
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",  # stuff all context into single prompt
            retriever=self.vector_store.get_retriever(k=5),
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt_template}
        )

    def query(self, question: str, document_type_filter: str = None) -> Dict[str, Any]:
        """Execute RAG query and return answer with source citations."""
        # optional metadata filtering for specific document types
        if document_type_filter:
            self.qa_chain.retriever.search_kwargs["filter"] = {
                "document_type": document_type_filter
            }

        result = self.qa_chain.invoke({"query": question})

        # format source documents for traceability
        sources = []
        for doc in result["source_documents"]:
            sources.append({
                "file": doc.metadata["source_file"],
                "page": doc.metadata["page_number"],
                "document_type": doc.metadata["document_type"],
                "content": doc.page_content[:300]  # preview for verification
            })

        return {
            "answer": result["result"],
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
