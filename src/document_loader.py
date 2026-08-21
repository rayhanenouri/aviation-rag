from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class AviationDocumentLoader:
    """Loads and chunks aviation maintenance PDFs for RAG retrieval."""

    def __init__(self, docs_dir: str = "docs"):
        self.docs_dir = Path(docs_dir)
        # 512 tokens ~= 400 words, preserves complete procedure steps
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=128,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def _determine_document_type(self, filename: str) -> str:
        """Map filename to aviation document category."""
        filename_lower = filename.lower()
        if "easa" in filename_lower or "_ad_" in filename_lower:
            return "airworthiness_directive"
        elif "nasa" in filename_lower or "technical" in filename_lower:
            return "technical_report"
        else:
            return "maintenance_manual"

    def load_pdf(self, pdf_path: Path) -> List[Document]:
        """Load single PDF with metadata for traceability."""
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

        document_type = self._determine_document_type(pdf_path.name)
        chunks = []

        for page in pages:
            # page.metadata already contains page number from PyPDFLoader
            page_number = page.metadata.get("page", 0) + 1

            # chunk each page to preserve procedure step boundaries
            page_chunks = self.text_splitter.split_text(page.page_content)

            for idx, chunk_text in enumerate(page_chunks):
                chunk = Document(
                    page_content=chunk_text,
                    metadata={
                        "source_file": pdf_path.name,
                        "page_number": page_number,
                        "document_type": document_type,
                        "chunk_index": len(chunks) + idx
                    }
                )
                chunks.append(chunk)

        return chunks

    def load_all(self) -> List[Document]:
        """Load all PDFs from docs directory."""
        if not self.docs_dir.exists():
            raise FileNotFoundError(f"docs directory not found: {self.docs_dir}")

        pdf_files = list(self.docs_dir.glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(f"no PDFs found in {self.docs_dir}")

        all_chunks = []
        for pdf_path in sorted(pdf_files):
            print(f"Loading {pdf_path.name}...")
            chunks = self.load_pdf(pdf_path)
            all_chunks.extend(chunks)
            print(f"  {len(chunks)} chunks created")

        print(f"\nTotal: {len(all_chunks)} chunks from {len(pdf_files)} documents")
        return all_chunks


if __name__ == "__main__":
    # test loader on aviation docs
    loader = AviationDocumentLoader()
    documents = loader.load_all()

    print("\nSample chunk:")
    sample = documents[0]
    print(f"Content: {sample.page_content[:200]}...")
    print(f"Metadata: {sample.metadata}")
