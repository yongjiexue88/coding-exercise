"""Document ingestion pipeline — loads, chunks, embeds, and stores documents."""

import os
import hashlib
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from services.embedding import EmbeddingService
from services.vector_store import VectorStoreService
from config import settings


DOCUMENTS_DIR = Path(__file__).parent / "documents"


def load_documents(directory: Path = DOCUMENTS_DIR) -> list[dict]:
    """Load all markdown documents from the directory."""
    documents = []
    for filepath in sorted(directory.glob("*.md")):
        content = filepath.read_text(encoding="utf-8")
        documents.append({
            "content": content,
            "source": filepath.name,
            "path": str(filepath),
        })
    return documents


def chunk_documents(
    documents: list[dict],
    chunk_size: int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap,
) -> list[dict]:
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, split in enumerate(splits):
            chunk_id = hashlib.md5(
                f"{doc['source']}_{i}".encode()
            ).hexdigest()
            chunks.append({
                "id": chunk_id,
                "content": split,
                "metadata": {
                    "source": doc["source"],
                    "chunk_index": i,
                    "total_chunks": len(splits),
                },
            })
    return chunks


def ingest(reset: bool = True) -> dict:
    """Run the full ingestion pipeline.

    Args:
        reset: If True, clear existing data before ingesting.

    Returns:
        Dict with ingestion statistics.
    """
    print("📄 Loading documents...")
    documents = load_documents()
    print(f"   Found {len(documents)} documents")

    print("✂️  Chunking documents...")
    chunks = chunk_documents(documents)
    print(f"   Created {len(chunks)} chunks")

    print("🧠 Generating embeddings...")
    embedding_service = EmbeddingService()
    texts = [c["content"] for c in chunks]
    embeddings = embedding_service.embed_batch(texts)
    print(f"   Generated {len(embeddings)} embeddings (dim={embedding_service.dimension})")

    print("💾 Storing in ChromaDB...")
    vector_store = VectorStoreService()
    if reset:
        vector_store.reset()

    vector_store.add_documents(
        ids=[c["id"] for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[c["metadata"] for c in chunks],
    )

    total = vector_store.get_document_count()
    print(f"✅ Done! {total} chunks indexed in ChromaDB")

    return {
        "documents_processed": len(documents),
        "chunks_created": len(chunks),
        "total_indexed": total,
    }


if __name__ == "__main__":
    ingest()
