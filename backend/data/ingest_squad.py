
import json
import logging
from pathlib import Path
from typing import List

from sqlmodel import Session, select, delete
from database import engine, create_db_and_tables
from models_sql import Document, Paragraph, Question
from services.embedding import EmbeddingService
from services.vector_store import VectorStoreService

# Configuring logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_FILE = Path(__file__).parent / "documents" / "SQuAD-v1.1.json"

def load_squad_data(file_path: Path) -> dict:
    """Load SQuAD JSON data."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def clear_data(session: Session):
    """Clear existing data from Postgres and Chroma."""
    logger.info("🗑️  Clearing existing data...")
    # Clear Postgres
    session.exec(delete(Question))
    session.exec(delete(Paragraph))
    session.exec(delete(Document))
    session.commit()
    
    # Clear Neon Vectors
    vs = VectorStoreService()
    vs.reset()
    logger.info("   Data cleared.")

def ingest_squad():
    """Main ingestion function."""
    logger.info("🚀 Starting SQuAD Ingestion...")
    
    # 1. Init DB
    create_db_and_tables()
    
    # 2. Load JSON
    if not DATA_FILE.exists():
        logger.error(f"❌ File not found: {DATA_FILE}")
        return
        
    data = load_squad_data(DATA_FILE)
    squad_docs = data.get("data", [])
    logger.info(f"   Found {len(squad_docs)} documents in SQuAD file.")
    
    embedding_service = EmbeddingService()
    vector_store = VectorStoreService()
    
    with Session(engine) as session:
        # Clear old data
        clear_data(session)
        
        total_paragraphs = 0
        total_questions = 0
        chunks_to_embed = []
        ids_to_embed = []
        metadatas_to_embed = []

        # 3. Process each document
        for i, doc_data in enumerate(squad_docs):
            title = doc_data.get("title", "Untitled")
            logger.info(f"   Processing Doc {i+1}/{len(squad_docs)}: {title}")
            
            # Create Document in Postgres
            db_doc = Document(title=title, full_json_blob=doc_data)
            session.add(db_doc)
            session.flush() # flush to get ID
            
            for p_data in doc_data.get("paragraphs", []):
                context = p_data.get("context", "")
                
                # Structure-Aware Chunking: Use the paragraph as is
                db_paragraph = Paragraph(
                    document_id=db_doc.id,
                    context_text=context
                )
                session.add(db_paragraph)
                session.flush() # flush to get ID
                
                total_paragraphs += 1
                
                # Collect Questions
                for q_data in p_data.get("qas", []):
                    db_question = Question(
                        paragraph_id=db_paragraph.id,
                        question_text=q_data.get("question"),
                        answers=q_data.get("answers", []),
                        is_impossible=q_data.get("is_impossible", False)
                    )
                    session.add(db_question)
                    total_questions += 1
                
                # Prepare for Vector Store
                # We embed the context. We associate it with the Paragraph ID.
                chunks_to_embed.append(context)
                ids_to_embed.append(str(db_paragraph.id))
                metadatas_to_embed.append({
                    "paragraph_id": str(db_paragraph.id),
                    "document_id": str(db_doc.id),
                    "title": title
                })
        
        # Commit to Postgres
        session.commit()
        logger.info(f"💾 Saved to Postgres: {len(squad_docs)} Docs, {total_paragraphs} Paragraphs, {total_questions} Questions.")

        # 4. Generate Embeddings & Upsert to Neon Vector Store
        logger.info("brain Generating embeddings (this may take a while)...")
        # In a real production app, we'd batch this. For now, assuming it fits in memory or `embed_batch` handles batching.
        # Actually, let's process in batches to be safe.
        BATCH_SIZE = 100
        total_vectors = 0
        
        for i in range(0, len(chunks_to_embed), BATCH_SIZE):
            batch_texts = chunks_to_embed[i : i + BATCH_SIZE]
            batch_ids = ids_to_embed[i : i + BATCH_SIZE]
            batch_meta = metadatas_to_embed[i : i + BATCH_SIZE]
            
            embeddings = embedding_service.embed_batch(batch_texts)
            vector_store.add_documents(
                ids=batch_ids,
                documents=batch_texts,
                embeddings=embeddings,
                metadatas=batch_meta
            )
            total_vectors += len(batch_ids)
            if i % 500 == 0:
                logger.info(f"   Indexed {total_vectors}/{len(chunks_to_embed)} chunks...")
                
        logger.info(f"✅ Vectors Indexed: {total_vectors}")

if __name__ == "__main__":
    ingest_squad()
