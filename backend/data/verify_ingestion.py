
import logging
from sqlmodel import Session, select, func
from database import engine
from models_sql import Document, Paragraph, Question
from services.vector_store import VectorStoreService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify():
    logger.info("🔍 Verifying SQuAD Ingestion...")
    
    with Session(engine) as session:
        # Postgres Counts
        doc_count = session.exec(select(func.count(Document.id))).one()
        para_count = session.exec(select(func.count(Paragraph.id))).one()
        q_count = session.exec(select(func.count(Question.id))).one()
        
        logger.info(f"📊 Postgres Stats:")
        logger.info(f"   Documents: {doc_count}")
        logger.info(f"   Paragraphs: {para_count}")
        logger.info(f"   Questions: {q_count}")
        
        if doc_count == 0:
            logger.error("❌ No documents found in Postgres!")
            return

        # Sample Query
        stmt = select(Question).limit(1)
        q = session.exec(stmt).first()
        if q:
            logger.info(f"   Sample Question: {q.question_text}")
            logger.info(f"   Linked Paragraph ID: {q.paragraph_id}")
            
            # Fetch Context
            para = session.get(Paragraph, q.paragraph_id)
            if para:
                logger.info(f"   Context Preview: {para.context_text[:100]}...")
            else:
                logger.error("❌ Linked Paragraph not found!")

    # Neon Vector Counts
    vs = VectorStoreService()
    vector_count = vs.get_document_count()
    logger.info(f"📊 Neon Vector Stats:")
    logger.info(f"   Vectors: {vector_count}")
    
    if vector_count == 0:
        logger.error("❌ No vectors found in Neon!")
    else:
        logger.info("✅ Verification Complete!")

if __name__ == "__main__":
    verify()
