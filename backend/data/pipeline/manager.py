"""Pipeline manager for orchestrating data ingestion."""

import logging
import hashlib
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable
from uuid import UUID

from sqlmodel import Session
from database import engine
from models_ingest import (
    IngestRun,
    IngestDocument,
    IngestChunk,
    IngestChunkEmbedding,
    CorpusState,
    IngestJob,
)
from services.embedding import EmbeddingService
from .parser import SQuADJsonParser
from .chunker import SQuADChunker

logger = logging.getLogger(__name__)

# Directory containing source documents
DATA_DIR = Path(__file__).parent.parent / "documents"


class IngestionCancelledError(RuntimeError):
    """Raised when a user cancels an in-flight ingestion job."""


class IngestionPipeline:
    """Orchestrates the full ingestion process: Parse -> Chunk -> Embed -> Commit."""

    def __init__(self, job_id: UUID, heartbeat_callback: Optional[Callable] = None):
        self.job_id = job_id
        self.heartbeat = heartbeat_callback or (lambda: None)
        self.parser = SQuADJsonParser()
        self.chunker = SQuADChunker()
        self.embedding_service = EmbeddingService()

    def _is_job_cancelled(self) -> bool:
        """Check whether the ingestion job has been cancelled by the user."""
        with Session(engine) as session:
            job = session.get(IngestJob, self.job_id)
            return bool(job and job.status == "cancelled")

    def _checkpoint(self) -> None:
        """Heartbeat + cooperative cancellation checkpoint."""
        self.heartbeat()
        if self._is_job_cancelled():
            raise IngestionCancelledError(f"Ingestion job {self.job_id} cancelled by user.")

    def run(self):
        """Execute the ingestion pipeline."""
        self._checkpoint()
        with Session(engine) as session:
            # 1. Create a new IngestRun (Blue/Green Deployment)
            run = IngestRun(
                id=UUID(int=uuid.uuid4().int), # usage of uuid package
                corpus_name="default",
                status="indexing",
                created_at=datetime.utcnow()
            )
            session.add(run)
            session.commit()
            session.refresh(run)
            logger.info(f"🚀 Starting IngestRun {run.id}")

            try:
                # 2. Identify Files
                # For now, we look for *.json in the data/documents folder, specifically SQuAD
                # In a real app, 'payload' from job would specify files.
                files = list(DATA_DIR.glob("*.json"))
                if not files:
                    logger.warning(f"⚠️ No JSON files found in {DATA_DIR}")
                    raise RuntimeError(f"No JSON files found in {DATA_DIR}")

                for file_path in files:
                    self._checkpoint()
                    self._process_file(session, run.id, file_path)

                # 3. Atomic Switch (Make this run active)
                self._activate_run(session, run.id)

            except IngestionCancelledError:
                logger.info("🛑 Ingestion cancelled before activation.")
                run.status = "archived"
                session.add(run)
                session.commit()
                raise
            except Exception as e:
                logger.error(f"❌ Ingestion failed: {e}")
                run.status = "failed"
                session.add(run)
                session.commit()
                # Propagate to JobWorker to mark job as failed
                raise e

    def _process_file(self, session: Session, run_id: UUID, file_path: Path):
        """Process a single file."""
        logger.info(f"📄 Processing {file_path.name}...")
        self._checkpoint()

        # Parse
        sections = self.parser.parse(file_path)
        self._checkpoint()

        # Chunk
        chunks_data = self.chunker.chunk(sections)
        self._checkpoint()

        if not chunks_data:
            return

        # Create Document Record
        # Generate stable source_id and content_hash
        content_hash = hashlib.md5(file_path.read_bytes()).hexdigest()

        doc = IngestDocument(
            id=uuid.uuid4(),
            run_id=run_id,
            source_id=file_path.name,
            content_hash=content_hash,
            metadata_json={"path": str(file_path)},
            created_at=datetime.utcnow()
        )
        session.add(doc)
        session.commit()
        session.refresh(doc)
        
        # Batch Insert Chunks & Embeddings
        # To avoid massive memory usage, we batch this
        BATCH_SIZE = 50

        for i in range(0, len(chunks_data), BATCH_SIZE):
            self._checkpoint()
            batch = chunks_data[i : i + BATCH_SIZE]

            # Prepare texts for embedding
            texts = [c["content"] for c in batch]
            embeddings = self.embedding_service.embed_batch(texts)
            self._checkpoint()

            for j, item in enumerate(batch):
                # Create Chunk
                chunk = IngestChunk(
                    id=uuid.uuid4(),
                    document_id=doc.id,
                    content=item["content"],
                    metadata_json=item["metadata"],
                    chunk_index=i + j
                )
                session.add(chunk)
                session.flush() # Get ID
                
                # Create Embedding
                emb = IngestChunkEmbedding(
                    chunk_id=chunk.id,
                    embedding=embeddings[j]
                )
                session.add(emb)

            session.commit()
            self._checkpoint()
            logger.info(f"   Saved batch {i//BATCH_SIZE + 1}/{(len(chunks_data)//BATCH_SIZE) + 1}")

    def _activate_run(self, session: Session, run_id: UUID):
        """Perform the atomic switch to make this run active."""
        self._checkpoint()
        logger.info(f"🔄 Switching Corpus 'default' to Run {run_id}...")

        # 1. Update IngestRun status
        run = session.get(IngestRun, run_id)
        run.status = "ready"
        session.add(run)

        # 2. Update CorpusState (Upsert)
        state = session.get(CorpusState, "default")
        if not state:
            state = CorpusState(corpus_name="default", active_run_id=run_id)
        else:
            state.active_run_id = run_id

        session.add(state)
        session.commit()
        logger.info("✅ Switch Complete. New data is live.")
