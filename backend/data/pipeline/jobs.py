"""Durable background job worker for ingestion pipeline.

Implements:
- Job Creation (JobManager)
- Safe Concurrency (SKIP LOCKED)
- Heartbeats & Lease Recovery
- Error Handling
"""

import time
import logging
import traceback
import asyncio
from datetime import datetime, timedelta
from typing import Optional
import uuid
from uuid import UUID

from sqlmodel import Session, select
from models_ingest import IngestJob
from database import engine
from data.pipeline.manager import IngestionCancelledError

# Configure logging
logger = logging.getLogger(__name__)

LEASE_DURATION_SECONDS = 30
POLL_INTERVAL_SECONDS = 5


class JobManager:
    """API-facing manager for creating and tracking jobs."""

    @staticmethod
    def create_job(payload: dict) -> UUID:
        """Create a new ingestion job."""
        with Session(engine) as session:
            job = IngestJob(
                id=uuid.uuid4(),
                payload=payload, 
                status="pending",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            session.add(job)
            session.commit()
            session.refresh(job)
            return job.id

    @staticmethod
    def get_job(job_id: UUID) -> Optional[IngestJob]:
        """Get job status."""
        with Session(engine) as session:
            return session.get(IngestJob, job_id)

    @staticmethod
    def cancel_job(job_id: UUID) -> bool:
        """Cancel a job if not already completed."""
        with Session(engine) as session:
            job = session.get(IngestJob, job_id)
            if not job or job.status in ("completed", "failed"):
                return False
            
            job.status = "cancelled"
            job.updated_at = datetime.utcnow()
            session.add(job)
            session.commit()
            return True
            
    @staticmethod
    def retry_job(job_id: UUID) -> bool:
        """Retry a failed job."""
        with Session(engine) as session:
            job = session.get(IngestJob, job_id)
            if not job or job.status != "failed":
                return False
            
            job.status = "pending"
            job.error_details = None
            job.retry_count += 1
            job.updated_at = datetime.utcnow()
            session.add(job)
            session.commit()
            return True


class JobWorker:
    """Background worker that processes ingestion jobs."""

    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        self.running = False

    async def run_loop(self):
        """Main worker loop."""
        self.running = True
        logger.info(f"👷 JobWorker {self.worker_id} started.")
        
        while self.running:
            try:
                processed = self._process_next_job()
                if not processed:
                    await asyncio.sleep(POLL_INTERVAL_SECONDS)
            except Exception as e:
                logger.error(f"❌ Worker loop error: {e}")
                await asyncio.sleep(POLL_INTERVAL_SECONDS)

    def _process_next_job(self) -> bool:
        """Acquire and process a single job via SKIP LOCKED."""
        with Session(engine) as session:
            # Atomic Lease Acquisition
            now = datetime.utcnow()
            stmt = (
                select(IngestJob)
                .where(
                    (IngestJob.status == "pending") |
                    ((IngestJob.status == "processing") & (IngestJob.lease_expires_at < now))
                )
                .order_by(IngestJob.created_at)
                .limit(1)
                .with_for_update(skip_locked=True)
            )
            
            job = session.exec(stmt).first()
            
            if not job:
                return False

            # Mark as processing
            job.status = "processing"
            job.worker_id = self.worker_id
            job.lease_expires_at = datetime.utcnow() + timedelta(seconds=LEASE_DURATION_SECONDS)
            job.updated_at = datetime.utcnow()
            session.add(job)
            session.commit()
            
            logger.info(f"🔄 Processing Job {job.id} (try {job.retry_count})...")

        # Execute Job logic (outside the lock transaction)
        try:
            self._execute_job(job.id)
            self._complete_job(job.id, status="completed")
            return True
        except IngestionCancelledError as e:
            logger.info(f"🛑 Job {job.id} cancelled cooperatively: {e}")
            self._complete_job(job.id, status="cancelled")
            return True
        except Exception as e:
            logger.error(f"❌ Job {job.id} failed: {e}")
            self._complete_job(job.id, status="failed", error=str(e))
            return True

    def _execute_job(self, job_id: UUID):
        """Run the actual pipeline logic.
        
        Note: We define the import inside the method to avoid circular dependencies
        if pipeline modules allow importing models or other shared things.
        """
        # Dynamic import to break potential cyclic dependency on startup
        from data.pipeline.manager import IngestionPipeline

        # Helper to update heartbeat
        def heartbeat_callback():
            with Session(engine) as session:
                j = session.get(IngestJob, job_id)
                if j and j.status == "cancelled":
                    raise IngestionCancelledError(f"Ingestion job {job_id} cancelled by user.")
                if j and j.status == "processing":
                    j.lease_expires_at = datetime.utcnow() + timedelta(seconds=LEASE_DURATION_SECONDS)
                    session.add(j)
                    session.commit()

        # Run pipeline
        pipeline = IngestionPipeline(job_id=job_id, heartbeat_callback=heartbeat_callback)
        pipeline.run()

    def _complete_job(self, job_id: UUID, status: str, error: str = None):
        """Update job status on completion."""
        with Session(engine) as session:
            job = session.get(IngestJob, job_id)
            if job:
                # Preserve an explicit user cancellation even if the worker reaches completion.
                if not (job.status == "cancelled" and status in ("completed", "failed")):
                    job.status = status
                if error:
                    job.error_details = {"message": error, "traceback": traceback.format_exc()}
                job.updated_at = datetime.utcnow()
                session.add(job)
                session.commit()
            logger.info(f"✅ Job {job_id} finished: {status}")
