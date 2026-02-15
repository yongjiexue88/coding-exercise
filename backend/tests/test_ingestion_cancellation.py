"""Tests for cooperative cancellation checkpoints in ingestion pipeline."""

import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.pipeline.manager import IngestionCancelledError, IngestionPipeline


def test_checkpoint_raises_when_job_is_cancelled(monkeypatch):
    pipeline = IngestionPipeline(job_id=uuid.uuid4(), heartbeat_callback=lambda: None)
    monkeypatch.setattr(pipeline, "_is_job_cancelled", lambda: True)

    with pytest.raises(IngestionCancelledError):
        pipeline._checkpoint()
