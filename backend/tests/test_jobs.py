"""Unit tests for JobManager."""

import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.pipeline.jobs import JobManager, IngestJob


@patch("data.pipeline.jobs.Session")
def test_create_job(mock_session_cls):
    mock_session = mock_session_cls.return_value.__enter__.return_value
    
    # Mock refresh to fill ID
    def mock_refresh(obj):
        if not obj.id:
            obj.id = uuid.uuid4()
    mock_session.refresh.side_effect = mock_refresh

    job_id = JobManager.create_job({"foo": "bar"})
    
    assert isinstance(job_id, uuid.UUID)
    mock_session.add.assert_called_once()
    mock_session.commit.assert_called_once()
    
    args, _ = mock_session.add.call_args
    job = args[0]
    assert isinstance(job, IngestJob)
    assert job.payload == {"foo": "bar"}
    assert job.status == "pending"


@patch("data.pipeline.jobs.Session")
def test_cancel_job_success(mock_session_cls):
    mock_session = mock_session_cls.return_value.__enter__.return_value
    
    # Setup mock job
    job = IngestJob(id=uuid.uuid4(), status="pending")
    mock_session.get.return_value = job
    
    result = JobManager.cancel_job(job.id)
    
    assert result is True
    assert job.status == "cancelled"
    mock_session.commit.assert_called_once()


@patch("data.pipeline.jobs.Session")
def test_cancel_job_failure(mock_session_cls):
    mock_session = mock_session_cls.return_value.__enter__.return_value
    
    # Setup mock job that is already completed
    job = IngestJob(id=uuid.uuid4(), status="completed")
    mock_session.get.return_value = job
    
    result = JobManager.cancel_job(job.id)
    
    assert result is False
    assert job.status == "completed"
    mock_session.commit.assert_not_called()
