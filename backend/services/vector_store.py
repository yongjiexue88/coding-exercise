"""Neon PostgreSQL vector store service using SQLModel and pgvector."""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import List
from uuid import UUID, uuid4

from sqlalchemy import text
from sqlmodel import Session, select

from database import engine
from models_ingest import CorpusState, IngestChunk, IngestChunkEmbedding, IngestDocument, IngestRun


class VectorStoreService:
    """Service for managing document embeddings in Neon PostgreSQL via SQLModel."""

    def __init__(self):
        self._ensure_extensions()

    def _ensure_extensions(self):
        """Ensure pgvector extension exists."""
        with Session(engine) as session:
            session.exec(text("CREATE EXTENSION IF NOT EXISTS vector"))
            session.commit()

    def _get_or_create_active_run_id(self, session: Session, corpus_name: str) -> UUID:
        """Return active run id for corpus, creating one if missing."""
        state = session.get(CorpusState, corpus_name)
        if state:
            existing_run = session.get(IngestRun, state.active_run_id)
            if existing_run:
                return state.active_run_id

        run = IngestRun(
            id=uuid4(),
            corpus_name=corpus_name,
            status="ready",
            created_at=datetime.utcnow(),
        )
        session.add(run)
        session.flush()

        if state is None:
            state = CorpusState(corpus_name=corpus_name, active_run_id=run.id)
        else:
            state.active_run_id = run.id
        session.add(state)
        return run.id

    def reset(self, corpus_name: str = "default") -> None:
        """Compatibility reset used by legacy ingestion scripts.

        Instead of mutating rows in-place, we create a fresh empty run and switch
        `corpus_state` to it. Existing historical runs remain available.
        """
        with Session(engine) as session:
            state = session.get(CorpusState, corpus_name)
            if state:
                previous_run = session.get(IngestRun, state.active_run_id)
                if previous_run:
                    previous_run.status = "archived"
                    session.add(previous_run)

            run = IngestRun(
                id=uuid4(),
                corpus_name=corpus_name,
                status="ready",
                created_at=datetime.utcnow(),
            )
            session.add(run)

            if state is None:
                state = CorpusState(corpus_name=corpus_name, active_run_id=run.id)
            else:
                state.active_run_id = run.id
            session.add(state)
            session.commit()

    def add_documents(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[dict] | None = None,
        corpus_name: str = "default",
    ) -> None:
        """Compatibility ingestion API used by legacy scripts.

        Inserts documents/chunks into the active run of the target corpus.
        """
        if not (len(ids) == len(embeddings) == len(documents)):
            raise ValueError("ids, embeddings, and documents must have identical lengths.")

        if metadatas is None:
            metadatas = [{} for _ in documents]
        if len(metadatas) != len(documents):
            raise ValueError("metadatas length must match documents length.")

        if not documents:
            return

        with Session(engine) as session:
            run_id = self._get_or_create_active_run_id(session, corpus_name)
            docs_by_source: dict[str, IngestDocument] = {}

            for idx, (legacy_id, embedding, content, metadata) in enumerate(
                zip(ids, embeddings, documents, metadatas)
            ):
                meta = dict(metadata or {})
                source = str(meta.get("source") or meta.get("title") or "legacy-source")

                doc = docs_by_source.get(source)
                if doc is None:
                    doc = session.exec(
                        select(IngestDocument).where(
                            IngestDocument.run_id == run_id,
                            IngestDocument.source_id == source,
                        )
                    ).first()
                    if doc is None:
                        doc = IngestDocument(
                            id=uuid4(),
                            run_id=run_id,
                            source_id=source,
                            content_hash=hashlib.md5(
                                f"{source}:{run_id}".encode("utf-8")
                            ).hexdigest(),
                            metadata_json={
                                "source": source,
                                "ingested_via": "legacy_add_documents",
                            },
                            created_at=datetime.utcnow(),
                        )
                        session.add(doc)
                        session.flush()
                    docs_by_source[source] = doc

                chunk_meta = meta
                chunk_meta.setdefault("source", source)
                chunk_meta.setdefault("legacy_id", str(legacy_id))
                raw_chunk_index = chunk_meta.get("chunk_index", idx)
                try:
                    chunk_index = int(raw_chunk_index)
                except (TypeError, ValueError):
                    chunk_index = idx

                chunk = IngestChunk(
                    id=uuid4(),
                    document_id=doc.id,
                    content=content,
                    metadata_json=chunk_meta,
                    chunk_index=chunk_index,
                )
                session.add(chunk)
                session.flush()

                session.add(
                    IngestChunkEmbedding(
                        chunk_id=chunk.id,
                        embedding=embedding,
                    )
                )

            session.commit()

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 3,
        corpus_name: str = "default",
    ) -> dict:
        """Search for similar chunks in the active ingestion run."""
        with Session(engine) as session:
            # SQLModel doesn't support vector operators in pure python syntax,
            # so we use SQL text for cosine distance ordering.
            stmt = text(
                """
                SELECT
                    c.id,
                    c.chunk_index,
                    c.content,
                    d.source_id,
                    c.metadata_json,
                    e.embedding <=> :embedding AS distance
                FROM ingest_chunk_embedding e
                JOIN ingest_chunk c ON e.chunk_id = c.id
                JOIN ingest_document d ON c.document_id = d.id
                JOIN corpus_state cs ON d.run_id = cs.active_run_id
                WHERE cs.corpus_name = :corpus_name
                ORDER BY distance ASC
                LIMIT :top_k
                """
            )

            results = session.exec(
                stmt,
                params={
                    "embedding": str(query_embedding),
                    "corpus_name": corpus_name,
                    "top_k": top_k,
                },
            ).fetchall()

            if not results:
                return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

            current_docs = []
            current_metas = []
            current_dists = []

            for row in results:
                chunk_id, chunk_index, content, source, meta, dist = row
                current_docs.append(content)
                if meta is None:
                    meta = {}
                meta["source"] = source
                meta["chunk_id"] = str(chunk_id)
                meta["chunk_index"] = chunk_index
                current_metas.append(meta)
                current_dists.append(float(dist))

            return {
                "documents": [current_docs],
                "metadatas": [current_metas],
                "distances": [current_dists],
            }

    def get_document_count(self, corpus_name: str = "default") -> int:
        """Get the total number of chunks in the active run."""
        with Session(engine) as session:
            stmt = text(
                """
                SELECT COUNT(*)
                FROM ingest_chunk c
                JOIN ingest_document d ON c.document_id = d.id
                JOIN corpus_state cs ON d.run_id = cs.active_run_id
                WHERE cs.corpus_name = :corpus_name
                """
            )
            result = session.exec(stmt, params={"corpus_name": corpus_name}).first()
            if result is None:
                return 0
            if isinstance(result, (tuple, list)):
                return int(result[0])
            try:
                return int(result[0])
            except (TypeError, KeyError, IndexError):
                return int(result)

    def get_all_sources(self, corpus_name: str = "default") -> List[dict]:
        """Get a summary of all indexed sources in the active run."""
        with Session(engine) as session:
            stmt = text(
                """
                SELECT d.source_id, COUNT(*) as chunk_count
                FROM ingest_chunk c
                JOIN ingest_document d ON c.document_id = d.id
                JOIN corpus_state cs ON d.run_id = cs.active_run_id
                WHERE cs.corpus_name = :corpus_name
                GROUP BY d.source_id
                ORDER BY d.source_id
                """
            )
            results = session.exec(stmt, params={"corpus_name": corpus_name}).fetchall()

            return [{"source": row[0], "chunk_count": row[1]} for row in results]
