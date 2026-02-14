"""Chunking logic for the ingestion pipeline."""

from typing import List, Dict, Any
from .parser import ParsedSection

class SQuADChunker:
    """Structure-aware chunker for SQuAD data."""

    def chunk(self, sections: List[ParsedSection]) -> List[Dict[str, Any]]:
        """Transform parsed sections into chunk payloads."""
        chunks = []
        
        for section in sections:
            # Format questions for embedding context
            # We only use the question text, not the answers, to simulate user query matching.
            questions_text = "\n".join([f"- {q['question']}" for q in section.questions])
            
            # Construct the rich content for embedding
            # Format: Title + Context + Questions
            full_content = f"{section.title}\n\n{section.content}"
            if questions_text:
                full_content += f"\n\nPotential Questions:\n{questions_text}"
            
            # Metadata includes the original raw Q&A for high-fidelity retrieval display
            metadata = section.metadata.copy()
            metadata["title"] = section.title
            metadata["questions"] = section.questions # Store full Q&A structure
            
            chunks.append({
                "content": full_content,
                "metadata": metadata,
                # We can add a hash here if we want chunk-level dedup later
            })
            
        return chunks
