"""Data parsers for the ingestion pipeline."""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class ParsedSection:
    """Transient object representing a logical section of a document."""
    title: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    questions: List[Dict[str, Any]] = field(default_factory=list)


class BaseParser:
    """Abstract base parser."""
    def parse(self, file_path: Path) -> List[ParsedSection]:
        raise NotImplementedError


class SQuADJsonParser(BaseParser):
    """Parser for SQuAD v1.1 style JSON."""

    def parse(self, file_path: Path) -> List[ParsedSection]:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        sections = []
        
        # SQuAD structure: data -> [ { title, paragraphs: [ { context, qas: [] } ] } ]
        for doc_idx, doc in enumerate(data.get("data", [])):
            title = doc.get("title", f"Document {doc_idx}")
            
            for p_idx, paragraph in enumerate(doc.get("paragraphs", [])):
                context = paragraph.get("context", "")
                qas = paragraph.get("qas", [])
                
                # We treat each Paragraph as a Section
                sections.append(ParsedSection(
                    title=title,
                    content=context,
                    metadata={
                        "source": file_path.name,
                        "doc_index": doc_idx,
                        "paragraph_index": p_idx
                    },
                    questions=qas
                ))
                
        return sections
