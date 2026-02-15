"""Unit tests for ingestion pipeline components."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.pipeline.chunker import SQuADChunker
from data.pipeline.parser import SQuADJsonParser

TEST_FILE = Path(__file__).parent / "data" / "test_squad.json"

def test_squad_parser():
    parser = SQuADJsonParser()
    sections = parser.parse(TEST_FILE)
    
    assert len(sections) == 1
    section = sections[0]
    
    assert section.title == "Super Mario"
    assert "platform game" in section.content
    assert len(section.questions) == 1
    assert section.questions[0]["question"] == "Who created Super Mario?"
    assert section.metadata["source"] == "test_squad.json"

def test_squad_chunker():
    parser = SQuADJsonParser()
    sections = parser.parse(TEST_FILE)
    
    chunker = SQuADChunker()
    chunks = chunker.chunk(sections)
    
    assert len(chunks) == 1
    chunk = chunks[0]
    
    # Check content format: Title + Content + Questions
    assert "Super Mario" in chunk["content"]
    assert "platform game" in chunk["content"]
    assert "Potential Questions:" in chunk["content"]
    assert "- Who created Super Mario?" in chunk["content"]
    
    # Check metadata preservation
    assert chunk["metadata"]["title"] == "Super Mario"
    assert chunk["metadata"]["questions"][0]["id"] == "q1"
