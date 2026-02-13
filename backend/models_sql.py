from typing import List, Optional
from uuid import UUID, uuid4
from sqlmodel import Field, SQLModel, Relationship, JSON

class Document(SQLModel, table=True):
    id: Optional[UUID] = Field(default_factory=uuid4, primary_key=True)
    title: str = Field(index=True)
    full_json_blob: Optional[dict] = Field(default=None, sa_type=JSON)
    
    paragraphs: List["Paragraph"] = Relationship(back_populates="document")

class Paragraph(SQLModel, table=True):
    id: Optional[UUID] = Field(default_factory=uuid4, primary_key=True)
    document_id: UUID = Field(foreign_key="document.id")
    context_text: str
    
    document: Document = Relationship(back_populates="paragraphs")
    questions: List["Question"] = Relationship(back_populates="paragraph")

class Question(SQLModel, table=True):
    id: Optional[UUID] = Field(default_factory=uuid4, primary_key=True)
    paragraph_id: UUID = Field(foreign_key="paragraph.id")
    question_text: str
    answers: list[dict] = Field(default=[], sa_type=JSON)
    is_impossible: bool = Field(default=False)
    
    paragraph: Paragraph = Relationship(back_populates="questions")
