from sqlalchemy import Column, BigInteger, Text, DateTime, func, ARRAY
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.ext.declarative import declarative_base
from pgvector.sqlalchemy import Vector
from typing import Dict, Any

Base = declarative_base()


class Document(Base):
    __tablename__ = "documents"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    title = Column(Text, nullable=False)
    markdown = Column(Text, nullable=False)
    doc_metadata = Column(JSON, default={})
    embedding = Column(Vector(1024))  # bge-m3:567m dimensions
    tags = Column(ARRAY(Text), default=[])
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<Document(id={self.id}, title='{self.title}')>"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "markdown": self.markdown,
            "metadata": self.doc_metadata,
            "tags": self.tags,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
 