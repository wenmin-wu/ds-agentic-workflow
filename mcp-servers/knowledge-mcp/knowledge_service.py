import logging
import re
from typing import List, Dict, Any, Optional
from sqlalchemy import select, text, and_
from models import Document
from database import get_session
from embedding_service import embedding_service
from config import settings

logger = logging.getLogger(__name__)


def clean_markdown_for_embedding(title: str, markdown_content: str) -> str:
    """Clean markdown for better embedding generation"""
    # Combine title and content with special separator
    combined_text = f"Title: {title}\n\nContent: {markdown_content}"

    # Remove code blocks
    text = re.sub(r"```[\s\S]*?```", "", combined_text)
    # Remove inline code
    text = re.sub(r"`[^`]*`", "", text)
    # Convert headers to text (keep the content, remove #)
    text = re.sub(r"#{1,6}\s+", "", text)
    # Convert links to just the text
    text = re.sub(r"\[([^\]]*)\]\([^\)]*\)", r"\1", text)
    # Remove bold/italic markers
    text = re.sub(r"\*\*([^\*]*)\*\*", r"\1", text)
    text = re.sub(r"\*([^\*]*)\*", r"\1", text)
    # Clean up whitespace
    text = " ".join(text.split())
    return text


def store_knowledge(
    title: str, markdown_content: str, tags: List[str] = None, metadata: Dict[str, Any] = None
) -> int:
    """Store knowledge document and return document ID"""
    try:
        # Clean markdown for embedding generation
        cleaned_text = clean_markdown_for_embedding(title, markdown_content)

        # Generate embedding from cleaned markdown
        embedding = embedding_service.generate_embedding(cleaned_text)

        # Create document
        doc = Document(
            title=title,
            markdown=markdown_content,
            doc_metadata=metadata or {},
            embedding=embedding,
            tags=tags or [],
        )

        with get_session() as db:
            db.add(doc)
            db.flush()  # To get the ID
            doc_id = doc.id

        logger.info(f"Stored knowledge document: {title} (ID: {doc_id})")
        return doc_id

    except Exception as e:
        logger.error(f"Failed to store knowledge: {e}")
        raise


def search_knowledge(query: str, limit: int = None) -> List[Dict[str, Any]]:
    """Search knowledge using vector similarity search"""
    try:
        limit = limit or settings.max_results

        # Generate query embedding
        query_embedding = embedding_service.generate_embedding(query)

        with get_session() as db:
            # Pure vector similarity search
            results = db.execute(
                select(
                    Document,
                    (1 - Document.embedding.cosine_distance(query_embedding)).label("similarity"),
                )
                .filter(
                    and_(
                        Document.embedding.isnot(None),
                        (1 - Document.embedding.cosine_distance(query_embedding))
                        > settings.similarity_threshold,
                    )
                )
                .order_by((1 - Document.embedding.cosine_distance(query_embedding)).desc())
                .limit(limit)
            ).all()

            # Format results
            search_results = []
            for doc, similarity in results:
                result = doc.to_dict()
                result["similarity"] = float(similarity or 0)
                result["relevance_score"] = float(similarity or 0)
                search_results.append(result)

            logger.info(f"Found {len(search_results)} results for query: {query}")
            return search_results

    except Exception as e:
        logger.error(f"Failed to search knowledge: {e}")
        raise


def get_document(doc_id: int) -> Optional[Dict[str, Any]]:
    """Get document by ID"""
    try:
        with get_session() as db:
            doc = db.execute(select(Document).filter(Document.id == doc_id)).scalar_one_or_none()

            return doc.to_dict() if doc else None

    except Exception as e:
        logger.error(f"Failed to get document {doc_id}: {e}")
        raise


def list_documents(limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    """List documents with pagination"""
    try:
        with get_session() as db:
            docs = (
                db.execute(
                    select(Document)
                    .order_by(Document.updated_at.desc())
                    .limit(limit)
                    .offset(offset)
                )
                .scalars()
                .all()
            )

            return [doc.to_dict() for doc in docs]

    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise


def delete_document(doc_id: int) -> bool:
    """Delete document by ID"""
    try:
        with get_session() as db:
            result = db.execute(
                select(Document).filter(Document.id == doc_id)
            ).scalar_one_or_none()

            if result:
                db.delete(result)
                logger.info(f"Deleted document {doc_id}")
                return True
            else:
                logger.warning(f"Document {doc_id} not found")
                return False

    except Exception as e:
        logger.error(f"Failed to delete document {doc_id}: {e}")
        raise


def update_document(
    doc_id: int,
    title: str = None,
    markdown_content: str = None,
    tags: List[str] = None,
    metadata: Dict[str, Any] = None,
) -> bool:
    """Update document"""
    try:
        with get_session() as db:
            doc = db.execute(select(Document).filter(Document.id == doc_id)).scalar_one_or_none()

            if not doc:
                return False

            # Update fields
            if title is not None:
                doc.title = title

            if markdown_content is not None:
                doc.markdown = markdown_content

            # Regenerate embedding if either title or content changed
            if title is not None or markdown_content is not None:
                # Use current values for unchanged fields
                current_title = title if title is not None else doc.title
                current_markdown = (
                    markdown_content if markdown_content is not None else doc.markdown
                )
                # Regenerate embedding from cleaned markdown with title
                cleaned_text = clean_markdown_for_embedding(current_title, current_markdown)
                doc.embedding = embedding_service.generate_embedding(cleaned_text)

            if tags is not None:
                doc.tags = tags

            if metadata is not None:
                doc.doc_metadata = metadata

            logger.info(f"Updated document {doc_id}")
            return True

    except Exception as e:
        logger.error(f"Failed to update document {doc_id}: {e}")
        raise


def reset_knowledge() -> None:
    """Clear all documents (useful for testing)"""
    try:
        with get_session() as db:
            db.execute(text("TRUNCATE TABLE documents RESTART IDENTITY CASCADE"))
            logger.info("Reset knowledge database")

    except Exception as e:
        logger.error(f"Failed to reset knowledge: {e}")
        raise
