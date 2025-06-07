#!/usr/bin/env python3
"""
Knowledge MCP Server using FastMCP
AI-friendly API with smart document handling
"""

import logging
import sys
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from fastmcp import FastMCP

# Configure logging to stderr
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)

logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP(
    name="knowledge-mcp-server",
    description="AI-powered knowledge management system with semantic search, document storage, and intelligent retrieval capabilities",
    version="1.0.0",
    instructions="""
This server provides comprehensive knowledge management capabilities designed for AI agents and applications.

Key Features:
- Store and manage knowledge documents in markdown format
- Document versioning and metadata management  
- Semantic search powered by BGE-M3 embeddings
- PostgreSQL + pgvector backend for performance
- AI-friendly response formats with structured metadata

Quick Start for Knowledge Management:
1. Store documents: Use store_knowledge() with title, markdown content, tags, and metadata
2. Search content: Use search_knowledge() with natural language queries for semantic retrieval
3. Manage documents: Use get_document(), update_document(), list_documents() for CRUD operations
4. Track operations: All tools return structured results with document IDs and metadata for chaining

Key Workflows:
- **Document Storage**: store_knowledge() → returns document ID and metadata for reference
- **Knowledge Retrieval**: search_knowledge() → returns ranked results with relevance scores
- **Document Management**: get_document() → retrieve full content and metadata
- **Bulk Operations**: list_documents() → overview of all stored knowledge
- **Content Updates**: update_document() → modify existing documents while preserving history

Search Capabilities:
- Natural language queries with semantic understanding
- Relevance scoring for result ranking  
- Tag-based filtering and metadata search
- Hybrid search combining text matching and vector similarity
- Content previews and summaries in results

Response Format:
All tools return structured data with:
- Success/failure status and human-readable messages
- Document IDs for operation tracking and chaining
- Rich metadata including titles, tags, word counts, timestamps
- Content previews and relevance scores
- Error details for troubleshooting

Best Practices:
- Use descriptive titles and comprehensive tags for better searchability
- Include relevant metadata for document categorization
- Use natural language in search queries for best semantic matching
- Chain operations using returned document IDs
- Monitor server status for performance insights

For advanced usage and troubleshooting, use get_server_status() to view system state and document tracking information.
""",
    on_duplicate_tools="warn",
    on_duplicate_resources="warn",
    on_duplicate_prompts="warn",
)

# Global state and tracking
_db_initialized = False
_last_operation = {
    "document_ids": [],
    "current_document_id": None,
    "search_results": [],
    "total_documents": 0,
}


@dataclass
class DocumentInfo:
    """Document metadata for AI decision making"""

    id: int
    title: str
    preview: str
    tags: List[str]
    updated_at: str
    relevance_score: Optional[float] = None
    has_full_content: bool = False
    word_count: Optional[int] = None
    content_summary: Optional[str] = None


@dataclass
class SearchResult:
    """Search result with document info and relevance"""

    query: str
    total_results: int
    documents: List[DocumentInfo]
    document_ids: List[int]


@dataclass
class OperationResult:
    """Standard operation result format"""

    success: bool
    message: str
    document_ids: List[int]
    documents: Optional[List[DocumentInfo]] = None
    current_document: Optional[DocumentInfo] = None
    error: Optional[str] = None
    total_documents: Optional[int] = None


def create_document_info(doc: Dict[str, Any], include_preview: bool = True) -> DocumentInfo:
    """Create DocumentInfo from document dict"""
    preview = ""
    if include_preview and doc.get("markdown"):
        preview = doc["markdown"][:150] + "..." if len(doc["markdown"]) > 150 else doc["markdown"]

    word_count = len(doc.get("markdown", "").split()) if doc.get("markdown") else None

    return DocumentInfo(
        id=doc["id"],
        title=doc["title"],
        preview=preview,
        tags=doc.get("tags", []),
        updated_at=doc["updated_at"],
        relevance_score=doc.get("relevance_score"),
        has_full_content=bool(doc.get("markdown")),
        word_count=word_count,
        content_summary=f"Document contains {word_count} words" if word_count else None,
    )


# Knowledge management tools
@mcp.tool(
    description="Store knowledge content in markdown format with metadata and tags for semantic search. Returns document ID and metadata for chaining operations. Use descriptive titles and comprehensive tags for better searchability."
)
def store_knowledge(title: str, markdown: str, tags: list = None, metadata: dict = None) -> Dict:
    """Store knowledge content in markdown format"""
    try:
        ensure_database()
        knowledge_service = safe_import_knowledge_service()
        if not knowledge_service:
            return asdict(
                OperationResult(
                    success=False,
                    message="❌ Error: Knowledge service not available",
                    document_ids=[],
                    error="Service unavailable",
                )
            )

        doc_id = knowledge_service.store_knowledge(
            title=title, markdown_content=markdown, tags=tags or [], metadata=metadata or {}
        )

        # Get full document info
        doc = knowledge_service.get_document(doc_id)
        doc_info = create_document_info(doc) if doc else None

        # Update tracking
        _last_operation["document_ids"] = [doc_id]
        _last_operation["current_document_id"] = doc_id

        return asdict(
            OperationResult(
                success=True,
                message=f"✅ Successfully stored knowledge document: '{title}' (ID: {doc_id})",
                document_ids=[doc_id],
                current_document=doc_info,
            )
        )
    except Exception as e:
        logger.error(f"Error storing knowledge: {e}")
        return asdict(
            OperationResult(
                success=False,
                message=f"❌ Error storing knowledge: {str(e)}",
                document_ids=[],
                error=str(e),
            )
        )


@mcp.tool(
    description="Search stored knowledge using hybrid text + vector search with semantic understanding and relevance ranking. Use natural language queries for best results. Returns ranked documents with relevance scores, document IDs, and content previews for AI decision making."
)
def search_knowledge(query: str, limit: int = 10) -> Dict:
    """Search stored knowledge using hybrid text + vector search

    Returns:
        - List of document IDs and metadata
        - Relevance scores for AI decision making
        - Content previews for quick assessment
    """
    try:
        ensure_database()
        knowledge_service = safe_import_knowledge_service()
        if not knowledge_service:
            return asdict(
                OperationResult(
                    success=False,
                    message="❌ Error: Knowledge service not available",
                    document_ids=[],
                    error="Service unavailable",
                )
            )

        # Validate limit
        if limit < 1 or limit > 50:
            limit = 10

        results = knowledge_service.search_knowledge(query=query, limit=limit)

        if results:
            # Create document infos
            doc_infos = [create_document_info(doc) for doc in results]
            doc_ids = [doc.id for doc in doc_infos]

            # Update tracking
            _last_operation["document_ids"] = doc_ids
            _last_operation["search_results"] = results

            # Format human-readable response
            response = f"🔍 Found {len(results)} results for: '{query}'\n\n"
            for i, doc in enumerate(doc_infos, 1):
                response += f"{i}. **{doc.title}** (ID: {doc.id})\n"
                response += f"   📊 Relevance: {doc.relevance_score:.3f}\n"
                if doc.tags:
                    response += f"   🏷️ Tags: {', '.join(doc.tags)}\n"
                response += f"   📄 Preview: {doc.preview}\n\n"

            return asdict(
                OperationResult(
                    success=True, message=response, document_ids=doc_ids, documents=doc_infos
                )
            )
        else:
            return asdict(
                OperationResult(
                    success=True,
                    message=f"❌ No results found for: '{query}'",
                    document_ids=[],
                    documents=[],
                )
            )

    except Exception as e:
        logger.error(f"Error searching knowledge: {e}")
        return asdict(
            OperationResult(
                success=False,
                message=f"❌ Error searching knowledge: {str(e)}",
                document_ids=[],
                error=str(e),
            )
        )


@mcp.tool(
    description="Retrieve a specific knowledge document by ID with full content and metadata. Returns complete document information including markdown content, tags, timestamps, and word count. Use document IDs from search results or list operations."
)
def get_document(document_id: int, include_content: bool = True) -> Dict:
    """Retrieve a specific knowledge document by ID

    Args:
        document_id: The ID of the document to retrieve
        include_content: Whether to include full markdown content (default: True)

    Returns:
        - Document metadata for AI processing
        - Full content if requested
        - Word count and content summary
    """
    try:
        ensure_database()
        knowledge_service = safe_import_knowledge_service()
        if not knowledge_service:
            return asdict(
                OperationResult(
                    success=False,
                    message="❌ Error: Knowledge service not available",
                    document_ids=[],
                    error="Service unavailable",
                )
            )

        doc = knowledge_service.get_document(document_id)

        if doc:
            # Update tracking
            _last_operation["current_document_id"] = document_id

            # Create document info
            doc_info = create_document_info(doc, include_preview=include_content)

            # Format human-readable response
            response = f"📄 **{doc['title']}** (ID: {doc['id']})\n\n"
            response += f"📅 Created: {doc['created_at']}\n"
            response += f"🔄 Updated: {doc['updated_at']}\n"
            if doc.get("tags"):
                response += f"🏷️ Tags: {', '.join(doc['tags'])}\n"
            if doc_info.word_count:
                response += f"📝 Words: {doc_info.word_count}\n"
            if include_content:
                response += f"\n📝 **Content:**\n```markdown\n{doc['markdown']}\n```"

            return asdict(
                OperationResult(
                    success=True,
                    message=response,
                    document_ids=[document_id],
                    current_document=doc_info,
                )
            )
        else:
            return asdict(
                OperationResult(
                    success=False,
                    message=f"❌ Document not found: {document_id}",
                    document_ids=[],
                    error="Document not found",
                )
            )

    except Exception as e:
        logger.error(f"Error getting document: {e}")
        return asdict(
            OperationResult(
                success=False,
                message=f"❌ Error getting document: {str(e)}",
                document_ids=[],
                error=str(e),
            )
        )


@mcp.tool(
    description="List all stored knowledge documents with metadata, previews, and pagination support. Returns document overviews with IDs, titles, tags, word counts, and content previews. Use for browsing and discovering existing knowledge."
)
def list_documents(limit: int = 20, offset: int = 0) -> Dict:
    """List all stored knowledge documents with metadata

    Returns:
        - List of document IDs and metadata
        - Content previews and word counts
        - Total document count
    """
    try:
        ensure_database()
        knowledge_service = safe_import_knowledge_service()
        if not knowledge_service:
            return asdict(
                OperationResult(
                    success=False,
                    message="❌ Error: Knowledge service not available",
                    document_ids=[],
                    error="Service unavailable",
                )
            )

        # Validate parameters
        if limit < 1 or limit > 100:
            limit = 20
        if offset < 0:
            offset = 0

        docs = knowledge_service.list_documents(limit=limit, offset=offset)

        if docs:
            # Create document infos
            doc_infos = [create_document_info(doc) for doc in docs]
            doc_ids = [doc.id for doc in doc_infos]

            # Update tracking
            _last_operation["document_ids"] = doc_ids
            _last_operation["total_documents"] = len(docs)

            # Format human-readable response
            response = f"📚 Found {len(docs)} documents:\n\n"
            for doc in doc_infos:
                response += f"• **{doc.title}** (ID: {doc.id})\n"
                response += f"  📅 Updated: {doc.updated_at}\n"
                if doc.tags:
                    response += f"  🏷️ Tags: {', '.join(doc.tags)}\n"
                if doc.word_count:
                    response += f"  📝 Words: {doc.word_count}\n"
                response += f"  📄 Preview: {doc.preview}\n\n"

            return asdict(
                OperationResult(
                    success=True,
                    message=response,
                    document_ids=doc_ids,
                    documents=doc_infos,
                    total_documents=len(docs),
                )
            )
        else:
            return asdict(
                OperationResult(
                    success=True,
                    message="📚 No documents found",
                    document_ids=[],
                    documents=[],
                    total_documents=0,
                )
            )

    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        return asdict(
            OperationResult(
                success=False,
                message=f"❌ Error listing documents: {str(e)}",
                document_ids=[],
                error=str(e),
            )
        )


@mcp.tool(
    description="Update an existing knowledge document with new content, metadata, or tags while preserving history. Allows partial updates - only provide fields you want to change. Returns updated document metadata for verification."
)
def update_document(
    document_id: int,
    title: str = None,
    markdown: str = None,
    tags: list = None,
    metadata: dict = None,
) -> Dict:
    """Update an existing knowledge document

    Args:
        document_id: The ID of the document to update
        title: New title (optional)
        markdown: New content (optional)
        tags: New tags (optional)
        metadata: New metadata (optional)
    """
    try:
        ensure_database()
        knowledge_service = safe_import_knowledge_service()
        if not knowledge_service:
            return asdict(
                OperationResult(
                    success=False,
                    message="❌ Error: Knowledge service not available",
                    document_ids=[],
                    error="Service unavailable",
                )
            )

        success = knowledge_service.update_document(
            doc_id=document_id,
            title=title,
            markdown_content=markdown,
            tags=tags,
            metadata=metadata,
        )

        if success:
            # Get updated document info
            doc = knowledge_service.get_document(document_id)
            doc_info = create_document_info(doc) if doc else None

            # Update tracking
            _last_operation["current_document_id"] = document_id

            return asdict(
                OperationResult(
                    success=True,
                    message=f"✅ Successfully updated document: {document_id}",
                    document_ids=[document_id],
                    current_document=doc_info,
                )
            )
        else:
            return asdict(
                OperationResult(
                    success=False,
                    message=f"❌ Document not found: {document_id}",
                    document_ids=[],
                    error="Document not found",
                )
            )

    except Exception as e:
        logger.error(f"Error updating document: {e}")
        return asdict(
            OperationResult(
                success=False,
                message=f"❌ Error updating document: {str(e)}",
                document_ids=[],
                error=str(e),
            )
        )


@mcp.tool(
    description="Delete a knowledge document by ID permanently from the knowledge base. This action cannot be undone. Returns information about the deleted document for confirmation."
)
def delete_document(document_id: int) -> Dict:
    """Delete a knowledge document by ID"""
    try:
        ensure_database()
        knowledge_service = safe_import_knowledge_service()
        if not knowledge_service:
            return asdict(
                OperationResult(
                    success=False,
                    message="❌ Error: Knowledge service not available",
                    document_ids=[],
                    error="Service unavailable",
                )
            )

        # Get document info before deletion for response
        doc = knowledge_service.get_document(document_id)
        doc_info = create_document_info(doc) if doc else None

        success = knowledge_service.delete_document(document_id)

        if success:
            return asdict(
                OperationResult(
                    success=True,
                    message=f"✅ Successfully deleted document: {document_id}",
                    document_ids=[document_id],
                    current_document=doc_info,
                )
            )
        else:
            return asdict(
                OperationResult(
                    success=False,
                    message=f"❌ Document not found: {document_id}",
                    document_ids=[],
                    error="Document not found",
                )
            )

    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        return asdict(
            OperationResult(
                success=False,
                message=f"❌ Error deleting document: {str(e)}",
                document_ids=[],
                error=str(e),
            )
        )


@mcp.tool(
    description="Get detailed server status including database health, document tracking, and system performance metrics. Returns comprehensive system information for monitoring and debugging."
)
def get_server_status() -> Dict:
    """Get detailed server status including document tracking"""
    status = {
        "status": "running",
        "version": "fastmcp-ai-1.0",
        "database_initialized": _db_initialized,
        "document_tracking": {
            "last_document_ids": _last_operation["document_ids"],
            "current_document": _last_operation["current_document_id"],
            "total_documents": _last_operation["total_documents"],
        },
    }

    try:
        if _db_initialized:
            knowledge_service = safe_import_knowledge_service()
            if knowledge_service:
                docs = knowledge_service.list_documents(limit=1)
                status.update(
                    {
                        "database": "postgresql+pgvector",
                        "embedding_model": "bge-m3:567m",
                        "has_documents": len(docs) > 0,
                    }
                )
    except Exception as e:
        status["database_error"] = str(e)

    return status


# Helper functions
def ensure_database():
    """Lazy database initialization"""
    global _db_initialized
    if not _db_initialized:
        try:
            from database import init_database

            init_database()
            _db_initialized = True
            logger.info("✅ Database initialized successfully!")
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            raise


def safe_import_knowledge_service():
    """Safely import knowledge service"""
    try:
        import knowledge_service

        return knowledge_service
    except Exception as e:
        logger.error(f"❌ Failed to import knowledge_service: {e}")
        return None


if __name__ == "__main__":
    logger.info("🚀 Starting Knowledge MCP Server (AI-Friendly Mode)...")
    logger.info("📡 Database will be initialized on first use")
    logger.info("🎉 Knowledge MCP Server is ready!")
    mcp.run()
