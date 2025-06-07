# Knowledge MCP Server

A **clean and simple** Knowledge Management MCP Server for Cursor using **FastMCP** that stores markdown knowledge with PostgreSQL + pgvector and uses Ollama's `bge-m3:567m` for embeddings.

## 🌟 Features

- **⚡ FastMCP Framework**: Modern, simplified MCP implementation 
- **📝 Markdown Storage**: Store knowledge as markdown (no separate content field needed)
- **🔍 Hybrid Search**: Combined full-text search + vector similarity search
- **🧠 BGE-M3 Embeddings**: High-quality 1024-dimensional embeddings via Ollama
- **🗄️ PostgreSQL + pgvector**: HNSW indexes for fast vector similarity search
- **🏷️ Tags & Metadata**: Organize knowledge with tags and custom metadata
- **🛠️ Simple Architecture**: Clean Python files with FastMCP decorators

## 🚀 Quick Start

### 1. Prerequisites

- **Python 3.8+**
- **PostgreSQL 15+ with pgvector extension**
- **Ollama with bge-m3:567m model**

#### Install PostgreSQL with pgvector

**macOS (Homebrew):**
```bash
# Install PostgreSQL 15
brew install postgresql@15

# Install pgvector extension (if needed in your environment)
# This script automates the 'pgvector' installation, ensuring it aligns with your PostgreSQL version.
source <(curl -H "PRIVATE-TOKEN: $GRAB_GITLAB_ACCESS_TOKEN" 'https://gitlab.myteksi.net/api/v4/projects/23870/repository/files/scripts%2Fsetup-pgvector.sh?ref=master' | jq -r '.content' | base64 --decode)

# Start PostgreSQL service
brew services start postgresql
```

**Alternative pgvector installation:**
```bash
# If the above script is not available, install pgvector manually
git clone --branch v0.5.1 https://github.com/pgvector/pgvector.git
cd pgvector
make
make install  # may need sudo
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Setup Database

```bash
# Create database
createdb knowledge_mcp

# Enable pgvector extension and run schema
psql knowledge_mcp -c "CREATE EXTENSION IF NOT EXISTS vector;"
psql knowledge_mcp < schema.sql
```

**Verify pgvector installation:**
```bash
psql knowledge_mcp -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
```

### 4. Setup Ollama

```bash
# Pull the embedding model
ollama pull bge-m3:567m
```

### 5. Configure Environment

```bash
cp env.example .env
# Edit .env with your database credentials
```

### 6. Run Server

```bash
python main.py
```

## 🛠️ Configuration

### Environment Variables

```env
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/knowledge_mcp

# Ollama
OLLAMA_URL=http://localhost:11434
EMBEDDING_MODEL=bge-m3:567m

# Search
SIMILARITY_THRESHOLD=0.7
MAX_RESULTS=10
```

### Cursor MCP Configuration

Add to your Cursor MCP config:

```json
{
  "mcpServers": {
    "knowledge-mcp": {
      "command": "python",
      "args": ["main.py"],
      "cwd": "/absolute/path/to/knowledge-mcp"
    }
  }
}
```

## 📖 Usage

### Available Tools

1. **`store_knowledge`** - Store markdown content
2. **`search_knowledge`** - Hybrid search with relevance scoring
3. **`get_document`** - Retrieve document by ID
4. **`list_documents`** - List all documents with pagination
5. **`update_document`** - Update existing document
6. **`delete_document`** - Delete document by ID
8. **`get_server_status`** - Server status and statistics

### Example Usage in Cursor

```
Store this markdown as knowledge:

Title: "Python Best Practices"
Tags: ["python", "coding", "best-practices"]

# Python Best Practices

## Code Style
- Use PEP 8
- Write descriptive variable names
- Keep functions small

## Testing
- Write unit tests
- Use pytest
- Aim for high coverage
```

Then search:
```
Search for "python testing frameworks"
```

## 🏗️ Architecture

**FastMCP Implementation** - Single file with decorators:

```
knowledge-mcp/
├── main.py                # FastMCP server with @mcp.tool() decorators ⭐
├── schema.sql             # Database schema with HNSW indexes
├── config.py              # Configuration management
├── models.py              # SQLAlchemy document model
├── database.py            # Database connection & session management
├── embedding_service.py   # Ollama bge-m3:567m integration
├── knowledge_service.py   # Core knowledge operations
├── requirements.txt       # Dependencies (now includes fastmcp)
├── mcp-config.json       # Cursor MCP configuration
└── README.md             # This file
```

### FastMCP Benefits

- **Simplified**: Single file with `@mcp.tool()` decorators
- **Cleaner**: No complex MCP protocol handling
- **Faster**: Streamlined initialization and tool registration
- **Debugging**: Built-in demo tools for testing connections

## 🔧 Database Schema

- **`documents`** table with `id`, `title`, `markdown`, `metadata`, `embedding`, `tags`
- **HNSW vector index** optimized for bge-m3:567m (1024 dimensions)
- **Full-text search** with PostgreSQL GIN indexes on markdown content
- **Automatic timestamps** with triggers

## 🎯 Search Algorithm

1. **Text Search**: PostgreSQL full-text search with `ts_rank_cd`
2. **Vector Search**: Cosine similarity using pgvector HNSW index
3. **Hybrid Scoring**: 30% text score + 70% vector similarity
4. **Threshold Filtering**: Only results above similarity threshold

## 🚦 Performance

- **HNSW Parameters**: `m=32, ef_construction=128` (optimized for bge-m3)
- **Index Strategy**: Separate indexes for time, text, tags, and vectors
- **Connection Pooling**: SQLAlchemy session management
- **Lazy Loading**: Embeddings only generated when needed

## 🔍 Troubleshooting

### Database Issues
```bash
# Check pgvector extension
psql -c "SELECT * FROM pg_extension WHERE extname = 'vector';"

# Test connection
psql $DATABASE_URL -c "SELECT version();"
```

### Ollama Issues
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Test embedding model
curl -X POST http://localhost:11434/api/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "bge-m3:567m", "prompt": "test"}'
```

### MCP Issues
- Check Cursor MCP logs in Developer Tools
- Verify absolute path in `mcp-config.json`
- Ensure Python environment is activated
- Test with `add(2, 3)` or `get_server_status()` for debugging

## 📊 Performance Metrics

- **Embedding Generation**: ~100ms per document (via Ollama)
- **Vector Search**: ~10ms for 1000+ documents (HNSW)
- **Text Search**: ~5ms for full-text queries
- **Storage**: ~1KB per document + embedding overhead

## 🤝 Contributing

This is a clean, simple implementation. Feel free to extend it:

- Add document versioning
- Implement document chunking for large content
- Add more metadata fields
- Optimize embedding caching

```mermaid
graph TB
    subgraph "Dev Layer (Local)"
        cursor[Cursor Agent]
        user[Data Scientist]
        local_git[Local Git]
        knowledge[Knowledge MCP Server]
        jupyter[Jupyter MCP]
        postgres[(PostgreSQL + pgvector)]
        embeddings[BGE-M3 Embeddings]
    end
    
    subgraph "Shared Repository"
        central_repo[Remote Repository]
    end
    
    subgraph "Remote Environment"
        sandbox[Remote Jupyter Environment]
        remote_git[Remote Git]
    end
    
    subgraph "External Sources"
        files[Uploaded Files]
        internet[Web Search]
        docs[Documentation]
    end
    
    %% User interactions
    user --> cursor
    
    %% Agent orchestration
    cursor <--> knowledge
    cursor <--> jupyter
    cursor <--> local_git
    cursor <--> files
    cursor <--> internet
    cursor <--> docs
    
    %% Knowledge layer connections
    knowledge <--> postgres
    knowledge <--> embeddings
    
    %% Execution layer connections
    jupyter <--> sandbox
    jupyter <--> remote_git
    
    %% Git workflow: Local → Remote Repository → Remote Environment
    local_git <--> central_repo
    central_repo <--> remote_git
    
    %% Jupyter MCP orchestrates remote git operations
    jupyter -.-> central_repo
    
    %% Styling for clarity
    classDef localComp fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef remoteComp fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef sharedComp fill:#fff3e0,stroke:#e65100,stroke-width:3px
    classDef externalComp fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    
    class cursor,user,local_git,knowledge,jupyter,postgres,embeddings localComp
    class sandbox,remote_git remoteComp
    class central_repo sharedComp
    class files,internet,docs externalComp
```