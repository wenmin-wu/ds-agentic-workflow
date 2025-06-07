import os
import logging
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    # Database Configuration
    database_url: str = os.getenv(
        "DATABASE_URL", "postgresql://user:password@localhost:5432/knowledge_mcp"
    )

    # Ollama Configuration
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_token: str = os.getenv("OLLAMA_TOKEN", "")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "bge-m3:567m")
    embedding_dimensions: int = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))

    # Search Configuration
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    max_results: int = int(os.getenv("MAX_RESULTS", "10"))

    # Server Configuration
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    class Config:
        env_file = ".env"


settings = Settings()

# Assert that environment variables are loaded correctly
# Uncomment the line below if you require OLLAMA_TOKEN to be set
# assert settings.ollama_token != "", "OLLAMA_TOKEN must be set in .env file"


# Verify .env file is being loaded (check if any non-default values are present)
def verify_env_loading():
    """Verify that .env file is being loaded successfully"""
    # Check if DATABASE_URL is not the default value (indicates .env is loaded)
    if settings.database_url == "postgresql://user:password@localhost:5432/knowledge_mcp":
        logger.warning("Using default DATABASE_URL - check if .env file exists and is loaded")

    # If OLLAMA_TOKEN is required for your setup, uncomment the assertion below:
    # assert settings.ollama_token != "", "OLLAMA_TOKEN must be set in .env file"

    logger.info(
        f"Config loaded - Ollama URL: {settings.ollama_url}, Model: {settings.embedding_model}"
    )


# Call verification on import
verify_env_loading()
