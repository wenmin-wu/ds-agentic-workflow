import requests
import logging
from typing import List
from config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self):
        self.ollama_url = settings.ollama_url
        self.model_name = settings.embedding_model
        self.ollama_token = settings.ollama_token

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using Ollama bge-m3:567m"""
        try:
            # Prepare headers
            headers = {"Content-Type": "application/json"}
            if self.ollama_token:
                headers["Authorization"] = f"Bearer {self.ollama_token}"

            response = requests.post(
                f"{self.ollama_url}/api/embeddings",
                json={"model": self.model_name, "prompt": text},
                headers=headers,
                timeout=30,
            )

            if response.status_code == 200:
                return response.json()["embedding"]
            else:
                logger.error(f"Ollama API error: {response.text}")
                raise Exception(f"Failed to generate embedding: {response.text}")

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            raise Exception(f"Ollama connection error: {e}")

    def generate_multiple_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        return [self.generate_embedding(text) for text in texts]


# Global instance
embedding_service = EmbeddingService()
