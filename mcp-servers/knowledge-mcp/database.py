from contextlib import contextmanager
import logging
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from models import Base
from config import settings

logger = logging.getLogger(__name__)

# Create engine
engine = create_engine(settings.database_url, echo=False)

# Create session factory
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Get database session with automatic cleanup"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

def init_database():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created/verified successfully!") 
