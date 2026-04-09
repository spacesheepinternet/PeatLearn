"""
Ray Peat Legacy - Central Configuration Settings
"""

import os
import secrets
from pathlib import Path
from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Project Information
    PROJECT_NAME: str = "Ray Peat Legacy"
    PROJECT_DESCRIPTION: str = "Bioenergetic Knowledge Platform"
    VERSION: str = "1.0.0"
    
    # Environment
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    DEBUG: bool = Field(default=True, env="DEBUG")
    
    # API Keys
    GEMINI_API_KEY: Optional[str] = Field(default=None, env="GEMINI_API_KEY")
    OPENAI_API_KEY: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    PINECONE_API_KEY: Optional[str] = Field(default=None, env="PINECONE_API_KEY")
    GROQ_API_KEY: Optional[str] = Field(default=None, env="GROQ_API_KEY")
    
    # Database Configuration
    DATABASE_URL: str = Field(
        default="sqlite:///./data/peat_legacy.db", 
        env="DATABASE_URL"
    )
    REDIS_URL: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    
    # Vector Database
    VECTOR_DB_TYPE: str = Field(default="pinecone", env="VECTOR_DB_TYPE")  # chromadb, pinecone, qdrant
    CHROMA_PERSIST_DIR: str = str(PROJECT_ROOT / "data" / "vectorstore" / "chroma")
    
    # Data Paths
    RAW_DATA_DIR: str = str(PROJECT_ROOT / "data" / "raw" / "raw_data")
    PROCESSED_DATA_DIR: str = str(PROJECT_ROOT / "data" / "processed")
    ANALYSIS_DATA_DIR: str = str(PROJECT_ROOT / "data" / "analysis")
    LOGS_DIR: str = str(PROJECT_ROOT / "logs")
    
    # Processing Settings
    BATCH_SIZE: int = Field(default=10, env="BATCH_SIZE")
    MAX_TOKENS_PER_CHUNK: int = Field(default=1000, env="MAX_TOKENS_PER_CHUNK")
    CHUNK_OVERLAP: int = Field(default=200, env="CHUNK_OVERLAP")
    
    # Quality Thresholds
    TIER1_FIDELITY_THRESHOLD: float = 4.0
    TIER1_ATOMICITY_THRESHOLD: float = 5.0
    TIER1_NOISE_THRESHOLD: float = 5.0
    
    # API Settings
    API_HOST: str = Field(default="localhost", env="API_HOST")
    API_PORT: int = Field(default=8000, env="API_PORT")
    API_WORKERS: int = Field(default=1, env="API_WORKERS")
    
    # Embedding Settings
    EMBEDDING_MODEL: str = Field(default="gemini-embedding-001", env="EMBEDDING_MODEL")
    EMBEDDING_DIMENSIONS: int = Field(default=3072, env="EMBEDDING_DIMENSIONS")
    
    # LLM Settings
    DEFAULT_LLM_MODEL: str = Field(default="gemini-2.5-flash", env="DEFAULT_LLM_MODEL")
    MAX_CONTEXT_LENGTH: int = Field(default=32000, env="MAX_CONTEXT_LENGTH")
    TEMPERATURE: float = Field(default=0.1, env="TEMPERATURE")
    
    # Rate Limiting
    API_RATE_LIMIT: int = Field(default=60, env="API_RATE_LIMIT")  # requests per minute
    GEMINI_RATE_LIMIT: int = Field(default=15, env="GEMINI_RATE_LIMIT")  # requests per minute
    
    # Security
    SECRET_KEY: str = Field(default_factory=lambda: secrets.token_hex(32), env="SECRET_KEY")
    CORS_ORIGINS: list = ["http://localhost:3000", "http://localhost:8000"]
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}"
    
    class Config:
        env_file = PROJECT_ROOT / ".env"
        case_sensitive = True

# Global settings instance
settings = Settings()

# Data paths for easy access
PATHS = {
    "project_root": PROJECT_ROOT,
    "data": PROJECT_ROOT / "data",
    "raw_data": Path(settings.RAW_DATA_DIR),
    "processed_data": Path(settings.PROCESSED_DATA_DIR),
    "analysis_data": Path(settings.ANALYSIS_DATA_DIR),
    "logs": Path(settings.LOGS_DIR),
    "preprocessing": PROJECT_ROOT / "preprocessing",
    "peatlearn": PROJECT_ROOT / "peatlearn",
    "app": PROJECT_ROOT / "app",
    "web_ui": PROJECT_ROOT / "web_ui",
    "tests": PROJECT_ROOT / "tests",
    "docs": PROJECT_ROOT / "docs",
    "config": PROJECT_ROOT / "config"
}

# Ensure required directories exist
def ensure_directories():
    """Create required directories if they don't exist."""
    for path in PATHS.values():
        if isinstance(path, Path):
            path.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    ensure_directories()
    print("Ray Peat Legacy Configuration")
    print(f"Environment: {settings.ENVIRONMENT}")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Directory: {PATHS['data']}")
    print("All required directories ensured.") 