"""
Centralized application configuration.

All environment-dependent settings are loaded through this module.
Secrets should never be hardcoded in source code.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    app_name: str = (
        "AI Business Intelligence & Decision Support Agent"
    )
    app_version: str = "1.0.0"
    environment: str = "development"
    debug: bool = True

    # LLM
    llm_provider: str = "openai"
    llm_model: str = "gpt-4.1-mini"
    llm_temperature: float = 0.0
    openai_api_key: str = ""

    # Database
    database_url: str = (
        "sqlite:///./data/business_intelligence.db"
    )

    # Agent
    max_agent_iterations: int = 5
    max_sql_rows: int = 1000
    agent_timeout_seconds: int = 60

    # RAG
    embedding_model: str = "all-MiniLM-L6-v2"
    rag_top_k: int = 5
    rag_min_score: float = 0.30

    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/agent.log"

    # API
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    # Dashboard
    streamlit_port: int = 8501

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached application settings instance."""
    return Settings()