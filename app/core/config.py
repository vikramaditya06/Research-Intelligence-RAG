from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://rag:rag@localhost:5432/research_rag"
    openai_api_key: str = ""
    openai_base_url: str | None = None
    llm_model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    chunk_size: int = 900
    chunk_overlap: int = 120
    parent_size: int = 1800
    top_k_vector: int = 12
    top_k_bm25: int = 12
    default_top_k: int = Field(default=8, validation_alias=AliasChoices("DEFAULT_TOP_K", "TOP_K_RERANK"))
    retrieval_candidate_k: int = 24
    max_upload_bytes: int = 50 * 1024 * 1024
    max_pages: int = 500
    embedding_batch_size: int = 64
    data_dir: str = "data/documents"
    enable_ocr: bool = False
    ocr_min_chars: int = 80
    enable_multi_query: bool = True
    enable_compression: bool = True
    enable_citation_audit: bool = True

    @field_validator("embedding_dimension")
    @classmethod
    def validate_embedding_dimension(cls, value: int) -> int:
        # The Alembic schema is intentionally fixed at 1536 dimensions to keep
        # migrations deterministic. Fail fast rather than allowing an env
        # override to create an ORM/database dimension mismatch.
        if value != 1536:
            raise ValueError("EMBEDDING_DIMENSION must be 1536 for the current database schema")
        return value

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
