from pydantic_settings import BaseSettings, SettingsConfigDict
import config


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Autonomous Research Assistant"
    openai_api_key: str | None = config.OPENAI_API_KEY
    tavily_api_key: str | None = config.TAVILY_API_KEY
    llm_model: str = config.LLM_MODEL
    database_path: str = config.DATABASE_PATH
    
    # Legacy vector store & pdf settings
    vector_backend: str = "faiss"
    vector_dim: int = 384
    memory_store_path: str = "data/memory.json"
    vector_store_path: str = "data/vector_store.json"
    pdf_storage_dir: str = "data/pdfs"
    max_search_results: int = 5
    use_langchain: bool = False


settings = Settings()
