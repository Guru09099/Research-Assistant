from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Autonomous Research Assistant"
    openai_api_key: str | None = None
    gemini_api_key: str | None = None
    serpapi_api_key: str | None = None
    vector_backend: str = "faiss"
    vector_dim: int = 384
    memory_store_path: str = "data/memory.json"
    vector_store_path: str = "data/vector_store.json"
    pdf_storage_dir: str = "data/pdfs"
    llm_model: str = "gpt-4o-mini"
    max_search_results: int = 5
    use_langchain: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
