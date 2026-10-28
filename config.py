import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()


def _clean_key(val: str | None) -> str | None:
    if not val or val.strip().startswith("your-"):
        return None
    return val.strip()


OPENAI_API_KEY = _clean_key(os.getenv("OPENAI_API_KEY"))
TAVILY_API_KEY = _clean_key(os.getenv("TAVILY_API_KEY"))
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
DATABASE_PATH = os.getenv("DATABASE_PATH", "research.db")
