"""Application settings using pydantic-settings."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """ReflectAI configuration loaded from environment variables."""

    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    llm_model: str = "nvidia/nemotron-3-super-120b-a12b:free"
    llm_temperature: float = 0.7
    debug: bool = False

    model_config = {
        "env_file": ".env",
        "env_prefix": "REFLECTAI_",
        "case_sensitive": False,
        "populate_by_name": True,
    }


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
