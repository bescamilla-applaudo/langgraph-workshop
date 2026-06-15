from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="REFLECTAI_",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # alias bypasses env_prefix — reads OPENROUTER_API_KEY, not REFLECTAI_OPENROUTER_API_KEY
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")

    llm_model: str = "nvidia/nemotron-3-super-120b-a12b:free"         # REFLECTAI_LLM_MODEL
    llm_temperature: float = 0.7                                      # REFLECTAI_LLM_TEMPERATURE
    debug: bool = False                                               # REFLECTAI_DEBUG


@lru_cache
def get_settings() -> Settings:
    return Settings()
