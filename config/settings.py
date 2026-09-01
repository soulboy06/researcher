# -*- coding: utf-8 -*-
"""
Configuration Management Module.
Utilizes Pydantic / python-dotenv for strict typing, default fallbacks, and validation.
Works seamlessly with or without pydantic_settings installed.
"""
import os
from typing import Optional
from pathlib import Path

# Root project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# 1. Load .env file automatically using python-dotenv if available, or manual fallback
try:
    from dotenv import load_dotenv
    # Try loading from researcher/.env first, then parent .env
    if (BASE_DIR / ".env").exists():
        load_dotenv(dotenv_path=str(BASE_DIR / ".env"), override=True)
    elif (BASE_DIR.parent / ".env").exists():
        load_dotenv(dotenv_path=str(BASE_DIR.parent / ".env"), override=True)
    else:
        load_dotenv(override=True)
except ImportError:
    # Manual simple .env parser fallback
    for env_path in [BASE_DIR / ".env", BASE_DIR.parent / ".env"]:
        if env_path.exists():
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip("'\"")
                            if k not in os.environ:
                                os.environ[k] = v
            except Exception:
                pass

# 2. Logger setup
try:
    from loguru import logger
    HAS_LOGURU = True
    log_dir = BASE_DIR.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_dir / "server.log"), rotation="20 MB", encoding="utf-8", level="INFO")
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s")
    logger = logging.getLogger("researcher")
    HAS_LOGURU = False

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    from pydantic import Field
    HAS_PYDANTIC_SETTINGS = True
except ImportError:
    from pydantic import BaseModel as BaseSettings, Field
    HAS_PYDANTIC_SETTINGS = False


class Settings(BaseSettings):
    """Global system configuration settings."""

    # 1. LLM Core Configuration
    llm_api_key: str = Field(
        default_factory=lambda: os.getenv("LLM_API_KEY", "EMPTY"),
        alias="LLM_API_KEY"
    )
    llm_base_url: str = Field(
        default_factory=lambda: os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
        alias="LLM_BASE_URL"
    )
    llm_model_name: str = Field(
        default_factory=lambda: os.getenv("LLM_MODEL_NAME", "gpt-4o"),
        alias="LLM_MODEL_NAME"
    )
    llm_timeout: float = Field(
        default_factory=lambda: float(os.getenv("LLM_TIMEOUT", "180.0")),
        alias="LLM_TIMEOUT"
    )
    llm_max_retries: int = Field(
        default_factory=lambda: int(os.getenv("LLM_MAX_RETRIES", "3")),
        alias="LLM_MAX_RETRIES"
    )
    llm_retry_base_delay: float = Field(
        default_factory=lambda: float(os.getenv("LLM_RETRY_BASE_DELAY", "1.0")),
        alias="LLM_RETRY_BASE_DELAY"
    )
    llm_retry_max_delay: float = Field(
        default_factory=lambda: float(os.getenv("LLM_RETRY_MAX_DELAY", "30.0")),
        alias="LLM_RETRY_MAX_DELAY"
    )

    # 2. Secondary / Summary Model
    summary_model_name: str = Field(
        default_factory=lambda: os.getenv("SUMMARY_MODEL_NAME", "gpt-4o-mini"),
        alias="SUMMARY_MODEL_NAME"
    )

    # 3. Search & Reader API Keys
    tavily_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("TAVILY_API_KEY") or None,
        alias="TAVILY_API_KEY"
    )
    serper_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("SERPER_API_KEY") or None,
        alias="SERPER_API_KEY"
    )
    jina_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("JINA_API_KEY") or None,
        alias="JINA_API_KEY"
    )

    # 4. Agent & Context Guard Settings
    log_level: str = Field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"),
        alias="LOG_LEVEL"
    )
    max_input_tokens: int = Field(
        default_factory=lambda: int(os.getenv("MAX_INPUT_TOKENS", "64000")),
        alias="MAX_INPUT_TOKENS"
    )
    obs_max_chars: int = Field(
        default_factory=lambda: int(os.getenv("OBS_MAX_CHARS", "30000")),
        alias="OBS_MAX_CHARS"
    )
    default_workspace: str = Field(
        default_factory=lambda: os.getenv("DEFAULT_WORKSPACE", "./workspace"),
        alias="DEFAULT_WORKSPACE"
    )
    max_agent_rounds: int = Field(
        default_factory=lambda: int(os.getenv("MAX_AGENT_ROUNDS", "20")),
        alias="MAX_AGENT_ROUNDS"
    )
    agent_timeout: float = Field(
        default_factory=lambda: float(os.getenv("AGENT_TIMEOUT", "1800.0")),
        alias="AGENT_TIMEOUT"
    )

    if HAS_PYDANTIC_SETTINGS:
        model_config = SettingsConfigDict(
            env_file=(str(BASE_DIR / ".env"), str(BASE_DIR.parent / ".env")),
            env_file_encoding="utf-8",
            extra="ignore",
            populate_by_name=True,
        )


# Global singleton settings instance
settings = Settings()

# Configure loguru logger if available
if HAS_LOGURU:
    import sys
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level.upper(),
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )
