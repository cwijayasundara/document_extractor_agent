"""Configuration module for the extraction workflow.

This module contains:
- Environment variable loading
- API configuration constants
- Client factory functions for LlamaParse, LlamaExtract, and LLM providers
"""

import warnings
warnings.filterwarnings("ignore")

from openai import AsyncOpenAI
from llama_cloud_services.extract import LlamaExtract
from llama_cloud_services.parse import LlamaParse
from dotenv import load_dotenv
import os
from pathlib import Path
from typing import Dict, Any

# Load environment variables
load_dotenv()

# LLM Provider Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()  # "openai" or "groq"
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5-nano")

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Legacy configuration (backward compatibility)
OPENAI_MODEL = LLM_MODEL  # For existing code that references OPENAI_MODEL

# Llama Cloud Configuration
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
LLAMA_CLOUD_PROJECT_ID = os.getenv("LLAMA_CLOUD_PROJECT_ID")
LLAMA_CLOUD_ORG_ID = os.getenv("LLAMA_CLOUD_ORG_ID")

# Cache Configuration
# Use root-level parsed directory (shared by both plain and agent implementations)
PARSE_CACHE_DIR = Path(__file__).parent.parent.parent / "parsed"

# Model-specific configurations
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    # OpenAI models
    "gpt-5-nano": {
        "temperature": 1.0,  # gpt-5-nano only supports temperature=1
        "supports_json": True,
        "max_tokens": 4096,
        "provider": "openai"
    },
    "gpt-4o": {
        "temperature": 0.7,
        "supports_json": True,
        "max_tokens": 16384,
        "provider": "openai"
    },
    "gpt-4o-mini": {
        "temperature": 0.7,
        "supports_json": True,
        "max_tokens": 16384,
        "provider": "openai"
    },
    # Groq models
    "moonshotai/kimi-k2-instruct-0905": {
        "temperature": 0.7,
        "supports_json": True,
        "max_tokens": 8192,
        "context_window": 256000,
        "provider": "groq"
    },
    "llama-3.3-70b-versatile": {
        "temperature": 0.7,
        "supports_json": True,
        "max_tokens": 8192,
        "context_window": 128000,
        "provider": "groq"
    },
    "mixtral-8x7b-32768": {
        "temperature": 0.7,
        "supports_json": True,
        "max_tokens": 8192,
        "context_window": 32768,
        "provider": "groq"
    }
}


def get_model_config(model: str) -> Dict[str, Any]:
    """Get configuration for a specific model.

    Args:
        model: Model identifier

    Returns:
        Dictionary with model configuration (temperature, max_tokens, etc.)
    """
    return MODEL_CONFIGS.get(model, {
        "temperature": 0.7,
        "supports_json": True,
        "max_tokens": 4096,
        "provider": "openai"
    })


# Client Factory Functions
async def get_parse_client(**kwargs):
    """Create and configure a LlamaParse client instance.

    Returns:
        LlamaParse: Configured parser client with optimized settings for document parsing.
    """
    return LlamaParse(
        api_key=LLAMA_CLOUD_API_KEY,
        parse_mode="parse_page_with_agent",
        model="gemini-2.5-flash",
        high_res_ocr=True,
        adaptive_long_table=True,
        outlined_table_extraction=True,
        output_tables_as_HTML=False,
        take_screenshot=False
    )

async def get_extract_client(**kwargs):
    """Create and configure a LlamaExtract client instance.

    Returns:
        LlamaExtract: Configured extraction client.
    """
    return LlamaExtract(
        api_key=LLAMA_CLOUD_API_KEY,
        project_id=LLAMA_CLOUD_PROJECT_ID,
        organization_id=LLAMA_CLOUD_ORG_ID,
        verbose=True
    )

async def get_openai_client(**kwargs):
    """Create and configure an OpenAI client instance.

    This is a legacy function for backward compatibility.
    For new code, use get_llm_client() instead.

    Returns:
        AsyncOpenAI: Configured OpenAI client for async operations.
    """
    provider = kwargs.get("provider", LLM_PROVIDER)
    model = kwargs.get("model", LLM_MODEL)
    return await get_llm_client(provider=provider, model=model)


async def get_llm_client(provider: str = None, model: str = None, **kwargs):
    """Create and configure an LLM client instance (OpenAI or Groq).

    This factory function creates an AsyncOpenAI client configured for either
    OpenAI or Groq, based on the provider parameter. Groq uses the OpenAI-compatible
    API, so the same client class works for both.

    Args:
        provider: LLM provider - "openai" or "groq" (defaults to LLM_PROVIDER env var)
        model: Model identifier (defaults to LLM_MODEL env var)
        **kwargs: Additional arguments passed to AsyncOpenAI

    Returns:
        AsyncOpenAI: Configured client for async operations

    Raises:
        ValueError: If provider is not supported or API key is missing

    Example:
        >>> client = await get_llm_client(provider="groq", model="moonshotai/kimi-k2-instruct-0905")
        >>> response = await client.chat.completions.create(...)
    """
    # Use defaults if not specified
    provider = (provider or LLM_PROVIDER).lower()
    model = model or LLM_MODEL

    if provider == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        return AsyncOpenAI(
            api_key=OPENAI_API_KEY,
            **kwargs
        )

    elif provider == "groq":
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY environment variable is not set")

        # Groq uses OpenAI-compatible API with custom base URL
        return AsyncOpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
            **kwargs
        )

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}. Use 'openai' or 'groq'.")
