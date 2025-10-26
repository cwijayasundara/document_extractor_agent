"""Configuration module for the extraction workflow.

This module contains:
- Environment variable loading
- API configuration constants
- Client factory functions for LlamaParse, LlamaExtract, and OpenAI
"""

import warnings
warnings.filterwarnings("ignore")

from openai import AsyncOpenAI
from llama_cloud_services.extract import LlamaExtract
from llama_cloud_services.parse import LlamaParse
from dotenv import load_dotenv
import os
from pathlib import Path

# Load environment variables
load_dotenv()

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-5-nano"
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
LLAMA_CLOUD_PROJECT_ID = os.getenv("LLAMA_CLOUD_PROJECT_ID")
LLAMA_CLOUD_ORG_ID = os.getenv("LLAMA_CLOUD_ORG_ID")

# Cache Configuration
# Use root-level parsed directory (shared by both plain and agent implementations)
PARSE_CACHE_DIR = Path(__file__).parent.parent.parent / "parsed"


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

    Returns:
        AsyncOpenAI: Configured OpenAI client for async operations.
    """
    return AsyncOpenAI(api_key=OPENAI_API_KEY)
