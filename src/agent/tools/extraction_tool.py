"""Data extraction tool using LlamaExtract."""

import logging
import uuid
from typing import Any, Dict, Type

from langchain_core.tools import BaseTool
from llama_cloud_services.extract import ExtractConfig, ExtractMode
from pydantic import BaseModel, Field

from src.agent.tools.path_utils import resolve_document_path

# Configure logging
logger = logging.getLogger(__name__)


class ExtractDataInput(BaseModel):
    """Input schema for extract_data tool."""

    file_path: str = Field(
        description=(
            "Path to the document file. Accepts absolute paths or paths relative "
            "to the project root, docs/, or uploads/ directories."
        )
    )
    schema: Dict[str, Any] = Field(description="Approved JSON schema for extraction")


class ExtractDataTool(BaseTool):
    """Tool to extract structured data from document using approved schema."""

    name: str = "extract_data"
    description: str = """Extract structured data from a document using an approved JSON schema.
    This tool uses LlamaExtract to perform the actual data extraction.

    Input:
    - file_path: Path to the document
    - schema: Approved JSON schema (must be approved by human first)

    Output: Extracted data as a dictionary matching the schema

    IMPORTANT: This tool should ONLY be called after the schema has been approved by a human.
    """
    args_schema: Type[BaseModel] = ExtractDataInput

    async def _arun(self, file_path: str, schema: dict) -> dict:
        """Extract data from document (async).

        Args:
            file_path: Path to document file
            schema: Approved JSON schema

        Returns:
            Extracted data dictionary
        """
        resolved_path_str = None
        try:
            logger.info(f"[EXTRACT_TOOL] Starting data extraction: {file_path}")
            logger.info(f"[EXTRACT_TOOL] Schema has {len(schema.get('properties', {}))} properties")

            resolved_path = resolve_document_path(file_path)
            resolved_path_str = str(resolved_path)
            logger.info(f"[EXTRACT_TOOL] Resolved document path: {resolved_path_str}")

            from src.plain.config import get_extract_client

            logger.info("[EXTRACT_TOOL] Creating extract client...")
            extract_client = await get_extract_client()

            # Create extraction agent with unique name
            logger.info("[EXTRACT_TOOL] Creating extraction agent with schema...")
            agent = extract_client.create_agent(
                name=f"extraction_tool_{uuid.uuid4()}",
                data_schema=schema,
                config=ExtractConfig(
                    extraction_mode=ExtractMode.BALANCED,
                ),
            )
            logger.info(f"[EXTRACT_TOOL] Agent created with ID: {agent.id}")

            try:
                # Run extraction
                logger.info("[EXTRACT_TOOL] Running extraction...")
                result = await agent.aextract(files=resolved_path_str)
                logger.info("[EXTRACT_TOOL] Extraction completed successfully")

                if result.data is None:
                    raise Exception(f"Extraction failed - no data returned for {resolved_path_str}")

                logger.info(f"[EXTRACT_TOOL] Extracted data keys: {list(result.data.keys())}")

                return result.data

            finally:
                # Cleanup agent
                logger.info(f"[EXTRACT_TOOL] Cleaning up agent {agent.id}...")
                if hasattr(extract_client, "delete_agent"):
                    try:
                        extract_client.delete_agent(agent.id)
                        logger.info("[EXTRACT_TOOL] Agent cleanup successful")
                    except Exception as cleanup_err:
                        logger.warning(f"[EXTRACT_TOOL] Agent cleanup failed: {cleanup_err}")

        except Exception as e:
            logger.error(f"[EXTRACT_TOOL] Error during extraction: {str(e)}")
            logger.error(f"[EXTRACT_TOOL] File path: {file_path}")
            if resolved_path_str:
                logger.error(f"[EXTRACT_TOOL] Resolved path: {resolved_path_str}")
            import traceback
            logger.error(f"[EXTRACT_TOOL] Traceback:\n{traceback.format_exc()}")
            raise Exception(f"Data extraction failed for {file_path}: {str(e)}") from e

    def _run(self, file_path: str, schema: dict) -> dict:
        """Sync wrapper for async implementation."""
        import asyncio

        return asyncio.run(self._arun(file_path, schema))
