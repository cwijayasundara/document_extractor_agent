"""Document parsing tool using LlamaParse."""

import logging
from typing import Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.agent.tools.path_utils import resolve_document_path

# Configure logging
logger = logging.getLogger(__name__)


class ParseDocumentInput(BaseModel):
    """Input schema for parse_document tool."""

    file_path: str = Field(
        description=(
            "Path to the document file (PDF, PNG, JPG, JPEG). Accepts absolute paths "
            "or paths relative to the project root, docs/, or uploads/ directories."
        )
    )


class ParseDocumentTool(BaseTool):
    """Tool to parse PDF/image documents to markdown using LlamaParse."""

    name: str = "parse_document"
    description: str = """Parse a PDF or image document to markdown text using LlamaParse.
    This tool extracts all text content from the document in a structured markdown format.
    The parsing result is cached for faster subsequent access.

    Input: file_path (absolute path to document)
    Output: Parsed document text in markdown format
    """
    args_schema: Type[BaseModel] = ParseDocumentInput

    async def _arun(self, file_path: str) -> str:
        """Parse document to markdown (async).

        Args:
            file_path: Absolute path to document file

        Returns:
            Parsed markdown text
        """
        resolved_path_str = None
        try:
            logger.info(f"[PARSE_TOOL] Starting document parsing: {file_path}")

            resolved_path = resolve_document_path(file_path)
            resolved_path_str = str(resolved_path)
            logger.info(f"[PARSE_TOOL] Resolved document path: {resolved_path_str}")

            from src.plain.cache_utils import get_cached_parse, save_parsed_content
            from src.plain.config import get_parse_client

            # Check cache first
            logger.info("[PARSE_TOOL] Checking cache...")
            cached = get_cached_parse(resolved_path_str)
            if cached:
                logger.info(f"[PARSE_TOOL] Cache hit! Returning {len(cached)} characters")
                return cached

            logger.info("[PARSE_TOOL] Cache miss. Creating parse client...")

            # Parse document
            parse_client = await get_parse_client()
            logger.info(f"[PARSE_TOOL] Parse client created. Calling aparse() on {resolved_path_str}...")

            result = await parse_client.aparse(resolved_path_str)
            logger.info(f"[PARSE_TOOL] Parse completed. Processing {len(result.pages)} pages...")

            # Combine all pages into single content string (like plain implementation)
            content = "\n\n".join([page.md for page in result.pages])
            logger.info(f"[PARSE_TOOL] Extracted {len(content)} characters from {len(result.pages)} pages")

            # Save to cache
            logger.info("[PARSE_TOOL] Saving to cache...")
            save_parsed_content(resolved_path_str, content)
            logger.info("[PARSE_TOOL] Parsing complete!")

            return content

        except Exception as e:
            logger.error(f"[PARSE_TOOL] Error during parsing: {str(e)}")
            logger.error(f"[PARSE_TOOL] File path: {file_path}")
            if resolved_path_str:
                logger.error(f"[PARSE_TOOL] Resolved path: {resolved_path_str}")
            import traceback
            logger.error(f"[PARSE_TOOL] Traceback:\n{traceback.format_exc()}")
            raise Exception(f"Document parsing failed for {file_path}: {str(e)}") from e

    def _run(self, file_path: str) -> str:
        """Sync wrapper for async implementation."""
        import asyncio

        return asyncio.run(self._arun(file_path))
