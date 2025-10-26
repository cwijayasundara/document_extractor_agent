"""Plain implementation using LlamaIndex Workflows.

This package contains the original implementation of the document extraction workflow.
"""

from src.plain.extraction_workflow import (
    IterativeExtractionWorkflow,
    ProgressEvent,
    ProposedSchema,
    ApprovedSchema,
    RunExtraction,
    ExtractedData,
    InputEvent,
    ParsedContent,
)
from src.plain.config import (
    get_parse_client,
    get_extract_client,
    get_openai_client,
    OPENAI_MODEL,
    PARSE_CACHE_DIR,
)
from src.plain.cache_utils import (
    get_cached_parse,
    save_parsed_content,
    clear_cache,
)
from src.plain.template_schema_generator import (
    TemplateSchemaGenerator,
    generate_schema_from_template,
)
from src.plain.excel_export import export_to_excel

__all__ = [
    # Workflow
    "IterativeExtractionWorkflow",
    "ProgressEvent",
    "ProposedSchema",
    "ApprovedSchema",
    "RunExtraction",
    "ExtractedData",
    "InputEvent",
    "ParsedContent",
    # Config
    "get_parse_client",
    "get_extract_client",
    "get_openai_client",
    "OPENAI_MODEL",
    "PARSE_CACHE_DIR",
    # Cache
    "get_cached_parse",
    "save_parsed_content",
    "clear_cache",
    # Template
    "TemplateSchemaGenerator",
    "generate_schema_from_template",
    # Export
    "export_to_excel",
]
