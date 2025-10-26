"""LangChain tools for document extraction workflow."""

from src.agent.tools.parse_tool import ParseDocumentTool
from src.agent.tools.schema_ai_tool import GenerateSchemaAITool
from src.agent.tools.schema_template_tool import GenerateSchemaTemplateTool
from src.agent.tools.extraction_tool import ExtractDataTool
from src.agent.tools.validation_tool import ValidateSchemaTool


def get_all_tools():
    """Get list of all available tools.

    Returns:
        List of BaseTool instances
    """
    return [
        ParseDocumentTool(),
        GenerateSchemaAITool(),
        GenerateSchemaTemplateTool(),
        ExtractDataTool(),
        ValidateSchemaTool(),
    ]


__all__ = [
    "ParseDocumentTool",
    "GenerateSchemaAITool",
    "GenerateSchemaTemplateTool",
    "ExtractDataTool",
    "ValidateSchemaTool",
    "get_all_tools",
]
