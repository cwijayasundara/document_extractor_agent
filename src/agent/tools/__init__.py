"""LangChain tools for document extraction workflow."""

from src.agent.tools.parse_tool import ParseDocumentTool
from src.agent.tools.schema_ai_tool import GenerateSchemaAITool
from src.agent.tools.schema_template_tool import GenerateSchemaTemplateTool
from src.agent.tools.extraction_tool import ExtractDataTool
from src.agent.tools.validation_tool import ValidateSchemaTool


def get_all_tools(provider: str = "openai"):
    """Get list of available tools based on LLM provider.

    Some tools may not be compatible with all providers. This function
    returns a provider-specific tool list.

    Args:
        provider: LLM provider - "openai" or "groq" (default: "openai")

    Returns:
        List of BaseTool instances compatible with the provider

    Note:
        - ValidateSchemaTool is excluded for Groq due to tool calling format issues
        - Schema validation still happens via sanitize_schema_for_llamaextract()
    """
    # Core tools available for all providers
    tools = [
        ParseDocumentTool(),
        GenerateSchemaAITool(),
        GenerateSchemaTemplateTool(),
        ExtractDataTool(),
    ]

    # ValidateSchemaTool has compatibility issues with Groq's tool calling format
    # Skip it for Groq providers (validation still happens in sanitization layer)
    if provider.lower() != "groq":
        tools.append(ValidateSchemaTool())

    return tools


__all__ = [
    "ParseDocumentTool",
    "GenerateSchemaAITool",
    "GenerateSchemaTemplateTool",
    "ExtractDataTool",
    "ValidateSchemaTool",
    "get_all_tools",
]
