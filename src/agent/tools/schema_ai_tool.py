"""AI-based schema generation tool using OpenAI."""

from langchain_core.tools import BaseTool
from pydantic import Field, BaseModel
from typing import Type
import json


class GenerateSchemaAIInput(BaseModel):
    """Input schema for generate_schema_ai tool."""

    document_text: str = Field(description="Parsed document text in markdown format")
    prompt: str = Field(description="User's extraction prompt describing what to extract")


class GenerateSchemaAITool(BaseTool):
    """Tool to generate JSON schema from document using AI analysis."""

    name: str = "generate_schema_ai"
    description: str = """Generate a JSON Schema Draft 7 for extracting structured data from a document.
    Uses AI to analyze the document content and create an appropriate extraction schema.

    Input:
    - document_text: The parsed document content
    - prompt: User's instructions for what to extract

    Output: JSON schema dictionary with type, properties, etc.

    IMPORTANT: After calling this tool, the agent MUST STOP and wait for human approval
    before proceeding with extraction.
    """
    args_schema: Type[BaseModel] = GenerateSchemaAIInput

    async def _arun(self, document_text: str, prompt: str) -> dict:
        """Generate schema using configured LLM (OpenAI or Groq).

        Args:
            document_text: Parsed markdown from document
            prompt: User's extraction requirements

        Returns:
            JSON schema dictionary
        """
        from src.plain.config import get_llm_client, get_model_config, LLM_PROVIDER, LLM_MODEL

        # Use configured LLM provider and model
        llm_client = await get_llm_client(provider=LLM_PROVIDER, model=LLM_MODEL)
        model_config = get_model_config(LLM_MODEL)

        # System prompt for schema generation
        system_prompt = """You are an expert at creating JSON Schema Draft 7 specifications for data extraction.

Generate a JSON schema that captures the structure needed to extract data from the provided document.

Requirements:
- Use JSON Schema Draft 7 format
- Include "type": "object" at the root
- Define all fields in "properties"
- Add helpful "description" for each field
- Identify repeating data (line items, etc.) as arrays of objects
- Use appropriate data types: string, number, boolean, array, object
- For arrays of objects, define the item properties in "items"

CRITICAL - Nested Objects:
- NEVER create nested objects with empty "properties": {}
- Every object-type field MUST have at least one property defined in its "properties"
- If you cannot identify specific sub-fields for a nested object, use "type": "string" instead
- Limit nesting depth to 3-4 levels maximum for best extraction quality

Return ONLY the JSON schema, no explanations."""

        response = await llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Document:\n{document_text[:4000]}\n\nExtraction Requirements: {prompt}",
                },
            ],
            response_format={"type": "json_object"},
            temperature=model_config.get("temperature", 1.0),
        )

        schema_str = response.choices[0].message.content
        schema = json.loads(schema_str)

        return schema

    def _run(self, document_text: str, prompt: str) -> dict:
        """Sync wrapper for async implementation."""
        import asyncio

        return asyncio.run(self._arun(document_text, prompt))
