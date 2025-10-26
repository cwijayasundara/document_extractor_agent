"""Template-based schema generation tool."""

from langchain_core.tools import BaseTool
from pydantic import Field, BaseModel
from typing import Type


class GenerateSchemaTemplateInput(BaseModel):
    """Input schema for generate_schema_template tool."""

    template_path: str = Field(description="Absolute path to Excel template file (.xlsx)")


class GenerateSchemaTemplateTool(BaseTool):
    """Tool to generate JSON schema from Excel template."""

    name: str = "generate_schema_template"
    description: str = """Generate a JSON Schema from an Excel template file.
    The template should have field names in the first row.
    Fields prefixed with 'Item' are treated as line item fields (e.g., ItemQuantity, ItemPrice).

    Input: template_path (absolute path to .xlsx file)
    Output: JSON schema dictionary

    IMPORTANT: After calling this tool, the agent MUST STOP and wait for human approval
    before proceeding with extraction.
    """
    args_schema: Type[BaseModel] = GenerateSchemaTemplateInput

    def _run(self, template_path: str) -> dict:
        """Generate schema from Excel template.

        Args:
            template_path: Path to Excel template file

        Returns:
            JSON schema dictionary
        """
        from src.plain.template_schema_generator import generate_schema_from_template

        schema = generate_schema_from_template(template_path)
        return schema

    async def _arun(self, template_path: str) -> dict:
        """Async version - calls sync version."""
        return self._run(template_path)
