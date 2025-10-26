"""Schema validation tool."""

from langchain_core.tools import BaseTool
from pydantic import Field, BaseModel
from typing import Type, Dict, Any


class ValidateSchemaInput(BaseModel):
    """Input schema for validate_schema tool."""

    schema: Dict[str, Any] = Field(description="JSON schema to validate")


class ValidateSchemaTool(BaseTool):
    """Tool to validate JSON schema format."""

    name: str = "validate_schema"
    description: str = """Validate that a JSON schema is properly formatted according to JSON Schema Draft 7.
    Checks for required fields, proper structure, and valid types.

    Input: schema (JSON schema dictionary)
    Output: Validation result with success status and any error messages
    """
    args_schema: Type[BaseModel] = ValidateSchemaInput

    def _run(self, schema: dict) -> dict:
        """Validate JSON schema.

        Args:
            schema: JSON schema dictionary to validate

        Returns:
            Dict with 'valid' (bool) and 'errors' (list) keys
        """
        import jsonschema

        errors = []

        # Check basic structure
        if not isinstance(schema, dict):
            return {"valid": False, "errors": ["Schema must be a dictionary"]}

        if "type" not in schema:
            errors.append("Schema must have a 'type' field")

        if schema.get("type") == "object" and "properties" not in schema:
            errors.append("Object schema must have 'properties' field")

        # Validate against JSON Schema meta-schema
        try:
            # Use Draft 7 validator
            jsonschema.Draft7Validator.check_schema(schema)
        except jsonschema.exceptions.SchemaError as e:
            errors.append(f"Schema validation error: {str(e)}")

        return {"valid": len(errors) == 0, "errors": errors}

    async def _arun(self, schema: dict) -> dict:
        """Async version - calls sync version."""
        return self._run(schema)
