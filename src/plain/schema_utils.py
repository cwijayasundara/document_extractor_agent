"""Schema validation and sanitization utilities for LlamaExtract compatibility.

This module provides utilities to ensure JSON schemas are compatible with
LlamaExtract API requirements, particularly for nested objects.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def sanitize_schema_for_llamaextract(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize a JSON schema to ensure LlamaExtract API compatibility.

    LlamaExtract requires:
    - Nested objects must have non-empty 'properties' dictionaries
    - Maximum nesting depth of 7 levels
    - Valid JSON Schema Draft 7 structure

    This function:
    - Removes nested objects with empty or missing properties
    - Converts objects with only invalid properties to strings
    - Recursively cleans all nested structures

    Args:
        schema: JSON schema dictionary to sanitize

    Returns:
        Sanitized schema dictionary safe for LlamaExtract

    Example:
        >>> schema = {
        ...     "type": "object",
        ...     "properties": {
        ...         "company": {
        ...             "type": "object",
        ...             "properties": {}  # Empty - will be removed
        ...         },
        ...         "name": {"type": "string"}
        ...     }
        ... }
        >>> sanitized = sanitize_schema_for_llamaextract(schema)
        # company field removed, only name remains
    """
    if not isinstance(schema, dict):
        return schema

    # Clone schema to avoid mutating input
    sanitized = dict(schema)

    # Process properties if this is an object schema
    if sanitized.get("type") == "object" and "properties" in sanitized:
        sanitized["properties"] = _sanitize_properties(sanitized["properties"])

        # If all properties were removed, convert to empty object warning
        if not sanitized["properties"]:
            logger.warning(
                "Schema object has no valid properties after sanitization. "
                "This may indicate overly nested or empty schema structure."
            )

    # Process array items if this is an array schema
    if sanitized.get("type") == "array" and "items" in sanitized:
        sanitized["items"] = sanitize_schema_for_llamaextract(sanitized["items"])

    # Clean up required fields that no longer exist
    if "required" in sanitized and "properties" in sanitized:
        valid_fields = set(sanitized["properties"].keys())
        sanitized["required"] = [
            field for field in sanitized.get("required", [])
            if field in valid_fields
        ]
        # Remove empty required list
        if not sanitized["required"]:
            del sanitized["required"]

    return sanitized


def _sanitize_properties(properties: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize properties dictionary, removing invalid nested objects.

    Args:
        properties: Dictionary of property name -> property schema

    Returns:
        Cleaned properties dictionary
    """
    if not isinstance(properties, dict):
        return {}

    sanitized_props = {}

    for prop_name, prop_schema in properties.items():
        if not isinstance(prop_schema, dict):
            # Keep primitive schemas as-is
            sanitized_props[prop_name] = prop_schema
            continue

        prop_type = prop_schema.get("type")

        # Handle object properties
        if prop_type == "object":
            nested_props = prop_schema.get("properties")

            # Skip objects with empty or missing properties
            if not nested_props or not isinstance(nested_props, dict):
                logger.warning(
                    f"Removing nested object '{prop_name}' with empty/invalid properties"
                )
                continue

            # Recursively sanitize nested properties
            sanitized_nested = _sanitize_properties(nested_props)

            # Skip if all nested properties were removed
            if not sanitized_nested:
                logger.warning(
                    f"Removing nested object '{prop_name}' - all nested properties invalid"
                )
                continue

            # Keep the object with sanitized properties
            sanitized_props[prop_name] = {
                **prop_schema,
                "properties": sanitized_nested
            }

        # Handle array properties
        elif prop_type == "array":
            items_schema = prop_schema.get("items")

            if items_schema and isinstance(items_schema, dict):
                # Recursively sanitize array items
                sanitized_items = sanitize_schema_for_llamaextract(items_schema)

                # Only keep array if items are valid
                if sanitized_items:
                    sanitized_props[prop_name] = {
                        **prop_schema,
                        "items": sanitized_items
                    }
                else:
                    logger.warning(f"Removing array '{prop_name}' with invalid items")
            else:
                # Keep arrays without items schema (though unusual)
                sanitized_props[prop_name] = prop_schema

        else:
            # Keep other property types (string, number, boolean, etc.)
            sanitized_props[prop_name] = prop_schema

    return sanitized_props


def validate_llamaextract_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Validate schema against LlamaExtract requirements.

    Args:
        schema: JSON schema to validate

    Returns:
        Dictionary with:
        - valid (bool): Whether schema is valid for LlamaExtract
        - errors (List[str]): List of validation errors
        - warnings (List[str]): List of warnings

    Example:
        >>> result = validate_llamaextract_schema(schema)
        >>> if not result['valid']:
        ...     print(f"Errors: {result['errors']}")
    """
    errors: List[str] = []
    warnings: List[str] = []

    # Check basic structure
    if not isinstance(schema, dict):
        errors.append("Schema must be a dictionary")
        return {"valid": False, "errors": errors, "warnings": warnings}

    # Check root type
    if schema.get("type") != "object":
        errors.append("Root schema type must be 'object'")

    # Check for properties
    if "properties" not in schema:
        errors.append("Root schema must have 'properties' field")
    elif not isinstance(schema["properties"], dict):
        errors.append("Schema 'properties' must be a dictionary")
    elif not schema["properties"]:
        warnings.append("Schema has empty 'properties' - no fields will be extracted")

    # Check nesting depth
    max_depth = _get_max_nesting_depth(schema)
    if max_depth > 7:
        errors.append(f"Schema nesting depth ({max_depth}) exceeds LlamaExtract limit of 7")
    elif max_depth > 4:
        warnings.append(
            f"Schema nesting depth ({max_depth}) is high. "
            "Consider simplifying for better extraction quality."
        )

    # Recursively check for empty nested objects
    empty_objects = _find_empty_nested_objects(schema)
    if empty_objects:
        errors.append(
            f"Schema contains nested objects with empty properties: {', '.join(empty_objects)}"
        )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }


def _get_max_nesting_depth(schema: Dict[str, Any], current_depth: int = 0) -> int:
    """Calculate maximum nesting depth in schema.

    Args:
        schema: Schema (or sub-schema) to analyze
        current_depth: Current depth level

    Returns:
        Maximum nesting depth found
    """
    if not isinstance(schema, dict):
        return current_depth

    max_depth = current_depth

    # Check properties for nested objects
    if "properties" in schema and isinstance(schema["properties"], dict):
        for prop_schema in schema["properties"].values():
            if isinstance(prop_schema, dict):
                depth = _get_max_nesting_depth(prop_schema, current_depth + 1)
                max_depth = max(max_depth, depth)

    # Check array items
    if "items" in schema and isinstance(schema["items"], dict):
        depth = _get_max_nesting_depth(schema["items"], current_depth + 1)
        max_depth = max(max_depth, depth)

    return max_depth


def _find_empty_nested_objects(schema: Dict[str, Any], path: str = "root") -> List[str]:
    """Find nested objects with empty properties.

    Args:
        schema: Schema to check
        path: Current path in schema (for error reporting)

    Returns:
        List of paths to empty nested objects
    """
    if not isinstance(schema, dict):
        return []

    empty_objects = []

    # Check if this is an object with empty properties
    if schema.get("type") == "object":
        props = schema.get("properties")
        if props is not None and not props:
            empty_objects.append(path)
        elif isinstance(props, dict):
            # Recursively check nested properties
            for prop_name, prop_schema in props.items():
                if isinstance(prop_schema, dict):
                    nested_empty = _find_empty_nested_objects(
                        prop_schema,
                        f"{path}.{prop_name}"
                    )
                    empty_objects.extend(nested_empty)

    # Check array items
    if "items" in schema and isinstance(schema["items"], dict):
        nested_empty = _find_empty_nested_objects(
            schema["items"],
            f"{path}[]"
        )
        empty_objects.extend(nested_empty)

    return empty_objects
