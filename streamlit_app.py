"""Streamlit UI for Iterative Extraction Workflow.

This application provides a web interface for:
- Uploading PDF/image documents
- Auto mode: AI-generated schema with approval workflow
- Manual mode: Template-based schema (coming soon)
- Visual schema display
- Data extraction and display
"""

import streamlit as st
import asyncio
import json
import re
import tempfile
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor
import threading

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.plain.extraction_workflow import (
    IterativeExtractionWorkflow,
    ProgressEvent,
    ProposedSchema,
    ApprovedSchema,
)
from src.plain.excel_export import export_to_excel


def parse_schema_structure(schema: dict) -> dict:
    """Parse JSON schema to extract scalars, objects, and arrays.

    Categorizes schema properties by type for dynamic UI rendering.

    Args:
        schema: JSON schema dictionary

    Returns:
        Dictionary with 'scalars', 'objects', and 'arrays' keys
    """
    structure = {
        "scalars": [],      # Simple fields (string, number, boolean, etc.)
        "objects": {},      # Nested objects {field_name: {description, fields}}
        "arrays": {},       # Arrays {field_name: {description, fields}}
    }

    if "properties" not in schema:
        return structure

    def extract_object_fields(properties: dict) -> list:
        """Extract fields from an object's properties.

        Preserves full schema for arrays and nested objects to avoid losing structure.
        """
        fields = []
        for field_name, field_info in properties.items():
            field_type = field_info.get("type", "unknown")
            description = field_info.get("description", "")

            # Convert type to string
            if isinstance(field_type, (list, dict)):
                field_type_str = json.dumps(field_type)
            else:
                field_type_str = str(field_type)

            # Build field entry
            field_entry = {
                "field": str(field_name),
                "type": field_type_str,
                "description": str(description)
            }

            # Preserve items schema for arrays (nested arrays within objects)
            if field_type == "array" and "items" in field_info:
                field_entry["items"] = field_info["items"]

            # Preserve properties schema for nested objects (objects within objects)
            if field_type == "object" and "properties" in field_info:
                field_entry["properties"] = field_info["properties"]

            fields.append(field_entry)
        return fields

    # Categorize all top-level properties
    for field_name, field_info in schema["properties"].items():
        field_type = field_info.get("type", "unknown")
        description = field_info.get("description", "")

        # Convert type to string
        if isinstance(field_type, (list, dict)):
            field_type_str = json.dumps(field_type)
        else:
            field_type_str = str(field_type)

        if field_type == "object" and "properties" in field_info:
            # Nested object
            structure["objects"][field_name] = {
                "description": description,
                "fields": extract_object_fields(field_info["properties"])
            }
        elif field_type == "array":
            # Array field
            items_schema = field_info.get("items", {})

            if items_schema.get("type") == "object" and "properties" in items_schema:
                # Array of objects - extract item properties
                structure["arrays"][field_name] = {
                    "description": description or f"Array of {field_name}",
                    "fields": extract_object_fields(items_schema["properties"])
                }
            else:
                # Array of primitives - treat as scalar with special type
                items_type = items_schema.get('type', 'items')
                if isinstance(items_type, (list, dict)):
                    items_type = json.dumps(items_type)
                structure["scalars"].append({
                    "field": str(field_name),
                    "type": f"array of {items_type}",
                    "description": str(description)
                })
        else:
            # Scalar field (string, number, boolean, etc.)
            structure["scalars"].append({
                "field": str(field_name),
                "type": field_type_str,
                "description": str(description)
            })

    return structure


def reconstruct_schema_from_structure(
    scalars: list,
    objects: dict,
    arrays: dict,
) -> dict:
    """Reconstruct JSON schema from edited scalars, objects, and arrays.

    Args:
        scalars: List of scalar field dictionaries (field, type, description)
        objects: Dict of nested objects {field_name: {description, fields}}
        arrays: Dict of arrays {field_name: {description, fields}}

    Returns:
        Reconstructed JSON schema dictionary
    """
    schema = {
        "type": "object",
        "properties": {}
    }

    def parse_type(type_str: str):
        """Parse type string back to appropriate format."""
        try:
            if type_str.startswith("[") or type_str.startswith("{"):
                return json.loads(type_str)
            elif type_str.startswith("array of "):
                items_type = type_str.replace("array of ", "").strip()
                if items_type.startswith("[") or items_type.startswith("{"):
                    items_type = json.loads(items_type)
                return {"type": "array", "items": {"type": items_type}}
            else:
                return type_str
        except:
            return type_str

    # Add scalar fields
    for field in scalars:
        field_name = field.get("field", "")
        field_type_str = field.get("type", "string")
        description = field.get("description", "")

        if not field_name:
            continue

        parsed_type = parse_type(field_type_str)

        # Handle special case of array type returned as dict
        if isinstance(parsed_type, dict) and "type" in parsed_type:
            schema["properties"][field_name] = {
                **parsed_type,
                "description": description
            }
        else:
            schema["properties"][field_name] = {
                "type": parsed_type,
                "description": description
            }

    # Add nested objects
    for obj_name, obj_data in objects.items():
        obj_properties = {}

        for field in obj_data.get("fields", []):
            field_name = field.get("field", "")
            field_type_str = field.get("type", "string")
            field_description = field.get("description", "")

            if not field_name:
                continue

            parsed_type = parse_type(field_type_str)

            # Build base field schema
            if isinstance(parsed_type, dict) and "type" in parsed_type:
                field_schema = {
                    **parsed_type,
                    "description": field_description
                }
            else:
                field_schema = {
                    "type": parsed_type,
                    "description": field_description
                }

            # Restore preserved items for arrays (nested arrays within objects)
            if "items" in field:
                field_schema["items"] = field["items"]

            # Restore preserved properties for nested objects (objects within objects)
            if "properties" in field:
                field_schema["properties"] = field["properties"]

            obj_properties[field_name] = field_schema

        # Only add object if it has at least one property (avoid empty nested objects)
        if obj_properties:
            schema["properties"][obj_name] = {
                "type": "object",
                "properties": obj_properties,
                "description": obj_data.get("description", f"Nested object: {obj_name}")
            }

    # Add arrays
    for array_name, array_data in arrays.items():
        array_item_properties = {}

        for field in array_data.get("fields", []):
            field_name = field.get("field", "")
            field_type_str = field.get("type", "string")
            field_description = field.get("description", "")

            if not field_name:
                continue

            parsed_type = parse_type(field_type_str)

            # Build base field schema
            if isinstance(parsed_type, dict) and "type" in parsed_type:
                field_schema = {
                    **parsed_type,
                    "description": field_description
                }
            else:
                field_schema = {
                    "type": parsed_type,
                    "description": field_description
                }

            # Restore preserved items for nested arrays (arrays within array items)
            if "items" in field:
                field_schema["items"] = field["items"]

            # Restore preserved properties for nested objects (objects within array items)
            if "properties" in field:
                field_schema["properties"] = field["properties"]

            array_item_properties[field_name] = field_schema

        # Only add array if it has at least one item property (avoid arrays with empty objects)
        if array_item_properties:
            schema["properties"][array_name] = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": array_item_properties
                },
                "description": array_data.get("description", f"Array of {array_name}")
            }

    return schema


def display_schema_visual(schema: dict):
    """Display schema in a visual, user-friendly format with editable fields.

    Dynamically creates sections based on schema structure (scalars, objects, arrays).

    Args:
        schema: JSON schema dictionary
    """
    st.subheader("📋 Proposed Schema Structure")
    st.info("💡 Tip: You can edit the fields below to customize the schema before extraction!")

    structure = parse_schema_structure(schema)
    schema_signature = json.dumps(schema, sort_keys=True)
    previous_signature = st.session_state.get("current_schema_signature")

    # Initialize session state for edited fields if schema changed
    if previous_signature != schema_signature:
        st.session_state.current_schema_signature = schema_signature
        st.session_state.edited_scalars = [dict(item) for item in structure["scalars"]]
        st.session_state.edited_objects = {
            obj_name: [dict(field) for field in obj_data["fields"]]
            for obj_name, obj_data in structure["objects"].items()
        }
        st.session_state.edited_arrays = {
            arr_name: [dict(field) for field in arr_data["fields"]]
            for arr_name, arr_data in structure["arrays"].items()
        }

    # Auto-expand debug if parsing found nothing
    total_fields = len(structure["scalars"]) + len(structure["objects"]) + len(structure["arrays"])
    auto_expand_debug = total_fields == 0

    # Debug: Show raw schema and parsed structure
    with st.expander("🔍 Debug: Raw Schema & Parsed Structure", expanded=auto_expand_debug):
        st.write("**Raw Schema:**")
        st.json(schema)
        st.write("**Parsed Structure:**")
        st.write(f"Scalar fields: {len(structure['scalars'])}")
        st.write(f"Nested objects: {len(structure['objects'])}")
        st.write(f"Arrays: {len(structure['arrays'])}")
        st.json(structure)

        # Show field categorization
        st.markdown("---")
        st.write("**Field Categorization:**")
        if "properties" in schema:
            for field_name, field_info in schema["properties"].items():
                field_type = field_info.get("type", "unknown")

                if field_name in [f["field"] for f in structure["scalars"]]:
                    category = "📊 Scalar"
                elif field_name in structure["objects"]:
                    category = "🔷 Nested Object"
                    obj_field_count = len(structure["objects"][field_name]["fields"])
                    st.text(f"• {field_name}: {field_type} → {category} ({obj_field_count} fields)")
                    continue
                elif field_name in structure["arrays"]:
                    category = "📋 Array"
                    arr_field_count = len(structure["arrays"][field_name]["fields"])
                    st.text(f"• {field_name}: {field_type} → {category} ({arr_field_count} item fields)")
                    continue
                else:
                    category = "❓ Unknown"

                st.text(f"• {field_name}: {field_type} → {category}")

    def format_field_name(name: str) -> str:
        """Format field name for display (e.g., 'line_items' → 'Line Items')."""
        formatted = name.replace("_", " ").replace("-", " ")
        formatted = re.sub(r"(?<!^)(?=[A-Z])", " ", formatted)
        return formatted.title()

    # Display scalar fields
    if structure["scalars"]:
        st.markdown("### 📊 Scalar Fields")
        st.caption("Simple fields (strings, numbers, booleans, etc.). Edit as needed.")

        scalar_editor_state = st.session_state.get("edited_scalars")
        scalar_editor_source = scalar_editor_state if scalar_editor_state is not None else structure["scalars"]
        if hasattr(scalar_editor_source, 'to_dict'):
            scalar_editor_source = scalar_editor_source.to_dict('records')

        edited_scalars = st.data_editor(
            scalar_editor_source,
            width='stretch',
            hide_index=True,
            num_rows="dynamic",
            column_config={
                "field": st.column_config.TextColumn("Field Name", required=True, width="medium"),
                "type": st.column_config.TextColumn("Type", required=True, width="small"),
                "description": st.column_config.TextColumn("Description", width="large")
            }
        )

        if hasattr(edited_scalars, 'to_dict'):
            st.session_state.edited_scalars = edited_scalars.to_dict('records')
        else:
            st.session_state.edited_scalars = edited_scalars
    else:
        st.session_state.edited_scalars = []

    # Display nested objects
    if structure["objects"]:
        st.markdown("### 🔷 Nested Objects")
        st.caption("Complex nested data structures. Each object is editable below.")

        # Initialize edited_objects if not exists
        if "edited_objects" not in st.session_state or st.session_state.edited_objects is None:
            st.session_state.edited_objects = {}

        for obj_name, obj_data in structure["objects"].items():
            with st.expander(f"🔷 {format_field_name(obj_name)}", expanded=True):
                st.caption(obj_data.get("description", f"Fields within {obj_name}"))

                obj_editor_state = st.session_state.edited_objects.get(obj_name)
                obj_editor_source = obj_editor_state if obj_editor_state is not None else obj_data["fields"]
                if hasattr(obj_editor_source, 'to_dict'):
                    obj_editor_source = obj_editor_source.to_dict('records')

                edited_obj_fields = st.data_editor(
                    obj_editor_source,
                    width='stretch',
                    hide_index=True,
                    num_rows="dynamic",
                    key=f"obj_editor_{obj_name}",
                    column_config={
                        "field": st.column_config.TextColumn("Field Name", required=True, width="medium"),
                        "type": st.column_config.TextColumn("Type", required=True, width="small"),
                        "description": st.column_config.TextColumn("Description", width="large")
                    }
                )

                if hasattr(edited_obj_fields, 'to_dict'):
                    st.session_state.edited_objects[obj_name] = edited_obj_fields.to_dict('records')
                else:
                    st.session_state.edited_objects[obj_name] = edited_obj_fields
    else:
        st.session_state.edited_objects = {}

    # Display arrays
    if structure["arrays"]:
        st.markdown("### 📋 Repeating Data (Arrays)")
        st.caption("Arrays of items. Each array's item structure is editable below.")

        # Initialize edited_arrays if not exists
        if "edited_arrays" not in st.session_state or st.session_state.edited_arrays is None:
            st.session_state.edited_arrays = {}

        for arr_name, arr_data in structure["arrays"].items():
            with st.expander(f"📋 {format_field_name(arr_name)}", expanded=True):
                st.caption(arr_data.get("description", f"Each item in {arr_name} contains these fields:"))

                arr_editor_state = st.session_state.edited_arrays.get(arr_name)
                arr_editor_source = arr_editor_state if arr_editor_state is not None else arr_data["fields"]
                if hasattr(arr_editor_source, 'to_dict'):
                    arr_editor_source = arr_editor_source.to_dict('records')

                edited_arr_fields = st.data_editor(
                    arr_editor_source,
                    width='stretch',
                    hide_index=True,
                    num_rows="dynamic",
                    key=f"arr_editor_{arr_name}",
                    column_config={
                        "field": st.column_config.TextColumn("Field Name", required=True, width="medium"),
                        "type": st.column_config.TextColumn("Type", required=True, width="small"),
                        "description": st.column_config.TextColumn("Description", width="large")
                    }
                )

                if hasattr(edited_arr_fields, 'to_dict'):
                    st.session_state.edited_arrays[arr_name] = edited_arr_fields.to_dict('records')
                else:
                    st.session_state.edited_arrays[arr_name] = edited_arr_fields
    else:
        st.session_state.edited_arrays = {}

    # Fallback if no structure was parsed
    if total_fields == 0:
        st.warning("⚠️ Could not parse schema structure automatically. Please check the raw schema in the debug section above.")
        st.info("💡 The schema might have an unexpected format. You can still proceed with extraction using the proposed schema, or reject and provide feedback to regenerate it.")

    # Show reconstructed schema preview
    with st.expander("🔍 Preview Final Schema (After Edits)"):
        edited_scalars = st.session_state.edited_scalars or []
        edited_objects = st.session_state.edited_objects or {}
        edited_arrays = st.session_state.edited_arrays or {}

        # Convert to list if DataFrame
        if hasattr(edited_scalars, 'to_dict'):
            edited_scalars = edited_scalars.to_dict('records')

        edited_objects_clean = {}
        for obj_name, obj_fields in edited_objects.items():
            if hasattr(obj_fields, 'to_dict'):
                edited_objects_clean[obj_name] = {"fields": obj_fields.to_dict('records'), "description": structure["objects"].get(obj_name, {}).get("description", "")}
            else:
                edited_objects_clean[obj_name] = {"fields": obj_fields, "description": structure["objects"].get(obj_name, {}).get("description", "")}

        edited_arrays_clean = {}
        for arr_name, arr_fields in edited_arrays.items():
            if hasattr(arr_fields, 'to_dict'):
                edited_arrays_clean[arr_name] = {"fields": arr_fields.to_dict('records'), "description": structure["arrays"].get(arr_name, {}).get("description", "")}
            else:
                edited_arrays_clean[arr_name] = {"fields": arr_fields, "description": structure["arrays"].get(arr_name, {}).get("description", "")}

        if edited_scalars or edited_objects_clean or edited_arrays_clean:
            reconstructed = reconstruct_schema_from_structure(
                edited_scalars,
                edited_objects_clean,
                edited_arrays_clean
            )
            st.json(reconstructed)
        else:
            st.json(schema)

    # Raw JSON schema in expander
    with st.expander("📄 View Original JSON Schema"):
        st.json(schema)


def flatten_dict(d: dict, parent_key: str = '', sep: str = ' → ') -> dict:
    """Flatten a nested dictionary (and lists) for display.

    Args:
        d: Dictionary to flatten
        parent_key: Parent key for nested items
        sep: Separator between nested keys

    Returns:
        Flattened dictionary with concatenated keys
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            if not v:
                items.append((new_key, []))
            elif all(isinstance(item, dict) for item in v):
                for idx, item in enumerate(v, start=1):
                    list_key = f"{new_key}[{idx}]"
                    items.extend(flatten_dict(item, list_key, sep=sep).items())
            else:
                items.append((new_key, v))
        else:
            items.append((new_key, v))
    return dict(items)


def display_extracted_data(data: dict):
    """Display extracted data in a structured format.

    Dynamically categorizes and displays data based on its structure (scalars, objects, arrays).

    Args:
        data: Extracted data dictionary
    """
    st.subheader("✅ Extracted Data")

    # Debug section - show raw data
    with st.expander("🔍 Debug: Raw Extracted Data", expanded=False):
        st.json(data)

    def format_label(label: str) -> str:
        """Format field name for display."""
        def prettify(segment: str) -> str:
            raw = segment.replace("_", " ").strip()
            if not raw:
                return segment
            spaced = re.sub(r"(?<!^)(?=[A-Z])", " ", raw)
            spaced = re.sub(r"\s+", " ", spaced).strip()
            return spaced[0].upper() + spaced[1:] if spaced else segment

        parts = []
        for segment in label.split(" → "):
            if "[" in segment:
                base, _, rest = segment.partition("[")
                formatted_base = prettify(base)
                formatted = f"{formatted_base or base}{'[' + rest if rest else ''}"
                parts.append(formatted)
            else:
                parts.append(prettify(segment))
        return " → ".join(parts)

    def format_value(val):
        """Format a value for display. Always returns a string to avoid Arrow serialization issues."""
        if val is None:
            return "—"
        if isinstance(val, str):
            return val if val.strip() else "—"
        if isinstance(val, (int, float)):
            return str(val)
        if isinstance(val, list):
            if not val:
                return "—"
            return ", ".join(str(item) for item in val)
        return str(val)

    # Categorize data by type
    scalars = {}
    objects = {}
    arrays = {}

    for key, value in data.items():
        if isinstance(value, dict):
            objects[key] = value
        elif isinstance(value, list):
            arrays[key] = value
        else:
            scalars[key] = value

    # Display scalar fields
    if scalars:
        st.markdown("### 📊 Scalar Fields")

        scalar_count = len(scalars)
        if scalar_count == 1:
            cols = st.columns(1)
        elif scalar_count <= 4:
            cols = st.columns(2)
        elif scalar_count <= 9:
            cols = st.columns(3)
        else:
            cols = st.columns(4)

        for idx, (key, value) in enumerate(scalars.items()):
            col_idx = idx % len(cols)
            with cols[col_idx]:
                st.metric(label=format_label(key), value=format_value(value))

    # Display nested objects
    if objects:
        st.markdown("### 🔷 Nested Objects")

        for obj_key, obj_value in objects.items():
            with st.expander(f"🔷 {format_label(obj_key)}", expanded=True):
                section_rows = []

                def collect_fields(node: dict, prefix: str = ""):
                    for sub_key, sub_value in node.items():
                        path = f"{prefix} → {sub_key}" if prefix else sub_key
                        if isinstance(sub_value, dict):
                            collect_fields(sub_value, path)
                        elif isinstance(sub_value, list):
                            if not sub_value:
                                section_rows.append({"Field": format_label(path), "Value": "—"})
                            elif all(isinstance(item, dict) for item in sub_value):
                                for idx, item in enumerate(sub_value, start=1):
                                    collect_fields(item, f"{path}[{idx}]")
                            else:
                                display = ", ".join(str(item) for item in sub_value if item not in (None, ""))
                                section_rows.append({
                                    "Field": format_label(path),
                                    "Value": display or "—"
                                })
                        else:
                            section_rows.append({
                                "Field": format_label(path),
                                "Value": format_value(sub_value)
                            })

                collect_fields(obj_value)

                if section_rows:
                    st.dataframe(section_rows, width='stretch', hide_index=True)
                else:
                    st.info("No populated fields detected in this section.")

    # Display arrays
    if arrays:
        st.markdown("### 📋 Repeating Data (Arrays)")

        for arr_key, arr_value in arrays.items():
            st.markdown(f"#### 📋 {format_label(arr_key)}")

            if not arr_value:
                st.warning(f"No items in {arr_key}")
                continue

            st.caption(f"Total items: {len(arr_value)}")

            # Convert all values to strings to avoid Arrow serialization issues
            def stringify_row(row):
                """Convert all values in a row to strings for Arrow compatibility."""
                if isinstance(row, dict):
                    return {k: str(v) if v is not None else "—" for k, v in row.items()}
                elif isinstance(row, (list, tuple)):
                    return ", ".join(str(item) for item in row)
                else:
                    return str(row)

            try:
                # Check if array contains objects or primitives
                if all(isinstance(item, dict) for item in arr_value):
                    # Array of objects - display as table
                    stringified_data = [stringify_row(row) for row in arr_value]
                    st.dataframe(
                        stringified_data,
                        width='stretch',
                        hide_index=False
                    )
                else:
                    # Array of primitives - display as list
                    primitive_display = [{"Index": i+1, "Value": format_value(item)} for i, item in enumerate(arr_value)]
                    st.dataframe(
                        primitive_display,
                        width='stretch',
                        hide_index=True
                    )
            except Exception as e:
                st.warning(f"Could not display as interactive table: {str(e)}")
                st.table(arr_value)

    # No data message
    if not scalars and not objects and not arrays:
        st.info("No data found in extraction results.")

    # Summary section
    st.markdown("---")
    summary_cols = st.columns(4)

    # Calculate counts
    total_array_items = sum(len(arr) if isinstance(arr, list) else 0 for arr in arrays.values())

    with summary_cols[0]:
        st.metric("📊 Scalar Fields", len(scalars))
    with summary_cols[1]:
        st.metric("🔷 Nested Objects", len(objects))
    with summary_cols[2]:
        st.metric("📋 Array Fields", len(arrays))
    with summary_cols[3]:
        st.metric("📦 Total Array Items", total_array_items)

    # Download buttons
    st.markdown("---")
    st.markdown("### 💾 Export Data")
    download_col1, download_col2 = st.columns(2)

    with download_col1:
        st.download_button(
            label="📄 Download JSON",
            data=json.dumps(data, indent=2),
            file_name="extracted_data.json",
            mime="application/json",
            width='stretch'
        )

    with download_col2:
        # Generate Excel file
        excel_buffer = export_to_excel(data)
        st.download_button(
            label="📊 Download Excel",
            data=excel_buffer,
            file_name="extracted_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width='stretch'
        )


async def run_workflow(
    file_path: str,
    prompt: str,
    auto_approve: bool = False,
    template_path: str | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None
):
    """Run the extraction workflow and handle events.

    Args:
        file_path: Path to the document file
        prompt: Extraction prompt
        auto_approve: If True, auto-approve schema (for testing)
        template_path: Optional path to Excel template for Manual mode
        llm_provider: LLM provider ("openai" or "groq")
        llm_model: LLM model identifier

    Returns:
        Tuple of (schema, extracted_data, agent_id) or (None, None, None) if waiting for approval
    """
    wf = IterativeExtractionWorkflow(timeout=None)

    handler = wf.run(
        file_path=file_path,
        prompt=prompt,
        template_path=template_path,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )

    proposed_schema = None

    # Stream events
    async for ev in handler.stream_events():
        if isinstance(ev, ProgressEvent):
            st.session_state.progress_messages.append(ev.msg)
        elif isinstance(ev, ProposedSchema):
            proposed_schema = ev.generated_schema
            st.session_state.proposed_schema = proposed_schema
            st.session_state.workflow_handler = handler
            st.session_state.workflow_state = "schema_approval"
            return proposed_schema, None, None

    # Workflow completed
    result = await handler
    return proposed_schema, result.data, result.agent_id


def init_session_state():
    """Initialize Streamlit session state variables."""
    if "workflow_state" not in st.session_state:
        st.session_state.workflow_state = "upload"  # upload, parsing, schema_approval, extracting, complete
    if "progress_messages" not in st.session_state:
        st.session_state.progress_messages = []
    if "proposed_schema" not in st.session_state:
        st.session_state.proposed_schema = None
    if "extracted_data" not in st.session_state:
        st.session_state.extracted_data = None
    if "agent_id" not in st.session_state:
        st.session_state.agent_id = None
    if "workflow_handler" not in st.session_state:
        st.session_state.workflow_handler = None
    if "temp_file_path" not in st.session_state:
        st.session_state.temp_file_path = None
    if "edited_scalars" not in st.session_state:
        st.session_state.edited_scalars = None
    if "edited_objects" not in st.session_state:
        st.session_state.edited_objects = None
    if "edited_arrays" not in st.session_state:
        st.session_state.edited_arrays = None
    if "final_schema" not in st.session_state:
        st.session_state.final_schema = None
    if "extraction_prompt" not in st.session_state:
        st.session_state.extraction_prompt = "Extract all the important information from the invoice."
    if "original_filename" not in st.session_state:
        st.session_state.original_filename = None
    if "current_schema_signature" not in st.session_state:
        st.session_state.current_schema_signature = None
    if "previous_mode" not in st.session_state:
        st.session_state.previous_mode = None
    # LLM configuration
    if "llm_provider" not in st.session_state:
        import os
        st.session_state.llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if "llm_model" not in st.session_state:
        import os
        st.session_state.llm_model = os.getenv("LLM_MODEL", "gpt-5-nano")


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Document Extraction Workflow",
        page_icon="📄",
        layout="wide"
    )

    init_session_state()

    st.title("📄 Document Extraction Workflow")
    st.markdown("---")

    # Mode selection
    col1, col2 = st.columns([1, 3])
    with col1:
        mode = st.radio(
            "Select Mode:",
            ["Auto", "Manual"],
            help="Auto: AI generates schema automatically\nManual: Upload your own template"
        )

    # Detect mode change and reset state
    if st.session_state.previous_mode is not None and st.session_state.previous_mode != mode:
        # Mode changed - reset all workflow state
        st.session_state.workflow_state = "upload"
        st.session_state.progress_messages = []
        st.session_state.proposed_schema = None
        st.session_state.extracted_data = None
        st.session_state.agent_id = None
        st.session_state.workflow_handler = None
        st.session_state.temp_file_path = None
        st.session_state.edited_scalars = None
        st.session_state.edited_objects = None
        st.session_state.edited_arrays = None
        st.session_state.final_schema = None
        st.session_state.original_filename = None
        st.session_state.current_schema_signature = None
        st.session_state.previous_mode = mode
        st.rerun()

    # Update previous mode
    st.session_state.previous_mode = mode

    with col2:
        if mode == "Manual":
            st.info("📋 Manual mode: Upload an Excel template to define the extraction schema. Use 'Item' prefix for line item fields (e.g., ItemQuantity, ItemPrice).")

    # LLM Provider and Model Selection
    st.markdown("### 🤖 LLM Configuration")
    llm_col1, llm_col2 = st.columns(2)

    with llm_col1:
        llm_provider = st.selectbox(
            "LLM Provider:",
            ["OpenAI", "Groq"],
            index=0 if st.session_state.get("llm_provider", "openai") == "openai" else 1,
            help="Select the AI provider for schema generation"
        )

    # Update session state
    st.session_state.llm_provider = llm_provider.lower()

    with llm_col2:
        if llm_provider == "OpenAI":
            available_models = ["gpt-5-nano", "gpt-4o", "gpt-4o-mini"]
            default_model = "gpt-5-nano"
        else:  # Groq
            available_models = [
                "moonshotai/kimi-k2-instruct-0905",
                "llama-3.3-70b-versatile",
                "mixtral-8x7b-32768"
            ]
            default_model = "moonshotai/kimi-k2-instruct-0905"

        # Get current model from session state, fallback to default
        current_model = st.session_state.get("llm_model", default_model)
        # If current model not in available models, use default
        if current_model not in available_models:
            current_model = default_model

        llm_model = st.selectbox(
            "Model:",
            available_models,
            index=available_models.index(current_model),
            help=f"Select the {'OpenAI' if llm_provider == 'OpenAI' else 'Groq'} model for schema generation"
        )

    # Update session state
    st.session_state.llm_model = llm_model

    # Show model information
    from src.plain.config import get_model_config
    model_config = get_model_config(llm_model)
    st.caption(
        f"ℹ️ Model: {llm_model} | "
        f"Temperature: {model_config.get('temperature', 'N/A')} | "
        f"Max Tokens: {model_config.get('max_tokens', 'N/A')}" +
        (f" | Context: {model_config.get('context_window', 'N/A'):,}" if 'context_window' in model_config else "")
    )

    st.markdown("---")

    # File upload section
    st.subheader("📁 Upload Document")

    col_upload1, col_upload2 = st.columns(2)

    with col_upload1:
        uploaded_file = st.file_uploader(
            "Upload PDF or Image",
            type=["pdf", "png", "jpg", "jpeg"],
            help="Upload the document you want to extract data from"
        )

    with col_upload2:
        template_file = None
        if mode == "Manual":
            template_file = st.file_uploader(
                "Upload Template (Excel)",
                type=["xlsx"],
                help="Upload an Excel template with field names in the first row. Prefix line item fields with 'Item' (e.g., ItemQuantity, ItemDescription).",
                key="template_uploader"
            )

    # Display uploaded file info
    if uploaded_file:
        st.success(f"✅ Document uploaded: {uploaded_file.name}")

        # Show image preview if it's an image
        if uploaded_file.type.startswith("image"):
            col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
            with col_img2:
                st.image(uploaded_file, caption="Uploaded Document", width='stretch')

    # Display template info for Manual mode
    if template_file:
        st.success(f"✅ Template uploaded: {template_file.name}")

    st.markdown("---")

    # Prompt input
    prompt = st.text_area(
        "Extraction Prompt",
        value="Extract all the important information from the invoice.",
        help="Describe what information you want to extract from the document"
    )

    # Store prompt in session state for later use
    st.session_state.extraction_prompt = prompt

    # Parse & Propose Schema button
    # Show button if document is uploaded AND (Auto mode OR Manual mode with template)
    show_parse_button = uploaded_file and (mode == "Auto" or (mode == "Manual" and template_file))

    if show_parse_button:
        if st.session_state.workflow_state == "upload":
            button_label = "🚀 Parse & Propose Schema" if mode == "Auto" else "🚀 Parse & Generate Schema from Template"
            if st.button(button_label, type="primary", width='stretch'):
                st.session_state.workflow_state = "parsing"
                st.session_state.progress_messages = []
                st.session_state.current_mode = mode  # Store mode for later use
                st.rerun()

        # Show progress messages
        if st.session_state.progress_messages:
            with st.expander("📊 Progress Log", expanded=True):
                for msg in st.session_state.progress_messages:
                    st.text(msg)

        # Parsing state
        if st.session_state.workflow_state == "parsing":
            spinner_text = "🔄 Parsing document and generating schema from template..." if mode == "Manual" else "🔄 Parsing document and proposing schema..."
            with st.spinner(spinner_text):
                # Save uploaded file to consistent location using original filename
                # This ensures cache reuse when same file is uploaded multiple times
                uploads_dir = Path(__file__).parent / "uploads"
                uploads_dir.mkdir(exist_ok=True)
                file_path = uploads_dir / uploaded_file.name

                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())

                st.session_state.temp_file_path = str(file_path)
                st.session_state.original_filename = uploaded_file.name

                # Save template file if in Manual mode
                template_path = None
                if mode == "Manual" and template_file:
                    template_path = uploads_dir / template_file.name
                    with open(template_path, 'wb') as f:
                        f.write(template_file.getbuffer())
                    st.session_state.template_path = str(template_path)

                # Run workflow with optional template_path and LLM configuration
                schema, data, agent_id = asyncio.run(
                    run_workflow(
                        str(file_path),
                        prompt,
                        template_path=str(template_path) if template_path else None,
                        llm_provider=st.session_state.llm_provider,
                        llm_model=st.session_state.llm_model
                    )
                )

                if schema and st.session_state.workflow_state == "schema_approval":
                    st.rerun()

        # Schema approval state
        if st.session_state.workflow_state == "schema_approval" and st.session_state.proposed_schema:
            st.markdown("---")
            display_schema_visual(st.session_state.proposed_schema)

            st.markdown("---")
            st.subheader("👍 Approve Schema?")

            col_approve1, col_approve2 = st.columns(2)

            with col_approve1:
                if st.button("✅ Approve & Extract Data", type="primary", width='stretch'):
                    # Reconstruct schema from edited fields
                    edited_scalars = st.session_state.edited_scalars or []
                    edited_objects = st.session_state.edited_objects or {}
                    edited_arrays = st.session_state.edited_arrays or {}

                    # Convert to list if it's a pandas DataFrame
                    if hasattr(edited_scalars, 'to_dict'):
                        edited_scalars = edited_scalars.to_dict('records')

                    # Get structure from parse to get descriptions
                    structure = parse_schema_structure(st.session_state.proposed_schema)

                    # Process objects
                    edited_objects_clean = {}
                    for obj_name, obj_fields in edited_objects.items():
                        if hasattr(obj_fields, 'to_dict'):
                            edited_objects_clean[obj_name] = {
                                "fields": obj_fields.to_dict('records'),
                                "description": structure["objects"].get(obj_name, {}).get("description", "")
                            }
                        else:
                            edited_objects_clean[obj_name] = {
                                "fields": obj_fields,
                                "description": structure["objects"].get(obj_name, {}).get("description", "")
                            }

                    # Process arrays
                    edited_arrays_clean = {}
                    for arr_name, arr_fields in edited_arrays.items():
                        if hasattr(arr_fields, 'to_dict'):
                            edited_arrays_clean[arr_name] = {
                                "fields": arr_fields.to_dict('records'),
                                "description": structure["arrays"].get(arr_name, {}).get("description", "")
                            }
                        else:
                            edited_arrays_clean[arr_name] = {
                                "fields": arr_fields,
                                "description": structure["arrays"].get(arr_name, {}).get("description", "")
                            }

                    final_schema = reconstruct_schema_from_structure(
                        edited_scalars,
                        edited_objects_clean,
                        edited_arrays_clean
                    )
                    st.session_state.final_schema = final_schema

                    st.session_state.workflow_state = "extracting"
                    st.rerun()

            with col_approve2:
                with st.expander("❌ Reject & Provide Feedback"):
                    feedback = st.text_area(
                        "What changes do you want to the schema?",
                        placeholder="e.g., Add a field for tax amount, change date format to YYYY-MM-DD"
                    )
                    if st.button("📝 Submit Feedback & Regenerate", width='stretch'):
                        if feedback:
                            # Send rejection event
                            st.session_state.workflow_handler.ctx.send_event(
                                ApprovedSchema(approved=False, feedback=feedback)
                            )
                            st.session_state.workflow_state = "parsing"
                            st.session_state.proposed_schema = None
                            st.rerun()
                        else:
                            st.warning("Please provide feedback before submitting.")

        # Extracting state
        if st.session_state.workflow_state == "extracting":
            # Check if we need to restart the workflow with approval
            if "extraction_started" not in st.session_state:
                st.session_state.extraction_started = True

                with st.spinner("🔄 Re-running workflow with approved schema..."):
                    # Re-run the workflow from scratch with the approved schema
                    # This time we'll bypass the approval step by using auto-approve
                    temp_path = st.session_state.temp_file_path
                    final_schema = st.session_state.final_schema or st.session_state.proposed_schema

                    # Create a new workflow run that will use the approved schema
                    async def run_extraction_with_schema(
                        file_path: str,
                        schema: dict,
                        prompt: str,
                        template_path: str | None = None,
                        llm_provider: str | None = None,
                        llm_model: str | None = None
                    ):
                        wf = IterativeExtractionWorkflow(timeout=None)
                        handler = wf.run(
                            file_path=file_path,
                            prompt=prompt,
                            template_path=template_path,
                            llm_provider=llm_provider,
                            llm_model=llm_model
                        )

                        # Stream until we get the proposed schema
                        async for ev in handler.stream_events():
                            if isinstance(ev, ProgressEvent):
                                st.session_state.progress_messages.append(ev.msg)
                            elif isinstance(ev, ProposedSchema):
                                # Auto-approve with our edited schema
                                handler.ctx.send_event(
                                    ApprovedSchema(approved=True, feedback="Approved", edited_schema=schema)
                                )
                                break

                        # Continue streaming until completion
                        async for ev in handler.stream_events():
                            if isinstance(ev, ProgressEvent):
                                st.session_state.progress_messages.append(ev.msg)

                        result = await handler
                        return result.data, result.agent_id

                    # Get the original prompt from session state
                    prompt = st.session_state.get("extraction_prompt", "Extract all the important information from the invoice.")
                    template_path_for_extraction = st.session_state.get("template_path", None)

                    data, agent_id = asyncio.run(
                        run_extraction_with_schema(
                            temp_path,
                            final_schema,
                            prompt,
                            template_path_for_extraction,
                            llm_provider=st.session_state.llm_provider,
                            llm_model=st.session_state.llm_model
                        )
                    )

                    st.session_state.extracted_data = data
                    st.session_state.agent_id = agent_id
                    st.session_state.workflow_state = "complete"
                    del st.session_state.extraction_started
                    st.rerun()

        # Complete state
        if st.session_state.workflow_state == "complete" and st.session_state.extracted_data:
            st.markdown("---")
            display_extracted_data(st.session_state.extracted_data)

            # Reset button
            if st.button("🔄 Process Another Document", width='stretch'):
                # Note: We keep the uploaded file in uploads/ directory for cache reuse
                # This allows faster processing if the same file is uploaded again

                # Reset state
                st.session_state.workflow_state = "upload"
                st.session_state.progress_messages = []
                st.session_state.proposed_schema = None
                st.session_state.extracted_data = None
                st.session_state.agent_id = None
                st.session_state.workflow_handler = None
                st.session_state.temp_file_path = None
                st.session_state.original_filename = None
                st.session_state.edited_scalars = None
                st.session_state.edited_objects = None
                st.session_state.edited_arrays = None
                st.session_state.final_schema = None
                st.session_state.current_schema_signature = None
                st.rerun()

    elif mode == "Manual":
        if not uploaded_file:
            st.info("👆 Please upload a document to get started.")
        elif not template_file:
            st.warning("📋 Manual mode requires an Excel template. Please upload a template file.")

    # Sidebar with info
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This tool helps you extract structured data from documents using AI.

        **Auto Mode:**
        1. Upload a document (PDF/image)
        2. AI proposes a schema
        3. Review and approve (or provide feedback)
        4. Extract data automatically

        **Manual Mode:**
        1. Upload a document (PDF/image)
        2. Upload an Excel template (.xlsx)
        3. Schema generated from template headers
        4. Review and approve (or edit)
        5. Extract data using template structure

        **Template Format:**
        - First row: Field names (e.g., InvoiceNo, Customer)
        - Prefix with `*` for required fields
        - Prefix with `Item` for line items (e.g., ItemQuantity, ItemPrice)
        """)

        if st.session_state.agent_id:
            st.markdown("---")
            st.caption(f"Agent ID: {st.session_state.agent_id}")

        st.markdown("---")
        st.subheader("🗂️ Cache Management")
        st.caption("Parsed documents are cached to speed up processing when the same file is uploaded again.")

        # Show cache info
        parsed_dir = Path(__file__).parent / "parsed"
        uploads_dir = Path(__file__).parent / "uploads"
        if parsed_dir.exists():
            cache_files = list(parsed_dir.glob("*.md"))
            st.caption(f"📦 Cached documents: {len(cache_files)}")

        # Clear cache button
        if st.button("🗑️ Clear Cache", help="Delete all cached parsed documents to force fresh parsing"):
            try:
                from src.plain.cache_utils import clear_cache
                clear_cache()
                st.success("✅ Cache cleared successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error clearing cache: {e}")


if __name__ == "__main__":
    main()
