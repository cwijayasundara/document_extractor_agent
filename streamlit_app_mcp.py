"""Streamlit UI for MCP Server-based Document Extraction.

This application provides a web interface that connects to the MCP server for:
- Uploading PDF/image documents
- Auto mode: AI-generated schema via MCP tools
- Manual mode: Template-based schema via MCP tools
- Visual schema display and editing
- Human-in-the-loop approval workflow via MCP session management
- Data extraction and export

This client communicates with the MCP server (mcp_server.py) using its exposed tools:
- start_document_extraction
- approve_extraction_schema
- reject_extraction_schema
"""

import streamlit as st
import json
import asyncio
from pathlib import Path
import sys

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import MCP server tools directly (async functions)
from mcp_server import (
    start_document_extraction,
    approve_extraction_schema,
    reject_extraction_schema
)

from src.plain.excel_export import export_to_excel

# Import display functions from plain app (reuse)
from streamlit_app import (
    parse_schema_structure,
    reconstruct_schema_from_structure,
    display_schema_visual,
    display_extracted_data,
    init_session_state as init_plain_session_state,
)


def init_mcp_session_state():
    """Initialize Streamlit session state variables for MCP client."""
    # Reuse plain initialization for common UI state
    init_plain_session_state()

    # MCP-specific state
    if "mcp_session_id" not in st.session_state:
        st.session_state.mcp_session_id = None
    if "mcp_state" not in st.session_state:
        st.session_state.mcp_state = "upload"  # upload, running, approval, complete, error
    if "mcp_result" not in st.session_state:
        st.session_state.mcp_result = None
    if "mcp_error" not in st.session_state:
        st.session_state.mcp_error = None
    if "show_mcp_logs" not in st.session_state:
        st.session_state.show_mcp_logs = False


async def call_mcp_tool(tool_func, **kwargs):
    """Wrapper to call async MCP tools with error handling.

    Args:
        tool_func: FastMCP FunctionTool to call
        **kwargs: Arguments to pass to the tool

    Returns:
        dict: Tool response or error dict
    """
    try:
        # FastMCP tools have a .fn attribute with the underlying async function
        if hasattr(tool_func, 'fn'):
            result = await tool_func.fn(**kwargs)
        else:
            # Fallback for regular async functions
            result = await tool_func(**kwargs)
        return result
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def run_async(coro):
    """Run an async coroutine in a sync context.

    Args:
        coro: Coroutine to run

    Returns:
        Result from the coroutine
    """
    # Python 3.10+ compatible approach
    try:
        # Check if there's a running event loop
        try:
            loop = asyncio.get_running_loop()
            # If we get here, we're already in an async context
            # This shouldn't happen in Streamlit, but handle it anyway
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(coro)
        except RuntimeError:
            # No running loop - create a new one (Python 3.10+ way)
            return asyncio.run(coro)
    except Exception as e:
        # If asyncio.run() fails, fall back to manual event loop management
        import traceback
        print(f"Asyncio error: {e}")
        print(traceback.format_exc())

        # Try legacy approach as last resort
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()
        except Exception as e2:
            print(f"Legacy asyncio approach also failed: {e2}")
            raise


def display_mcp_logs():
    """Display MCP operation logs if available."""
    if not st.session_state.mcp_result:
        return

    result = st.session_state.mcp_result

    # Show agent logs if available
    if "agent_logs" in result and result["agent_logs"]:
        with st.expander("🔧 MCP Agent Logs", expanded=st.session_state.show_mcp_logs):
            for i, log in enumerate(result["agent_logs"], 1):
                st.markdown(f"**Step {i}:** {log}")


def main():
    """Main Streamlit application for MCP client."""
    st.set_page_config(
        page_title="Document Extraction - MCP Client",
        page_icon="🔌",
        layout="wide"
    )

    init_mcp_session_state()

    # Header with MCP indicator
    col_title1, col_title2 = st.columns([3, 1])
    with col_title1:
        st.title("🔌 Document Extraction - MCP Client")
    with col_title2:
        st.caption("Powered by Model Context Protocol")
        st.caption("Session-based Agent Workflow")

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
    if st.session_state.get("previous_mode") is not None and st.session_state.previous_mode != mode:
        # Mode changed - reset MCP state
        st.session_state.mcp_state = "upload"
        st.session_state.mcp_session_id = None
        st.session_state.mcp_result = None
        st.session_state.mcp_error = None
        st.session_state.proposed_schema = None
        st.session_state.extracted_data = None
        st.session_state.edited_scalars = None
        st.session_state.edited_objects = None
        st.session_state.edited_arrays = None
        st.session_state.current_schema_signature = None
        st.session_state.previous_mode = mode
        st.rerun()

    st.session_state.previous_mode = mode

    with col2:
        if mode == "Manual":
            st.info("📋 Manual mode: Upload an Excel template to define the extraction schema. Use 'Item' prefix for line item fields.")

        # MCP logs toggle
        st.session_state.show_mcp_logs = st.checkbox(
            "Show MCP agent logs",
            value=st.session_state.show_mcp_logs,
            help="Display the agent's execution logs from MCP server"
        )

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
                help="Upload an Excel template with field names in the first row. Prefix line item fields with 'Item'.",
                key="template_uploader"
            )

    # Display uploaded file info
    if uploaded_file:
        st.success(f"✅ Document uploaded: {uploaded_file.name}")

        # Show image preview if it's an image
        if uploaded_file.type.startswith("image"):
            col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
            with col_img2:
                st.image(uploaded_file, caption="Uploaded Document", use_container_width=True)

    # Display template info for Manual mode
    if template_file:
        st.success(f"✅ Template uploaded: {template_file.name}")

    st.markdown("---")

    # Prompt input
    prompt = st.text_area(
        "Extraction Prompt",
        value="Extract all the important information from the document.",
        help="Describe what information you want to extract from the document"
    )

    st.session_state.extraction_prompt = prompt

    # Parse & Extract button
    show_extract_button = uploaded_file and (mode == "Auto" or (mode == "Manual" and template_file))

    if show_extract_button:
        # Show session info if active
        if st.session_state.mcp_session_id:
            st.info(f"🔗 Active MCP Session: `{st.session_state.mcp_session_id}`")

        if st.session_state.mcp_state == "upload":
            button_label = "🔌 Start MCP Extraction" if mode == "Auto" else "🔌 Start MCP with Template"
            if st.button(button_label, type="primary", use_container_width=True):
                st.session_state.mcp_state = "running"
                st.session_state.mcp_result = None
                st.session_state.mcp_error = None
                st.rerun()

        # Running state - call MCP tool
        if st.session_state.mcp_state == "running":
            with st.spinner("🔌 Calling MCP server..."):
                # Save uploaded files
                uploads_dir = Path(__file__).parent / "uploads"
                uploads_dir.mkdir(exist_ok=True)

                file_path = uploads_dir / uploaded_file.name
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())

                st.session_state.temp_file_path = str(file_path)

                # Save template if in Manual mode
                template_path = None
                if mode == "Manual" and template_file:
                    template_path = uploads_dir / template_file.name
                    with open(template_path, 'wb') as f:
                        f.write(template_file.getbuffer())

                # If we have a session_id, we're continuing from rejection
                if st.session_state.mcp_session_id:
                    # This happens after rejection - agent should auto-regenerate
                    # We don't need to do anything, just display approval UI
                    st.session_state.mcp_state = "approval"
                    st.rerun()
                else:
                    # Call start_document_extraction MCP tool
                    mcp_mode = mode.lower()

                    tool_args = {
                        "file_path": str(file_path),
                        "mode": mcp_mode,
                        "prompt": prompt,
                        "llm_provider": st.session_state.llm_provider,
                        "llm_model": st.session_state.llm_model,
                        "verbose": True
                    }

                    if template_path:
                        tool_args["template_path"] = str(template_path)

                    # Call MCP tool
                    result = run_async(call_mcp_tool(start_document_extraction, **tool_args))
                    st.session_state.mcp_result = result

                    # Check result status
                    if result.get("status") == "awaiting_approval":
                        # Schema generated, waiting for approval
                        st.session_state.mcp_session_id = result["session_id"]
                        st.session_state.proposed_schema = result["schema"]
                        st.session_state.mcp_state = "approval"
                        st.rerun()
                    elif result.get("status") == "error":
                        st.session_state.mcp_error = result.get("error", "Unknown error")
                        st.session_state.mcp_state = "error"
                        st.rerun()
                    else:
                        # Unexpected result
                        st.session_state.mcp_error = f"Unexpected MCP response: {result.get('status', 'unknown')}"
                        st.session_state.mcp_state = "error"
                        st.rerun()

        # Display MCP logs
        display_mcp_logs()

        # Schema approval state
        if st.session_state.mcp_state == "approval" and st.session_state.proposed_schema:
            st.markdown("---")
            display_schema_visual(st.session_state.proposed_schema)

            st.markdown("---")
            st.subheader("👍 Approve Schema?")

            col_approve1, col_approve2 = st.columns(2)

            with col_approve1:
                if st.button("✅ Approve & Extract Data", type="primary", use_container_width=True):
                    # Reconstruct schema from edited fields
                    edited_scalars = st.session_state.edited_scalars or []
                    edited_objects = st.session_state.edited_objects or {}
                    edited_arrays = st.session_state.edited_arrays or {}

                    # Convert to list if DataFrame
                    if hasattr(edited_scalars, 'to_dict'):
                        edited_scalars = edited_scalars.to_dict('records')

                    # Get structure for descriptions
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

                    # Call approve_extraction_schema MCP tool
                    with st.spinner("🔌 Calling MCP approve tool..."):
                        result = run_async(call_mcp_tool(
                            approve_extraction_schema,
                            session_id=st.session_state.mcp_session_id,
                            approved_schema=final_schema
                        ))
                        st.session_state.mcp_result = result

                        # Check result
                        if result.get("status") == "completed":
                            st.session_state.extracted_data = result.get("extracted_data")
                            st.session_state.mcp_state = "complete"
                            st.rerun()
                        elif result.get("status") == "error":
                            st.session_state.mcp_error = result.get("error", "Unknown error")
                            st.session_state.mcp_state = "error"
                            st.rerun()

            with col_approve2:
                with st.expander("❌ Reject & Provide Feedback"):
                    feedback = st.text_area(
                        "What changes do you want to the schema?",
                        placeholder="e.g., Add a field for tax amount, change date format to YYYY-MM-DD",
                        key="rejection_feedback"
                    )
                    if st.button("📝 Submit Feedback & Regenerate", use_container_width=True):
                        if feedback:
                            # Call reject_extraction_schema MCP tool
                            with st.spinner("🔌 Calling MCP reject tool..."):
                                result = run_async(call_mcp_tool(
                                    reject_extraction_schema,
                                    session_id=st.session_state.mcp_session_id,
                                    feedback=feedback
                                ))
                                st.session_state.mcp_result = result

                                # Check result
                                if result.get("status") == "awaiting_approval":
                                    # New schema generated
                                    st.session_state.proposed_schema = result["schema"]
                                    st.success("✅ Schema regenerated based on your feedback!")
                                    st.rerun()
                                elif result.get("status") == "error":
                                    st.session_state.mcp_error = result.get("error", "Unknown error")
                                    st.session_state.mcp_state = "error"
                                    st.rerun()
                        else:
                            st.warning("Please provide feedback before submitting.")

        # Error state
        if st.session_state.mcp_state == "error":
            st.error("❌ An error occurred during MCP operation.")

            # Show detailed error if available
            if st.session_state.mcp_error:
                st.markdown("### Error Details")
                st.code(st.session_state.mcp_error, language="text")

            if st.session_state.mcp_result and "traceback" in st.session_state.mcp_result:
                with st.expander("📋 View Full Traceback"):
                    st.code(st.session_state.mcp_result["traceback"], language="python")

            # Show MCP result for debugging
            if st.session_state.mcp_result:
                with st.expander("🔍 MCP Response"):
                    st.json(st.session_state.mcp_result)

            if st.button("🔄 Reset MCP Session", use_container_width=True):
                st.session_state.mcp_state = "upload"
                st.session_state.mcp_session_id = None
                st.session_state.mcp_result = None
                st.session_state.mcp_error = None
                st.rerun()

        # Complete state
        if st.session_state.mcp_state == "complete" and st.session_state.extracted_data:
            st.markdown("---")
            st.success("✅ Data extraction completed successfully!")

            # Try to parse extracted data
            try:
                if isinstance(st.session_state.extracted_data, str):
                    data = json.loads(st.session_state.extracted_data)
                else:
                    data = st.session_state.extracted_data

                display_extracted_data(data)

                # Export buttons
                st.markdown("---")
                st.subheader("📥 Export Data")

                col_export1, col_export2 = st.columns(2)

                with col_export1:
                    # JSON export
                    json_data = json.dumps(data, indent=2)
                    st.download_button(
                        label="📄 Download JSON",
                        data=json_data,
                        file_name="extracted_data.json",
                        mime="application/json",
                        use_container_width=True
                    )

                with col_export2:
                    # Excel export
                    try:
                        excel_buffer = export_to_excel(data)
                        st.download_button(
                            label="📊 Download Excel",
                            data=excel_buffer.getvalue(),
                            file_name="extracted_data.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"Excel export failed: {str(e)}")

            except Exception as e:
                st.error(f"Error displaying extracted data: {str(e)}")
                st.json(st.session_state.extracted_data)

            # Reset button
            if st.button("🔄 Process Another Document", use_container_width=True):
                # Reset state
                st.session_state.mcp_state = "upload"
                st.session_state.mcp_session_id = None
                st.session_state.mcp_result = None
                st.session_state.mcp_error = None
                st.session_state.proposed_schema = None
                st.session_state.extracted_data = None
                st.session_state.edited_scalars = None
                st.session_state.edited_objects = None
                st.session_state.edited_arrays = None
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
        This tool is an **MCP Client** that connects to the Document Extraction MCP Server.

        **MCP Architecture:**
        - Client-server communication
        - Session-based workflow
        - Human-in-the-loop approval
        - LangChain ReAct Agent backend

        **Workflow:**
        1. Upload document
        2. MCP tool: `start_document_extraction`
        3. Server parses & generates schema
        4. **You review & approve**
        5. MCP tool: `approve_extraction_schema`
        6. Server extracts data
        7. Export results

        **MCP Tools:**
        - `start_document_extraction`: Initialize workflow
        - `approve_extraction_schema`: Approve and extract
        - `reject_extraction_schema`: Regenerate schema
        """)

        st.markdown("---")
        st.subheader("🔌 MCP Session Info")
        if st.session_state.mcp_session_id:
            st.caption(f"✅ Session: `{st.session_state.mcp_session_id[:8]}...`")
            st.caption(f"State: {st.session_state.mcp_state}")
        else:
            st.caption("⏳ No active session")

        st.markdown("---")
        st.subheader("🗂️ Cache Management")
        st.caption("Parsed documents are cached in the `parsed/` directory.")

        parsed_dir = Path(__file__).parent / "parsed"
        if parsed_dir.exists():
            cache_files = list(parsed_dir.glob("*.md"))
            st.caption(f"📦 Cached documents: {len(cache_files)}")

        if st.button("🗑️ Clear Cache", help="Delete all cached parsed documents"):
            try:
                from src.plain.cache_utils import clear_cache
                clear_cache()
                st.success("✅ Cache cleared!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
