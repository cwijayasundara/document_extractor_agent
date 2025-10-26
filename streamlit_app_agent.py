"""Streamlit UI for LangChain Agent-based Document Extraction.

This application provides a web interface for:
- Uploading PDF/image documents
- Auto mode: AI-generated schema with LangChain ReAct agent
- Manual mode: Template-based schema
- Visual schema display and editing
- Human-in-the-loop approval workflow
- Data extraction and export
"""

import streamlit as st
import json
import tempfile
from pathlib import Path
import sys

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.agent import create_extraction_agent
from src.plain.excel_export import export_to_excel

# Import display functions from plain app (reuse)
from streamlit_app import (
    parse_schema_structure,
    reconstruct_schema_from_structure,
    display_schema_visual,
    display_extracted_data,
    init_session_state as init_plain_session_state,
)


def init_agent_session_state():
    """Initialize Streamlit session state variables for agent."""
    # Reuse plain initialization for common UI state
    init_plain_session_state()

    # Agent-specific state
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "agent_state" not in st.session_state:
        st.session_state.agent_state = "upload"  # upload, running, approval, complete, error
    if "agent_result" not in st.session_state:
        st.session_state.agent_result = None
    if "agent_logs" not in st.session_state:
        st.session_state.agent_logs = []
    if "show_agent_reasoning" not in st.session_state:
        st.session_state.show_agent_reasoning = False


def display_agent_logs():
    """Display agent reasoning and tool calls."""
    if not st.session_state.agent_logs:
        return

    with st.expander("🤖 Agent Reasoning & Tool Calls", expanded=st.session_state.show_agent_reasoning):
        for i, log in enumerate(st.session_state.agent_logs, 1):
            st.markdown(f"**Step {i}:** {log}")


def main():
    """Main Streamlit application for agent-based extraction."""
    st.set_page_config(
        page_title="Document Extraction - LangChain Agent",
        page_icon="🤖",
        layout="wide"
    )

    init_agent_session_state()

    # Header with agent indicator
    col_title1, col_title2 = st.columns([3, 1])
    with col_title1:
        st.title("🤖 Document Extraction - LangChain Agent")
    with col_title2:
        st.caption("Powered by LangChain 1.0.2")
        st.caption("ReAct Agent Architecture")

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
        # Mode changed - reset agent state
        st.session_state.agent_state = "upload"
        st.session_state.agent = None
        st.session_state.agent_result = None
        st.session_state.proposed_schema = None
        st.session_state.extracted_data = None
        st.session_state.agent_logs = []
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

        # Agent reasoning toggle
        st.session_state.show_agent_reasoning = st.checkbox(
            "Show agent reasoning & tool calls",
            value=st.session_state.show_agent_reasoning,
            help="Display the agent's decision-making process"
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

    st.session_state.extraction_prompt = prompt

    # Parse & Extract button
    show_extract_button = uploaded_file and (mode == "Auto" or (mode == "Manual" and template_file))

    if show_extract_button:
        if st.session_state.agent_state == "upload":
            button_label = "🤖 Run Agent Extraction" if mode == "Auto" else "🤖 Run Agent with Template"
            if st.button(button_label, type="primary", width='stretch'):
                st.session_state.agent_state = "running"
                st.session_state.agent_logs = []
                st.rerun()

        # Running state - invoke agent
        if st.session_state.agent_state == "running":
            with st.spinner("🤖 Agent is working..."):
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

                # Create agent with LLM configuration
                agent_mode = mode.lower()
                if st.session_state.agent is None:
                    st.session_state.agent = create_extraction_agent(
                        mode=agent_mode,
                        verbose=True,
                        llm_provider=st.session_state.llm_provider,
                        llm_model=st.session_state.llm_model
                    )

                # Prepare agent inputs
                agent_inputs = {
                    "input": f"Extract structured data from the document at {file_path}. {prompt}",
                    "mode": agent_mode,
                    "file_path": str(file_path),
                }

                if template_path:
                    agent_inputs["template_path"] = str(template_path)

                # Invoke agent
                try:
                    result = st.session_state.agent.invoke(agent_inputs)
                    st.session_state.agent_result = result

                    # Log messages from graph execution
                    if "messages" in result:
                        from langchain_core.messages import AIMessage, ToolMessage

                        for msg in result["messages"]:
                            # Log AI messages with tool calls
                            if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
                                for tool_call in msg.tool_calls:
                                    tool_name = tool_call.get("name", "unknown")
                                    tool_args = tool_call.get("args", {})
                                    st.session_state.agent_logs.append(
                                        f"🔧 **Tool Call:** `{tool_name}`\n**Args:** {tool_args}"
                                    )

                            # Log tool responses
                            elif isinstance(msg, ToolMessage):
                                content = msg.content
                                # Truncate long responses
                                if len(str(content)) > 300:
                                    content_preview = str(content)[:300] + "..."
                                else:
                                    content_preview = str(content)

                                st.session_state.agent_logs.append(
                                    f"✅ **Tool Response:**\n```\n{content_preview}\n```"
                                )

                    # Check result status
                    if result.get("status") == "awaiting_approval":
                        # Schema generated, waiting for approval
                        st.session_state.proposed_schema = result["schema"]
                        st.session_state.agent_state = "approval"
                        st.rerun()
                    elif result.get("status") == "error":
                        st.session_state.agent_state = "error"
                        st.rerun()
                    elif "output" in result:
                        # Completed successfully
                        st.session_state.extracted_data = result["output"]
                        st.session_state.agent_state = "complete"
                        st.rerun()

                except Exception as e:
                    import traceback
                    error_tb = traceback.format_exc()

                    st.error(f"❌ Agent execution failed: {str(e)}")
                    st.session_state.agent_state = "error"
                    st.session_state.agent_logs.append(f"**ERROR:** {str(e)}")
                    st.session_state.agent_logs.append(f"**TRACEBACK:**\n```\n{error_tb}\n```")

        # Display agent logs
        display_agent_logs()

        # Schema approval state
        if st.session_state.agent_state == "approval" and st.session_state.proposed_schema:
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

                    # Approve schema in agent
                    st.session_state.agent.approve_schema(final_schema)
                    st.session_state.agent_logs.append("**APPROVAL:** Schema approved by user")

                    # Continue agent execution
                    st.session_state.agent_state = "running"
                    st.rerun()

            with col_approve2:
                with st.expander("❌ Reject & Provide Feedback"):
                    feedback = st.text_area(
                        "What changes do you want to the schema?",
                        placeholder="e.g., Add a field for tax amount, change date format to YYYY-MM-DD"
                    )
                    if st.button("📝 Submit Feedback & Regenerate", width='stretch'):
                        if feedback:
                            # Reject schema with feedback
                            st.session_state.agent.reject_schema(feedback)
                            st.session_state.agent_logs.append(f"**REJECTION:** {feedback}")

                            # Reset to running to regenerate
                            st.session_state.proposed_schema = None
                            st.session_state.agent_state = "running"
                            st.rerun()
                        else:
                            st.warning("Please provide feedback before submitting.")

        # Error state
        if st.session_state.agent_state == "error":
            st.error("❌ An error occurred during agent execution.")

            # Show detailed error if available
            if st.session_state.agent_result:
                result = st.session_state.agent_result

                if "error" in result:
                    st.markdown("### Error Details")
                    st.code(result["error"], language="text")

                if "traceback" in result:
                    with st.expander("📋 View Full Traceback"):
                        st.code(result["traceback"], language="python")

            # Show error from agent logs
            st.markdown("### Execution Logs")
            st.info("Check the agent reasoning logs above for step-by-step execution details.")

            if st.button("🔄 Reset Agent", width='stretch'):
                st.session_state.agent_state = "upload"
                st.session_state.agent = None
                st.session_state.agent_result = None
                st.rerun()

        # Complete state
        if st.session_state.agent_state == "complete" and st.session_state.extracted_data:
            st.markdown("---")

            # Try to parse extracted data
            try:
                if isinstance(st.session_state.extracted_data, str):
                    data = json.loads(st.session_state.extracted_data)
                else:
                    data = st.session_state.extracted_data

                display_extracted_data(data)
            except Exception as e:
                st.error(f"Error displaying extracted data: {str(e)}")
                st.json(st.session_state.extracted_data)

            # Reset button
            if st.button("🔄 Process Another Document", width='stretch'):
                # Reset state
                st.session_state.agent_state = "upload"
                st.session_state.agent = None
                st.session_state.agent_result = None
                st.session_state.proposed_schema = None
                st.session_state.extracted_data = None
                st.session_state.agent_logs = []
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
        This tool uses a **LangChain ReAct Agent** to extract structured data from documents.

        **Agent Architecture:**
        - Uses OpenAI GPT-5-nano for reasoning
        - Tool-calling pattern (ReAct)
        - Human-in-the-loop approval
        - Transparent decision-making

        **Workflow:**
        1. Upload document
        2. Agent parses document
        3. Agent generates schema
        4. **You review & approve**
        5. Agent extracts data
        6. Export results

        **Tools Available:**
        - `parse_document`: Parse PDF/images
        - `generate_schema_ai`: AI schema
        - `generate_schema_template`: Template schema
        - `extract_data`: Extract with schema
        - `validate_schema`: Validate format
        """)

        st.markdown("---")
        st.subheader("🤖 Agent Info")
        if st.session_state.agent:
            st.caption("✅ Agent initialized")
            st.caption(f"State: {st.session_state.agent_state}")
            st.caption(f"Steps logged: {len(st.session_state.agent_logs)}")
        else:
            st.caption("⏳ No agent initialized")

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
