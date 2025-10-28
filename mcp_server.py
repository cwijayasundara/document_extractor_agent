"""MCP Server for Document Extraction Agent.

This server exposes the LangChain document extraction agent via the Model Context
Protocol (MCP), preserving the full agent experience including:
- Agent reasoning and tool orchestration
- Human-in-the-loop schema approval workflow
- Iterative schema refinement with feedback

The agent uses session-based state management to maintain continuity across
the multi-step approval workflow.

Usage:
    # Install in Claude Desktop config (~/.config/Claude/claude_desktop_config.json):
    {
      "mcpServers": {
        "document-extraction-agent": {
          "command": "python",
          "args": ["/path/to/document_wf_v1/mcp_server.py"]
        }
      }
    }

    # Or run standalone for testing:
    python mcp_server.py

Environment:
    Requires .env file with:
    - LLAMA_CLOUD_API_KEY
    - OPENAI_API_KEY (or GROQ_API_KEY)
    - LLM_PROVIDER (openai or groq)
    - LLM_MODEL (e.g., gpt-5-nano, llama-3.3-70b-versatile)
"""

import os
import uuid
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from fastmcp import FastMCP

# Import agent creation
from src.agent.agent import create_extraction_agent


# ============================================================================
# CONFIGURATION
# ============================================================================

# Server metadata
SERVER_NAME = "Document Extraction Server"
SERVER_VERSION = "1.0.0"
SERVER_DESCRIPTION = "Extract structured data from PDF and image documents"

# Default LLM configuration
DEFAULT_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5-nano")

# Session management
MAX_SESSIONS = int(os.getenv("MCP_MAX_SESSIONS", "100"))
SESSION_TIMEOUT_MINUTES = int(os.getenv("MCP_SESSION_TIMEOUT_MINUTES", "60"))


# ============================================================================
# SESSION MANAGEMENT
# ============================================================================

@dataclass
class AgentSession:
    """Represents an active agent extraction session.

    Each session maintains the agent instance and its state throughout
    the multi-step workflow (parse → schema → approval → extract).
    """
    session_id: str
    agent: Any  # ApprovalAgentWrapper instance
    file_path: str
    mode: str  # "auto" or "manual"
    prompt: Optional[str] = None
    template_path: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    status: str = "initialized"  # initialized, awaiting_approval, completed, error
    parsed_text: Optional[str] = None
    proposed_schema: Optional[Dict[str, Any]] = None
    extracted_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class SessionManager:
    """Manages active agent sessions."""

    def __init__(self, max_sessions: int = 100):
        self.sessions: Dict[str, AgentSession] = {}
        self.max_sessions = max_sessions

    def create_session(
        self,
        file_path: str,
        mode: str,
        llm_provider: str,
        llm_model: str,
        prompt: Optional[str] = None,
        template_path: Optional[str] = None,
        verbose: bool = True
    ) -> str:
        """Create a new agent session.

        Args:
            file_path: Path to document
            mode: "auto" or "manual"
            llm_provider: "openai" or "groq"
            llm_model: Model identifier
            prompt: Extraction prompt (auto mode)
            template_path: Excel template path (manual mode)
            verbose: Enable agent logging

        Returns:
            session_id: Unique session identifier
        """
        # Cleanup old sessions if at limit
        if len(self.sessions) >= self.max_sessions:
            self._cleanup_oldest_sessions(keep=self.max_sessions - 1)

        # Generate unique session ID
        session_id = str(uuid.uuid4())

        # Create agent instance
        agent = create_extraction_agent(
            mode=mode,
            verbose=verbose,
            llm_provider=llm_provider,
            llm_model=llm_model
        )

        # Create session
        session = AgentSession(
            session_id=session_id,
            agent=agent,
            file_path=file_path,
            mode=mode,
            prompt=prompt,
            template_path=template_path
        )

        self.sessions[session_id] = session
        return session_id

    def get_session(self, session_id: str) -> Optional[AgentSession]:
        """Get session by ID."""
        session = self.sessions.get(session_id)
        if session:
            session.last_activity = datetime.now()
        return session

    def remove_session(self, session_id: str):
        """Remove session."""
        if session_id in self.sessions:
            del self.sessions[session_id]

    def _cleanup_oldest_sessions(self, keep: int):
        """Remove oldest sessions, keeping only 'keep' most recent."""
        sorted_sessions = sorted(
            self.sessions.items(),
            key=lambda x: x[1].last_activity,
            reverse=True
        )

        # Keep most recent sessions
        to_keep = dict(sorted_sessions[:keep])
        self.sessions = to_keep


# Global session manager
session_manager = SessionManager(max_sessions=100)


# ============================================================================
# MCP SERVER INITIALIZATION
# ============================================================================

mcp = FastMCP(
    name=SERVER_NAME
    # Note: FastMCP v2.13+ doesn't accept description parameter
    # Description: SERVER_DESCRIPTION
)


# ============================================================================
# AGENT WORKFLOW TOOLS
# ============================================================================

@mcp.tool()
async def start_document_extraction(
    file_path: str,
    mode: str = "auto",
    prompt: Optional[str] = None,
    template_path: Optional[str] = None,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """Start a document extraction workflow with the agent.

    This initiates the agent workflow:
    1. Creates a new agent session
    2. Parses the document
    3. Generates an extraction schema (AI or template-based)
    4. Pauses for human approval

    Returns a session_id and the proposed schema for review.

    Args:
        file_path: Absolute path to document (PDF, PNG, JPG, JPEG)
        mode: Extraction mode - "auto" (AI schema) or "manual" (template schema)
        prompt: Extraction instructions (auto mode only, default: extract all)
        template_path: Path to Excel template (manual mode only)
        llm_provider: LLM provider - "openai" or "groq" (default from env)
        llm_model: Model identifier (default from env)
        verbose: Enable detailed agent logging (default: True)

    Returns:
        Dictionary with:
        {
            "session_id": str,
            "status": "awaiting_approval",
            "file_path": str,
            "mode": str,
            "parsed_text": str,  # Markdown content from document
            "schema": dict,  # Proposed JSON schema for review
            "agent_logs": list  # Agent reasoning and tool calls (if verbose)
        }

    Example:
        >>> result = await start_document_extraction(
        ...     file_path="/path/to/invoice.pdf",
        ...     mode="auto",
        ...     prompt="Extract invoice header and line items"
        ... )
        >>> print(result["session_id"])
        'abc-123-def-456'
        >>> print(result["schema"]["properties"].keys())
        dict_keys(['InvoiceNo', 'Customer', 'line_items'])
    """
    try:
        # Use defaults if not specified
        provider = llm_provider or DEFAULT_LLM_PROVIDER
        model = llm_model or DEFAULT_LLM_MODEL

        # Validate mode
        if mode not in ["auto", "manual"]:
            return {
                "status": "error",
                "error": f"Invalid mode: {mode}. Use 'auto' or 'manual'."
            }

        # Validate template for manual mode
        if mode == "manual" and not template_path:
            return {
                "status": "error",
                "error": "template_path is required for manual mode"
            }

        # Set default prompt for auto mode
        if mode == "auto" and not prompt:
            prompt = "Extract all important information from the document."

        # Create new session
        session_id = session_manager.create_session(
            file_path=file_path,
            mode=mode,
            llm_provider=provider,
            llm_model=model,
            prompt=prompt,
            template_path=template_path,
            verbose=verbose
        )

        session = session_manager.get_session(session_id)

        # Build agent input
        agent_input = {
            "input": prompt or f"Extract data from {file_path}",
            "file_path": file_path,
            "mode": mode
        }

        if template_path:
            agent_input["template_path"] = template_path

        # Run agent until it pauses for approval
        result = session.agent.invoke(agent_input)

        # Check if agent is awaiting approval
        if result.get("status") == "awaiting_approval":
            session.status = "awaiting_approval"
            session.proposed_schema = result.get("schema")
            session.parsed_text = result.get("parsed_text", "")

            response = {
                "session_id": session_id,
                "status": "awaiting_approval",
                "file_path": file_path,
                "mode": mode,
                "parsed_text": session.parsed_text,
                "schema": session.proposed_schema,
            }

            # Add agent logs if available
            if verbose and hasattr(session.agent, 'last_messages'):
                response["agent_logs"] = [
                    str(msg) for msg in session.agent.last_messages
                ]

            return response
        else:
            # Unexpected: agent completed without pausing
            session.status = "error"
            session.error_message = "Agent completed without pausing for approval"

            return {
                "session_id": session_id,
                "status": "error",
                "error": "Agent did not pause for schema approval",
                "result": result
            }

    except Exception as e:
        # Store error in session if we created one
        if 'session_id' in locals():
            session = session_manager.get_session(session_id)
            if session:
                session.status = "error"
                session.error_message = str(e)

        return {
            "status": "error",
            "error": str(e)
        }


@mcp.tool()
async def approve_extraction_schema(
    session_id: str,
    approved_schema: Dict[str, Any]
) -> Dict[str, Any]:
    """Approve the proposed schema and continue extraction.

    After reviewing the schema from start_document_extraction, use this
    tool to approve it (with optional edits) and continue the agent workflow.

    The agent will:
    1. Accept the approved/edited schema
    2. Extract data from the document using the schema
    3. Return the extracted data

    Args:
        session_id: Session ID from start_document_extraction
        approved_schema: The schema to use (may be edited from original proposal)

    Returns:
        Dictionary with:
        {
            "session_id": str,
            "status": "completed",
            "extracted_data": dict,  # Structured data matching schema
            "agent_logs": list  # Agent reasoning and tool calls (if verbose)
        }

    Example:
        >>> # After start_document_extraction returned proposed schema
        >>> # User reviews and possibly edits the schema
        >>> result = await approve_extraction_schema(
        ...     session_id="abc-123",
        ...     approved_schema={
        ...         "type": "object",
        ...         "properties": {
        ...             "InvoiceNo": {"type": "string"},
        ...             "Customer": {"type": "string"},
        ...             "Date": {"type": "string"}  # User added this field
        ...         }
        ...     }
        ... )
        >>> print(result["extracted_data"]["InvoiceNo"])
        'INV-001'
    """
    try:
        # Get session
        session = session_manager.get_session(session_id)
        if not session:
            return {
                "status": "error",
                "error": f"Session not found: {session_id}"
            }

        # Verify session is waiting for approval
        if session.status != "awaiting_approval":
            return {
                "status": "error",
                "error": f"Session is not awaiting approval. Current status: {session.status}"
            }

        # Approve schema with agent
        session.agent.approve_schema(approved_schema)

        # Continue agent execution
        result = session.agent.invoke({
            "input": "Continue extraction with approved schema"
        })

        # Check if extraction completed
        if result.get("status") == "completed" or "output" in result:
            session.status = "completed"
            session.extracted_data = result.get("output", result)

            response = {
                "session_id": session_id,
                "status": "completed",
                "extracted_data": session.extracted_data
            }

            # Add agent logs if available
            if hasattr(session.agent, 'last_messages'):
                response["agent_logs"] = [
                    str(msg) for msg in session.agent.last_messages
                ]

            # Cleanup session after completion
            session_manager.remove_session(session_id)

            return response
        else:
            # Unexpected result
            session.status = "error"
            session.error_message = f"Unexpected agent result: {result}"

            return {
                "session_id": session_id,
                "status": "error",
                "error": "Agent did not complete extraction",
                "result": result
            }

    except Exception as e:
        # Update session status
        session = session_manager.get_session(session_id)
        if session:
            session.status = "error"
            session.error_message = str(e)

        return {
            "session_id": session_id,
            "status": "error",
            "error": str(e)
        }


@mcp.tool()
async def reject_extraction_schema(
    session_id: str,
    feedback: str
) -> Dict[str, Any]:
    """Reject the proposed schema and request regeneration with feedback.

    If the proposed schema is not satisfactory, use this tool to reject it
    and provide feedback. The agent will:
    1. Receive your feedback
    2. Regenerate the schema incorporating your feedback
    3. Pause again for approval with the new schema

    Args:
        session_id: Session ID from start_document_extraction
        feedback: Human feedback explaining what to change in the schema

    Returns:
        Dictionary with:
        {
            "session_id": str,
            "status": "awaiting_approval",
            "schema": dict,  # Regenerated schema based on feedback
            "agent_logs": list  # Agent reasoning and tool calls (if verbose)
        }

    Example:
        >>> # After start_document_extraction returned proposed schema
        >>> # User reviews and wants changes
        >>> result = await reject_extraction_schema(
        ...     session_id="abc-123",
        ...     feedback="Add a Date field and make Customer required"
        ... )
        >>> print(result["schema"]["required"])
        ['InvoiceNo', 'Customer']  # Customer now required
        >>> print("Date" in result["schema"]["properties"])
        True  # Date field added
    """
    try:
        # Get session
        session = session_manager.get_session(session_id)
        if not session:
            return {
                "status": "error",
                "error": f"Session not found: {session_id}"
            }

        # Verify session is waiting for approval
        if session.status != "awaiting_approval":
            return {
                "status": "error",
                "error": f"Session is not awaiting approval. Current status: {session.status}"
            }

        # Reject schema with feedback
        session.agent.reject_schema(feedback)

        # Agent regenerates schema
        result = session.agent.invoke({
            "input": f"Regenerate schema with this feedback: {feedback}"
        })

        # Check if agent is awaiting approval again
        if result.get("status") == "awaiting_approval":
            session.proposed_schema = result.get("schema")

            response = {
                "session_id": session_id,
                "status": "awaiting_approval",
                "schema": session.proposed_schema,
            }

            # Add agent logs if available
            if hasattr(session.agent, 'last_messages'):
                response["agent_logs"] = [
                    str(msg) for msg in session.agent.last_messages
                ]

            return response
        else:
            # Unexpected result
            session.status = "error"
            session.error_message = f"Unexpected agent result after rejection: {result}"

            return {
                "session_id": session_id,
                "status": "error",
                "error": "Agent did not pause for approval after regeneration",
                "result": result
            }

    except Exception as e:
        # Update session status
        session = session_manager.get_session(session_id)
        if session:
            session.status = "error"
            session.error_message = str(e)

        return {
            "session_id": session_id,
            "status": "error",
            "error": str(e)
        }


# ============================================================================
# SERVER ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Load environment variables
    import os
    from dotenv import load_dotenv

    load_dotenv()

    # Verify required environment variables
    required_vars = ["LLAMA_CLOUD_API_KEY"]
    llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if llm_provider == "openai":
        required_vars.append("OPENAI_API_KEY")
    elif llm_provider == "groq":
        required_vars.append("GROQ_API_KEY")

    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"ERROR: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set them in your .env file")
        exit(1)

    print(f"Starting {SERVER_NAME} v{SERVER_VERSION}")
    print(f"LLM Provider: {llm_provider}")
    print(f"LLM Model: {os.getenv('LLM_MODEL', 'default')}")
    print("\nAgent-based extraction with human-in-the-loop approval")
    print("\nAvailable MCP tools:")
    print("  1. start_document_extraction")
    print("     - Initialize agent workflow")
    print("     - Parse document and generate schema")
    print("     - Returns session_id + proposed schema")
    print("\n  2. approve_extraction_schema")
    print("     - Accept schema (possibly edited)")
    print("     - Complete data extraction")
    print("     - Returns extracted data")
    print("\n  3. reject_extraction_schema")
    print("     - Reject schema with feedback")
    print("     - Agent regenerates schema")
    print("     - Returns new schema for approval")
    print("\nWorkflow: start → approve/reject → (if reject: approve again) → complete")
    print("\nServer ready. Listening for MCP requests...")

    # Run the MCP server
    mcp.run()
