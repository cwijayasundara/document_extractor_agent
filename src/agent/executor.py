"""Custom agent wrapper with human-in-the-loop approval for schema."""

from typing import Dict, Any, Optional, List
import json
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ApprovalAgentWrapper:
    """Wrapper for LangGraph agent with human-in-the-loop approval for schema generation.

    This wrapper pauses execution after schema generation to allow human review
    and approval. The workflow can be resumed after approval is granted.

    Attributes:
        graph: The LangGraph CompiledStateGraph instance
        pending_schema: The schema waiting for approval
        approved_schema: The schema that was approved by human
        waiting_for_approval: Flag indicating if executor is waiting for approval
        rejection_feedback: User feedback if schema was rejected
        verbose: Enable verbose output
    """

    def __init__(self, graph, verbose: bool = True):
        """Initialize the approval wrapper.

        Args:
            graph: CompiledStateGraph from create_agent()
            verbose: Enable verbose logging
        """
        self.graph = graph
        self.pending_schema: Optional[dict] = None
        self.approved_schema: Optional[dict] = None
        self.waiting_for_approval: bool = False
        self.rejection_feedback: Optional[str] = None
        self.verbose = verbose
        self.last_messages: List = []
        self.last_error: Optional[str] = None
        self.last_error_traceback: Optional[str] = None
        self.file_path: Optional[str] = None  # Store file path for resumption
        self.parsed_content: Optional[str] = None  # Store parsed document content

    def invoke(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Run agent with approval checkpoints.

        Args:
            inputs: Input dict with:
                - input: The task description
                - file_path: Path to document (optional)
                - mode: "auto" or "manual" (optional)
                - template_path: Path to template for manual mode (optional)

        Returns:
            Dict with either:
                - status: "awaiting_approval" with pending schema
                - output: Final extracted data
                - error: Error message if something went wrong
        """
        from langchain_core.messages import HumanMessage

        # Convert inputs to message format for LangGraph
        user_message = inputs.get("input", "")

        # Store file_path from inputs for use in resumption messages
        if "file_path" in inputs and not self.file_path:
            self.file_path = inputs.get("file_path")
            logger.info(f"[EXECUTOR] Stored file_path: {self.file_path}")

        logger.info(f"[EXECUTOR] Invoking agent with message: {user_message[:100]}...")

        # If resuming after approval
        if self.approved_schema:
            user_message = f"The document at {self.file_path} has already been parsed. Use the approved schema to extract data directly. DO NOT call parse_document again - the document is already parsed. Call extract_data with file_path='{self.file_path}' and the following approved schema: {json.dumps(self.approved_schema)}"
            logger.info("[EXECUTOR] Resuming after schema approval")
            self.approved_schema = None
            self.waiting_for_approval = False

        # If resuming after rejection
        if self.rejection_feedback:
            user_message = f"The schema for file {self.file_path} was rejected. Feedback: {self.rejection_feedback}. Please regenerate the schema with these changes."
            logger.info(f"[EXECUTOR] Resuming after schema rejection: {self.rejection_feedback}")
            self.rejection_feedback = None
            self.pending_schema = None

        try:
            # Prepare graph inputs (LangGraph uses message-based state)
            graph_inputs = {"messages": [HumanMessage(content=user_message)]}
            logger.info("[EXECUTOR] Calling graph.invoke()...")

            # Run the graph
            result = self.graph.invoke(graph_inputs)
            logger.info("[EXECUTOR] Graph execution completed")

            # Store messages for inspection
            self.last_messages = result.get("messages", [])
            logger.info(f"[EXECUTOR] Received {len(self.last_messages)} messages from graph")

            # Log message types for debugging
            if self.verbose:
                for i, msg in enumerate(self.last_messages):
                    msg_type = type(msg).__name__
                    logger.info(f"[EXECUTOR]   Message {i}: {msg_type}")

            # Extract parsed content if parse_document was called (to avoid re-parsing on resume)
            if not self.parsed_content:
                from langchain_core.messages import ToolMessage, AIMessage

                # Find parse_document tool call and its response
                for i, msg in enumerate(self.last_messages):
                    if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls"):
                        for tool_call in msg.tool_calls:
                            if tool_call.get("name") == "parse_document":
                                # Find the corresponding ToolMessage response
                                for j in range(i + 1, len(self.last_messages)):
                                    if isinstance(self.last_messages[j], ToolMessage):
                                        self.parsed_content = self.last_messages[j].content
                                        logger.info(f"[EXECUTOR] Stored parsed content ({len(self.parsed_content)} chars)")
                                        break
                                break

            # Check if we generated a schema that needs approval
            schema_generated = self._check_for_schema_generation(self.last_messages)

            if schema_generated:
                logger.info("[EXECUTOR] Schema generation detected, waiting for approval")
                self.waiting_for_approval = True
                return {
                    "status": "awaiting_approval",
                    "schema": self.pending_schema,
                    "messages": self.last_messages,
                    "message": "Schema generated. Waiting for human approval before proceeding with extraction.",
                }

            # Normal completion - extract structured data from extract_data tool response
            output = None

            # Look for extract_data tool call and its response
            from langchain_core.messages import ToolMessage, AIMessage

            for i, msg in enumerate(self.last_messages):
                if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls"):
                    for tool_call in msg.tool_calls:
                        if tool_call.get("name") == "extract_data":
                            # Find the corresponding ToolMessage response
                            for j in range(i + 1, len(self.last_messages)):
                                if isinstance(self.last_messages[j], ToolMessage):
                                    output = self.last_messages[j].content
                                    logger.info(f"[EXECUTOR] Extracted structured data from extract_data tool")

                                    # Try to parse JSON if it's a string
                                    if isinstance(output, str):
                                        try:
                                            output = json.loads(output)
                                            logger.info(f"[EXECUTOR] Parsed JSON output successfully")
                                        except:
                                            logger.warning(f"[EXECUTOR] Output is not valid JSON, returning as-is")
                                    break
                            break

            if output is None:
                # Fallback: use last message content if no extract_data tool found
                last_message = self.last_messages[-1] if self.last_messages else None
                output = last_message.content if last_message else None
                logger.warning("[EXECUTOR] No extract_data tool response found, using last message")

            logger.info(f"[EXECUTOR] Execution complete. Output type: {type(output)}")

            return {
                "status": "complete",
                "output": output,
                "messages": self.last_messages
            }

        except Exception as e:
            # Capture detailed error information
            error_msg = str(e)
            error_tb = traceback.format_exc()

            logger.error(f"[EXECUTOR] Agent execution failed: {error_msg}")
            logger.error(f"[EXECUTOR] Traceback:\n{error_tb}")

            # Store for retrieval
            self.last_error = error_msg
            self.last_error_traceback = error_tb

            return {
                "status": "error",
                "error": error_msg,
                "traceback": error_tb,
                "message": f"Agent execution failed: {error_msg}"
            }

    def _check_for_schema_generation(self, messages: List) -> bool:
        """Check if a schema generation tool was called.

        Args:
            messages: List of messages from LangGraph state

        Returns:
            True if schema was generated, False otherwise
        """
        from langchain_core.messages import ToolMessage, AIMessage

        # Look through messages for tool calls and responses
        for i, message in enumerate(messages):
            # Check if this is an AI message with tool calls
            if isinstance(message, AIMessage) and hasattr(message, "tool_calls"):
                for tool_call in message.tool_calls:
                    tool_name = tool_call.get("name", "")

                    # Check if this is a schema generation tool
                    if "schema" in tool_name.lower() and "generate" in tool_name.lower():
                        # Find the corresponding ToolMessage response
                        for j in range(i + 1, len(messages)):
                            if isinstance(messages[j], ToolMessage):
                                observation = messages[j].content

                                # Try to parse observation as schema
                                try:
                                    if isinstance(observation, str):
                                        schema = json.loads(observation)
                                    elif isinstance(observation, dict):
                                        schema = observation
                                    else:
                                        continue

                                    # Validate it looks like a schema
                                    if "type" in schema or "properties" in schema:
                                        self.pending_schema = schema
                                        return True
                                except:
                                    # If we can't parse it, it might still be a schema
                                    # Store it as-is
                                    self.pending_schema = observation
                                    return True

        return False

    def approve_schema(self, schema: dict):
        """Approve the pending schema and allow execution to continue.

        Args:
            schema: The approved schema (can be edited version of pending schema)
        """
        self.approved_schema = schema
        self.waiting_for_approval = False
        self.pending_schema = None

    def reject_schema(self, feedback: str):
        """Reject the pending schema with feedback for regeneration.

        Args:
            feedback: User feedback describing what to change
        """
        self.rejection_feedback = feedback
        self.waiting_for_approval = False
        # Keep pending_schema for reference

    def is_waiting_for_approval(self) -> bool:
        """Check if executor is waiting for schema approval.

        Returns:
            True if waiting for approval, False otherwise
        """
        return self.waiting_for_approval

    def get_pending_schema(self) -> Optional[dict]:
        """Get the schema that is pending approval.

        Returns:
            Pending schema or None
        """
        return self.pending_schema

    def reset(self):
        """Reset the wrapper state."""
        self.pending_schema = None
        self.approved_schema = None
        self.waiting_for_approval = False
        self.rejection_feedback = None
        self.last_messages = []
        self.last_error = None
        self.last_error_traceback = None
        self.file_path = None
        self.parsed_content = None
        logger.info("[EXECUTOR] State reset")
