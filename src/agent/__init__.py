"""LangChain agent-based document extraction system.

This package provides a ReAct agent implementation for document extraction with:
- Multi-step workflow (parse → generate schema → approve → extract)
- Human-in-the-loop schema approval
- Dual mode operation (AI-generated or template-based schemas)
- Tool-based architecture for flexibility

Usage:
    from src.agent import create_extraction_agent

    agent = create_extraction_agent(mode="auto")
    result = agent.invoke({
        "input": "Extract data from invoice.pdf",
        "file_path": "/path/to/invoice.pdf"
    })

    if result.get("status") == "awaiting_approval":
        # Show schema to user
        schema = result["schema"]
        # Get approval
        agent.approve_schema(edited_schema)
        # Continue
        result = agent.invoke({"input": "Continue extraction"})
"""

from src.agent.agent import create_extraction_agent, get_agent_description
from src.agent.executor import ApprovalAgentWrapper
from src.agent.tools import get_all_tools

__all__ = [
    "create_extraction_agent",
    "get_agent_description",
    "ApprovalAgentWrapper",
    "get_all_tools",
]
