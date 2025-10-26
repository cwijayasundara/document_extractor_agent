"""LangChain agent for document extraction."""

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from src.agent.tools import get_all_tools
from src.agent.prompts import get_system_prompt
from src.agent.executor import ApprovalAgentWrapper


def create_extraction_agent(mode: str = "auto", verbose: bool = True, model: str = "gpt-4"):
    """Create a document extraction agent using LangChain 1.0.

    Args:
        mode: Extraction mode - "auto" (AI-generated schema) or "manual" (template-based schema)
        verbose: Enable verbose logging of agent actions
        model: OpenAI model to use for agent reasoning

    Returns:
        ApprovalAgentWrapper configured for document extraction with human-in-the-loop approval

    Example:
        >>> agent = create_extraction_agent(mode="auto")
        >>> result = agent.invoke({
        ...     "input": "Extract data from /path/to/invoice.pdf",
        ...     "file_path": "/path/to/invoice.pdf",
        ...     "mode": "auto"
        ... })
        >>> if result.get("status") == "awaiting_approval":
        ...     # Display schema to user, get approval
        ...     agent.approve_schema(edited_schema)
        ...     result = agent.invoke({"input": "Continue extraction"})
    """
    # Initialize LLM
    llm = ChatOpenAI(model=model, temperature=0)

    # Get all available tools
    tools = get_all_tools()

    # Get system prompt
    system_prompt = get_system_prompt()

    # Create agent graph using LangChain 1.0 API
    graph = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        debug=verbose
    )

    # Wrap the graph with approval support
    wrapper = ApprovalAgentWrapper(graph=graph, verbose=verbose)

    return wrapper


def get_agent_description():
    """Get a description of the agent's capabilities.

    Returns:
        String describing what the agent can do
    """
    return """Document Extraction Agent (LangChain 1.0)

Capabilities:
- Parse PDF and image documents to text
- Generate extraction schemas (AI-powered or template-based)
- Human-in-the-loop schema approval workflow
- Extract structured data using approved schemas
- Dual mode operation (Auto/Manual)

Workflow:
1. Parse document → markdown text
2. Generate schema (AI or template)
3. **PAUSE for human approval**
4. Extract data with approved schema
5. Return extracted data

This agent uses LangChain 1.0's LangGraph architecture with tool calling
to break down the extraction process into clear, observable steps.
"""
