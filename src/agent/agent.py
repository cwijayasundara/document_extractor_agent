"""LangChain agent for document extraction."""

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from src.agent.tools import get_all_tools
from src.agent.prompts import get_system_prompt
from src.agent.executor import ApprovalAgentWrapper


def create_extraction_agent(
    mode: str = "auto",
    verbose: bool = True,
    model: str | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None
):
    """Create a document extraction agent using LangChain 1.0.

    Args:
        mode: Extraction mode - "auto" (AI-generated schema) or "manual" (template-based schema)
        verbose: Enable verbose logging of agent actions
        model: (Deprecated) OpenAI model to use for agent reasoning. Use llm_model instead.
        llm_provider: LLM provider - "openai" or "groq" (defaults to LLM_PROVIDER env var)
        llm_model: Model identifier (defaults to LLM_MODEL env var)

    Returns:
        ApprovalAgentWrapper configured for document extraction with human-in-the-loop approval

    Example:
        >>> agent = create_extraction_agent(
        ...     mode="auto",
        ...     llm_provider="groq",
        ...     llm_model="moonshotai/kimi-k2-instruct-0905"
        ... )
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
    from src.plain.config import get_model_config, LLM_PROVIDER, LLM_MODEL

    # Handle legacy 'model' parameter
    if model is not None and llm_model is None:
        llm_model = model

    # Use defaults from config if not specified
    provider = llm_provider or LLM_PROVIDER
    model_id = llm_model or LLM_MODEL

    # Get model configuration
    model_config = get_model_config(model_id)
    temperature = model_config.get("temperature", 0.7)

    # Initialize LLM based on provider
    if provider.lower() == "openai":
        llm = ChatOpenAI(model=model_id, temperature=temperature)
    elif provider.lower() == "groq":
        # Get Groq API key from environment FIRST
        import os
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")

        # Groq uses OpenAI-compatible API with custom base URL
        # Pass API key as string directly for sync/async client compatibility
        llm = ChatOpenAI(
            model=model_id,
            temperature=temperature,
            base_url="https://api.groq.com/openai/v1",  # Modern parameter
            api_key=groq_api_key,  # Direct string enables both sync and async clients
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}. Use 'openai' or 'groq'.")

    # Get provider-specific tools
    # Some tools (like ValidateSchemaTool) have compatibility issues with Groq
    tools = get_all_tools(provider=provider)

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
