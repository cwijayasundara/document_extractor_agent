"""Document Extraction Package.

This package provides two parallel implementations for document extraction:

1. Plain (src.plain) - LlamaIndex Workflows implementation
2. Agent (src.agent) - LangChain 1.0.2 ReAct Agent implementation

Both implementations share common utilities and services.
"""

# Export plain implementation (for backward compatibility)
from src.plain import *  # noqa: F401, F403

# Export agent implementation
from src.agent import *  # noqa: F401, F403

# Note: Subpackages can be imported directly:
# from src.plain import IterativeExtractionWorkflow
# from src.agent import create_extraction_agent
