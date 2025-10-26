# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a document extraction system with **TWO PARALLEL IMPLEMENTATIONS**:

### 1. Plain Implementation (`src/plain/`)
- **Architecture:** LlamaIndex Workflows (event-driven)
- **Workflow:** Document Parsing → Schema Generation → Human Approval → Data Extraction → Export
- **UI:** `streamlit_app.py`
- **Use Case:** Stable, production-ready, event-driven workflow

### 2. Agent Implementation (`src/agent/`)
- **Architecture:** LangChain 1.0.2 ReAct Agent (tool-based)
- **Workflow:** Agent orchestrates tools with human-in-the-loop approval
- **UI:** `streamlit_app_agent.py`
- **Use Case:** Transparent reasoning, extensible tools, debugging-friendly

**This document primarily covers the Plain Implementation.** For Agent Implementation details, see `README_AGENT.md` and `AGENTS.md`.

## Running the Application

### Plain Implementation (LlamaIndex Workflows)
```bash
# Streamlit UI (recommended)
streamlit run streamlit_app.py

# CLI mode
python src/plain/main.py docs/invoice.pdf

# With specific prompt
python src/plain/main.py docs/invoice.pdf --prompt "Extract all fields from the invoice"
```

### Agent Implementation (LangChain ReAct)
```bash
# Streamlit UI with agent reasoning
streamlit run streamlit_app_agent.py

# Programmatic usage
python -c "
from src.agent import create_extraction_agent
agent = create_extraction_agent(mode='auto', verbose=True)
result = agent.invoke({'input': 'Extract from invoice.pdf', 'file_path': 'docs/invoice.pdf'})
"
```

### Cache Management
```bash
# Clear cache to force fresh parsing
rm -rf parsed/

# Cache is stored in parsed/ directory (from PARSE_CACHE_DIR in src/config.py)
```

## Architecture

The project contains two parallel implementations with different orchestration strategies.

---

## Plain Implementation Architecture

### Entry Points
- **Streamlit UI:** `streamlit_app.py`
- **CLI:** `src/plain/main.py`

### Core Approach
- Uses LlamaIndex Workflows for event-driven processing
- Async/await for non-blocking I/O
- Simple, linear workflow with human-in-the-loop approval

### Plain Implementation Structure

```
src/plain/
├── extraction_workflow.py           # Main workflow with event handlers
│   ├── IterativeExtractionWorkflow  # Core workflow class
│   ├── parse_file()                 # Step 1: Parse document
│   ├── propose_schema()             # Step 2: Generate schema (AI or template)
│   ├── handle_schema_approval()     # Step 3: Handle user approval/rejection
│   └── run_extraction()             # Step 4: Extract data using approved schema
│
├── config.py                        # Configuration and client factories (shared)
│   ├── API keys and settings
│   ├── get_parse_client()           # LlamaParse client factory
│   ├── get_extract_client()         # LlamaExtract client factory
│   └── get_openai_client()          # OpenAI client factory
│
├── cache_utils.py                   # Parsing cache management (shared)
│   ├── get_cached_parse()           # Check if document is cached
│   ├── save_parsed_content()        # Save parsed content to cache
│   └── clear_cache()                # Clear all cached results
│
├── template_schema_generator.py    # Excel template → JSON schema (shared)
│   ├── TemplateSchemaGenerator      # Template processor class
│   └── generate_schema_from_template()  # Convenience function
│
├── excel_export.py                  # Export extracted data to Excel (shared)
│   └── export_to_excel()            # Generate Excel file with multiple sheets
│
├── main.py                          # CLI entry point
└── __init__.py                      # Package exports

streamlit_app.py                     # Plain implementation Streamlit UI
```

---

## Agent Implementation Architecture

### Entry Points
- **Streamlit UI:** `streamlit_app_agent.py`
- **Programmatic:** `from src.agent import create_extraction_agent`

### Core Approach
- Uses LangChain 1.0.2 ReAct Agent for tool-calling
- Human-in-the-loop via `ApprovalAgentWrapper`
- Transparent reasoning with tool call visibility
- Modular tool architecture for extensibility

### Agent Implementation Structure

```
src/agent/
├── agent.py                         # Agent creation and setup
│   ├── create_extraction_agent()    # Main factory function
│   └── get_agent_description()      # Agent capabilities description
│
├── executor.py                      # Custom approval wrapper
│   └── ApprovalAgentWrapper         # Pauses for schema approval
│       ├── invoke()                 # Run agent with checkpoints
│       ├── approve_schema()         # Accept and continue
│       ├── reject_schema()          # Reject with feedback
│       └── _check_for_schema_generation()  # Detect schema step
│
├── prompts.py                       # System prompts
│   └── get_system_prompt()          # Agent instructions and workflow
│
├── tools/                           # LangChain tool definitions
│   ├── __init__.py                  # Tool registry
│   ├── parse_tool.py                # ParseDocumentTool (uses LlamaParse)
│   ├── schema_ai_tool.py            # GenerateSchemaAITool (uses OpenAI)
│   ├── schema_template_tool.py      # GenerateSchemaTemplateTool (uses Excel)
│   ├── extraction_tool.py           # ExtractDataTool (uses LlamaExtract)
│   └── validation_tool.py           # ValidateSchemaTool (jsonschema)
│
└── __init__.py                      # Package exports

streamlit_app_agent.py               # Agent implementation Streamlit UI
```

### Shared Components

Both implementations share these modules from `src/plain/`:
- `config.py` - API client factories (LlamaParse, LlamaExtract, OpenAI)
- `cache_utils.py` - Document parsing cache
- `template_schema_generator.py` - Excel template processing
- `excel_export.py` - Export utilities

---

## Key Features (Both Implementations)

### 1. Dual Mode Operation

Both implementations support:

**Auto Mode:**
- AI analyzes document and proposes schema using OpenAI
- User reviews and can edit schema
- Iterative refinement with feedback

**Manual Mode:**
- User uploads Excel template (.xlsx)
- Schema generated from template headers
- Fields starting with "Item" become line items
- Consistent schema across similar documents

### 2. Workflow Steps

**Plain Implementation Workflow:**

```
1. Parse Document (LlamaParse)
   ↓
2. Generate Schema (OpenAI or Excel Template)
   ↓
3. User Approval Loop
   ├─ Approve → Continue
   └─ Reject → Provide feedback → Regenerate schema
   ↓
4. Extract Data (LlamaExtract)
   ↓
5. Display & Export (Streamlit UI or JSON)
```

**Agent Implementation Workflow:**

```
Agent receives task
   ↓
1. Agent calls parse_document tool
   → LlamaParse converts document to markdown
   ↓
2. Agent calls schema generation tool
   → Auto: generate_schema_ai (OpenAI)
   → Manual: generate_schema_template (Excel)
   ↓
3. ApprovalAgentWrapper detects schema generation
   → Pauses execution
   → Returns status: "awaiting_approval"
   → UI displays schema for review
   ↓
4. Human reviews and responds
   ├─ Approve → wrapper.approve_schema(edited_schema)
   └─ Reject → wrapper.reject_schema(feedback)
   ↓
5. Agent resumes execution
   → Calls extract_data tool with approved schema
   → Returns extracted data
   ↓
6. Display & Export
```

**Key Difference:** Agent implementation uses tool-calling pattern where each step is an explicit tool invocation visible in logs. Plain implementation uses event-driven state machine.

### 3. Agent Tools (Agent Implementation Only)

The agent has access to 5 LangChain tools:

1. **`parse_document`** - Parse PDF/image to markdown (LlamaParse)
2. **`generate_schema_ai`** - Generate JSON schema with AI (OpenAI)
3. **`generate_schema_template`** - Generate schema from Excel template
4. **`extract_data`** - Extract structured data with approved schema (LlamaExtract)
5. **`validate_schema`** - Validate JSON schema format (jsonschema)

Each tool wraps the same underlying services used by the plain implementation but exposes them as discrete, callable functions for the agent.

### 4. Caching Strategy

All parsing results are cached to `parsed/` directory:
- Filename: `{original_filename}.md`
- Format: Markdown text from LlamaParse
- Automatically reused when same document uploaded

Cache speeds up repeated extractions and allows schema iteration without re-parsing.

**Note:** Both implementations use the same cache system from `src/plain/cache_utils.py`.

### 5. Excel Export

Export extracted data to Excel (.xlsx) with 4 sheets:
1. **Metadata**: Export timestamp, field counts
2. **Document Headers**: Flattened key-value pairs
3. **Line Items**: Tabular data for repeating items
4. **Raw Data**: Complete JSON for reference

**Note:** Both implementations use the same export utility from `src/plain/excel_export.py`.

### 6. Template-Based Extraction

Create Excel template with field names in first row:
```
*InvoiceNo | *Customer | *Date | ItemDescription | ItemQuantity | ItemRate | ItemAmount
```

Conventions:
- `*` prefix = required field
- `Item` prefix = line item field (removed in output)
- Spaces → underscores in property names

Generated schema:
- Document fields: `InvoiceNo`, `Customer`, `Date`
- Line items: `Description`, `Quantity`, `Rate`, `Amount`

**Note:** Both implementations use the same template processor from `src/plain/template_schema_generator.py`.

---

## Plain Implementation Details

This section describes implementation-specific details for the Plain (LlamaIndex Workflows) implementation.

### Event Classes

Defined in `src/plain/extraction_workflow.py`:

```python
# Workflow Events
InputEvent         # Start workflow (file_path, prompt, template_path)
ParsedContent      # Document parsed (file_content, prompt)
ProposedSchema     # Schema generated (generated_schema)
ApprovedSchema     # User response (approved, feedback, edited_schema)
RunExtraction      # Run extraction (generated_schema, file_content)
ExtractedData      # Extraction complete (data, agent_id)
ProgressEvent      # Progress update (msg)
```

### Workflow State

Stored in `WorkflowState` (Pydantic model in `src/plain/extraction_workflow.py`):
```python
file_path: str | None           # Path to uploaded document
file_content: str | None        # Parsed markdown content
current_schema: dict | None     # Current JSON schema
current_feedback: str | None    # User feedback for iteration
original_prompt: str | None     # Initial extraction prompt
template_path: str | None       # Excel template path (Manual mode)
```

---

## Agent Implementation Details

This section describes implementation-specific details for the Agent (LangChain ReAct) implementation.

### Agent State Management

The agent's state is managed by `ApprovalAgentWrapper` in `src/agent/executor.py`:

```python
class ApprovalAgentWrapper:
    pending_schema: Optional[dict]      # Schema awaiting approval
    approved_schema: Optional[dict]     # Approved schema to continue with
    waiting_for_approval: bool          # Paused for human approval
    rejection_feedback: Optional[str]   # Feedback if schema rejected
    last_messages: List                 # Message history from graph
    last_error: Optional[str]           # Error message if failed
```

### Tool Calling Pattern

Each tool is a subclass of `langchain_core.tools.BaseTool`:

```python
class ParseDocumentTool(BaseTool):
    name: str = "parse_document"
    description: str = "Parse a PDF or image..."
    args_schema: Type[BaseModel] = ParseDocumentInput

    def _run(self, file_path: str) -> str:
        # Implementation using LlamaParse
        return parsed_markdown
```

Tools are registered in `src/agent/tools/__init__.py` via `get_all_tools()`.

### Human-in-the-Loop Approval

The `ApprovalAgentWrapper` intercepts schema generation:

1. **Detection:** Scans messages for tool calls containing "schema" and "generate"
2. **Pause:** Returns `{"status": "awaiting_approval", "schema": ...}`
3. **Resume:** After `approve_schema()` or `reject_schema()`, continues execution
4. **Feedback Loop:** If rejected, agent regenerates schema with feedback

This is different from Plain implementation's event-based approval (ProposedSchema/ApprovedSchema events).

### Agent System Prompt

Defined in `src/agent/prompts.py`:

- Instructs agent on tool usage order
- Enforces pause after schema generation
- Provides workflow structure (parse → schema → approve → extract)
- Explains Auto vs Manual mode differences

---

## Configuration

All configuration in `src/plain/config.py` (shared by both implementations):

```python
# Environment variables (from .env)
OPENAI_API_KEY
OPENAI_MODEL = "gpt-5-nano"
LLAMA_CLOUD_API_KEY
LLAMA_CLOUD_PROJECT_ID
LLAMA_CLOUD_ORG_ID

# Directories
PARSE_CACHE_DIR = Path("parsed")

# Client factories
get_parse_client()     # Returns LlamaParse instance
get_extract_client()   # Returns LlamaExtract instance
get_openai_client()    # Returns AsyncOpenAI instance
```

---

## Streamlit UIs

### Plain Implementation UI (`streamlit_app.py`)

**Features:**
- Mode selector (Auto/Manual)
- File upload (PDF, PNG, JPG, JPEG)
- Template upload (Excel .xlsx for Manual mode)
- Visual schema editor with editable tables
- Progress log with real-time updates
- Approval/rejection workflow
- Data display with formatted headers and line items
- Export buttons (JSON, Excel)

**Session State:**
- `workflow_state`: Current state (upload/parsing/schema_approval/extracting/complete)
- `proposed_schema`: AI or template-generated schema
- `extracted_data`: Final extracted data
- `progress_messages`: Workflow progress log
- `edited_scalars`: User edits to scalar fields
- `edited_objects`: User edits to object fields
- `edited_arrays`: User edits to array fields

### Agent Implementation UI (`streamlit_app_agent.py`)

**Additional Features:**
- 🤖 Agent reasoning toggle (show/hide tool calls)
- Tool call logs with arguments and responses
- Agent state indicator (upload/running/approval/complete/error)
- Explicit agent initialization status

**Session State:**
- `agent`: LangChain agent instance
- `agent_state`: Current state (upload/running/approval/complete/error)
- `agent_result`: Result from agent invocation
- `agent_logs`: List of tool calls and reasoning steps
- `show_agent_reasoning`: Toggle for reasoning visibility
- Plus all UI state from plain implementation

**Shared UI Components:**
Both UIs share these display functions from `streamlit_app.py`:
- `parse_schema_structure()` - Parse JSON schema into UI-friendly structure
- `reconstruct_schema_from_structure()` - Rebuild schema from edited fields
- `display_schema_visual()` - Render schema as editable tables
- `display_extracted_data()` - Show extracted data with formatting

---

## Common Development Tasks

### Modifying Schema Generation

**Plain Implementation (Auto mode):**
Edit `propose_schema()` in `src/plain/extraction_workflow.py`
- Modify the `extract_prompt` template
- Adjust OpenAI temperature/model settings
- File: `src/plain/extraction_workflow.py:182`

**Agent Implementation (Auto mode):**
Edit `GenerateSchemaAITool` in `src/agent/tools/schema_ai_tool.py`
- Modify the prompt passed to OpenAI
- Adjust tool description for agent
- Change tool parameters

**Template-based (Both implementations - shared):**
Edit `TemplateSchemaGenerator` in `src/plain/template_schema_generator.py`
- Modify field classification logic (`classify_fields()`)
- Adjust type inference (`_infer_field_type()`)
- Change field name cleaning (`_clean_field_name()`)

### Adding New Workflow Steps

**Plain Implementation:**
1. Define new event class in `src/plain/extraction_workflow.py`
2. Add `@step` decorated method to `IterativeExtractionWorkflow`
3. Return appropriate event to trigger next step
4. Update `streamlit_app.py` to handle new events

**Agent Implementation:**
1. Create new tool in `src/agent/tools/my_tool.py`:
   ```python
   from langchain_core.tools import BaseTool
   from pydantic import BaseModel, Field
   from typing import Type

   class MyToolInput(BaseModel):
       param: str = Field(description="...")

   class MyTool(BaseTool):
       name: str = "my_tool"
       description: str = "Detailed description for agent"
       args_schema: Type[BaseModel] = MyToolInput

       def _run(self, param: str) -> str:
           # Implementation
           return result
   ```
2. Register in `src/agent/tools/__init__.py`:
   ```python
   from src.agent.tools.my_tool import MyTool

   def get_all_tools():
       return [..., MyTool()]
   ```
3. Update system prompt in `src/agent/prompts.py` if needed
4. Tool automatically available to agent

### Modifying Export Format

Edit `src/plain/excel_export.py` (shared by both):
- Add new sheets (e.g., summary, analytics)
- Change formatting (fonts, colors, column widths)
- Add data transformations
- Changes apply to both implementations

### Changing UI Layout

**Plain Implementation UI:**
Edit `streamlit_app.py`:
- Modify mode selection UI
- Change schema display (`display_schema_visual()`)
- Update data display (`display_extracted_data()`)
- Add new UI components

**Agent Implementation UI:**
Edit `streamlit_app_agent.py`:
- Same as above, plus:
- Modify agent reasoning display (`display_agent_logs()`)
- Change agent state indicators
- Update tool call formatting

**Shared UI Components:**
Both UIs import display functions from `streamlit_app.py`, so changes to those functions affect both

---

## Data Flow

### Plain Implementation - Auto Mode
```
1. Upload Document
   ↓
2. Parse with LlamaParse (cached)
   → Save to parsed/{filename}.md
   ↓
3. Generate Schema with OpenAI
   → Uses extracted_prompt template
   → Validates JSON Schema Draft 7
   ↓
4. User Reviews/Edits Schema
   → Edit in visual table (Streamlit)
   → Can reject with feedback
   ↓
5. Extract Data with LlamaExtract
   → Uses approved schema
   → Cleans up agent after extraction
   ↓
6. Display & Export
   → Streamlit tables
   → JSON download
   → Excel download (4 sheets)
```

### Plain Implementation - Manual Mode
```
1. Upload Document + Excel Template
   ↓
2. Parse with LlamaParse (cached)
   ↓
3. Generate Schema from Template
   → Read headers from Excel row 1
   → Classify: Item prefix = line items
   → Infer types from field names
   ↓
4. User Reviews/Edits Schema
   → Same as Auto mode
   ↓
5. Extract & Export
   → Same as Auto mode
```

### Agent Implementation - Auto Mode

```
1. User uploads document via streamlit_app_agent.py
   ↓
2. Agent receives task: "Extract from {file_path}. {prompt}"
   ↓
3. Agent calls parse_document tool
   → Tool uses LlamaParse (cached)
   → Returns markdown text to agent
   ↓
4. Agent calls generate_schema_ai tool
   → Tool uses OpenAI to analyze document
   → Returns JSON schema
   ↓
5. ApprovalAgentWrapper detects schema generation
   → Scans message history for schema tool call
   → Extracts schema from ToolMessage
   → Pauses execution
   → Returns {"status": "awaiting_approval", "schema": {...}}
   ↓
6. UI displays schema in editable tables
   → User reviews, edits, and clicks Approve/Reject
   ↓
7a. If APPROVED:
   → wrapper.approve_schema(edited_schema)
   → Agent resumes with approved schema
   → Agent calls extract_data tool
   → Tool uses LlamaExtract
   → Returns extracted data
   ↓
7b. If REJECTED:
   → wrapper.reject_schema(feedback)
   → Agent regenerates schema with feedback
   → Returns to step 4
   ↓
8. Display & Export
   → Streamlit tables (same display functions as plain)
   → JSON download
   → Excel download (same export utility)
```

### Agent Implementation - Manual Mode

```
1. User uploads document + Excel template
   ↓
2. Agent receives task with template_path
   ↓
3. Agent calls parse_document tool
   → Same as Auto mode
   ↓
4. Agent calls generate_schema_template tool
   → Tool uses TemplateSchemaGenerator
   → Reads Excel headers
   → Returns JSON schema
   ↓
5-8. Same approval and extraction flow as Auto mode
```

**Key Difference:** Agent implementation makes each step an explicit tool call that's logged and visible. Plain implementation uses internal event handlers.

---

## Dependencies

Core libraries (from `requirements.txt`):

**Plain Implementation:**
```
llama-index-workflows      # Workflow engine (event-driven)
```

**Agent Implementation:**
```
langchain==1.0.2           # LangChain framework
langchain-openai==1.0.1    # OpenAI integration for LangChain
langchain-core==1.0.1      # Core LangChain components
langchain-community==0.4.0 # Community tools
```

**Shared Dependencies:**
```
llama-cloud-services       # LlamaParse, LlamaExtract APIs
openai                     # OpenAI API (used by both)
streamlit                  # Web UI framework
openpyxl                   # Excel file creation
jsonschema                 # JSON Schema validation
python-dotenv              # Environment variable management
```

Install: `pip install -r requirements.txt`

---

## Testing

### Plain Implementation Testing

**Streamlit UI:**
```bash
streamlit run streamlit_app.py
```

**CLI:**
```bash
# Test Auto mode
python src/plain/main.py docs/IMG_4693.JPEG

# Test with custom prompt
python src/plain/main.py docs/invoice.pdf --prompt "Extract vendor and line items"
```

### Agent Implementation Testing

**Streamlit UI:**
```bash
streamlit run streamlit_app_agent.py
```

**Programmatic:**
```bash
python -c "
from src.agent import create_extraction_agent

# Create agent with verbose output
agent = create_extraction_agent(mode='auto', verbose=True)

# Run extraction
result = agent.invoke({
    'input': 'Extract all fields from the invoice',
    'file_path': 'docs/invoice.pdf',
    'mode': 'auto'
})

# Check status
if result.get('status') == 'awaiting_approval':
    print('Schema:', result['schema'])
    # In real usage, display to user and get approval
    agent.approve_schema(result['schema'])
    result = agent.invoke({'input': 'Continue extraction'})

print('Result:', result)
"
```

**Test with Mock Approval:**
```bash
python src/auto_schema_gen.py
# This runs the plain workflow with interactive approval prompts
```

**Cache Testing (Both implementations):**
```bash
# First run - parses document
python src/plain/main.py docs/invoice.pdf

# Second run - uses cache (faster)
python src/plain/main.py docs/invoice.pdf

# Clear cache and re-test
rm -rf parsed/
python src/plain/main.py docs/invoice.pdf
```

---

## Important Implementation Details

### Why LlamaExtract for Extraction (Both Implementations)

Uses LlamaExtract for data extraction (not OpenAI) because:
- Designed specifically for structured data extraction
- Better handling of JSON schemas
- More consistent output format

### Agent Cleanup (Both Implementations)

LlamaExtract creates persistent agents. Both implementations automatically clean them up:

**Plain Implementation:**
```python
# In src/plain/extraction_workflow.py
finally:
    if hasattr(extract, 'delete_agent'):
        extract.delete_agent(agent.id)
```

**Agent Implementation:**
```python
# In src/agent/tools/extraction_tool.py
finally:
    if hasattr(extract, 'delete_agent'):
        extract.delete_agent(agent.id)
```

### Nested Schema Support

Streamlit UI flattens nested schemas for editing:
- `seller.name` instead of `seller: {name: ...}`
- Reconstructs nested structure on approval
- Uses dot notation for clarity

### Arrow Serialization

All DataFrame values converted to strings for Streamlit compatibility:
```python
# Prevents PyArrow errors
value = str(value) if value is not None else "—"
```

## File Locations

**User Data:**
- Uploaded files: `uploads/` (gitignored)
- Cached parses: `parsed/` (gitignored)
- Excel templates: `docs/` (versioned)

**Source Code:**
- Core workflow: `src/`
- Web UI: `streamlit_app.py`
- Documentation: `*.md` files

**Configuration:**
- Environment: `.env` (gitignored)
- Requirements: `requirements.txt`
- Git rules: `.gitignore`

---

## Troubleshooting

### Import Errors (Both Implementations)
```
ModuleNotFoundError: No module named 'src' or 'src.plain'
```
**Solution:**
- Run from project root, not from src/ directory
- Ensure virtual environment is activated
- Check that `__init__.py` files exist in `src/`, `src/plain/`, and `src/agent/`

### Cache Issues (Both Implementations)
```
Document always re-parses even though cached
```
**Solution:**
- Check that filename matches exactly (case-sensitive)
- Verify `parsed/` directory exists and is writable
- Clear cache with `rm -rf parsed/` and retry

### Schema Not Displaying (Both Implementations)
```
Schema generated but not shown in UI tables
```
**Solution:**
- Check browser console for errors
- Ensure nested objects are flattened correctly
- Verify schema is valid JSON Schema Draft 7
- Check Streamlit version compatibility

### Template Not Working (Both Implementations)
```
Manual mode: schema generation failed
```
**Solution:**
- Ensure .xlsx format (not .xls)
- Check headers in row 1
- Verify "Item" prefix for line item fields
- Template file in `src/plain/template_schema_generator.py` is shared by both

### Agent-Specific Issues

**Agent Not Stopping for Approval:**
```
Agent continues to extraction without waiting for approval
```
**Solution:**
- Check `ApprovalAgentWrapper._check_for_schema_generation()` in `src/agent/executor.py`
- Ensure tool names contain "schema" and "generate"
- Verify tool response contains valid JSON
- Enable `verbose=True` when creating agent to see message flow

**Agent Tool Call Failures:**
```
Tool execution failed with error
```
**Solution:**
- Check tool implementation in `src/agent/tools/`
- Verify tool is using shared services from `src/plain/config.py`
- Check API keys in `.env`
- Review agent logs in UI (enable "Show agent reasoning")

**Agent Hangs or Loops:**
```
Agent keeps calling same tool repeatedly
```
**Solution:**
- Check system prompt in `src/agent/prompts.py`
- Ensure tool descriptions are clear
- Verify agent receives proper responses from tools
- Check if approval wrapper is properly detecting schema generation

---

## When to Use Which Implementation

### Choose Plain Implementation If:
- You want proven stability for production
- Event-driven architecture fits your mental model
- You don't need visibility into decision-making
- You prefer simpler execution flow
- You're comfortable with LlamaIndex Workflows

### Choose Agent Implementation If:
- You need transparent reasoning and debugging
- You want to see each step as an explicit tool call
- You plan to extend with custom tools
- You're building on LangChain ecosystem
- You want fine-grained control over agent behavior
- You need to explain extraction logic to stakeholders

### Migration Strategy

Both implementations are production-ready. You can:

1. **Run both in parallel** (already supported)
2. **Compare outputs** with same documents
3. **Switch users gradually** based on use case
4. **Maintain both** for different scenarios

No code changes needed - both share core services and display functions.

---

## Additional Resources

- **Plain Implementation:** See this file for workflow details
- **Agent Implementation:** See `README_AGENT.md` for architecture comparison
- **Development Guide:** See `AGENTS.md` for coding guidelines
- **Quick Start:** See `QUICKSTART.md` for setup instructions
- **Templates:** See `TEMPLATE_USAGE.md` for Excel template help
