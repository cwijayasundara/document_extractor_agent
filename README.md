# Document Extraction System

**Complete Guide - All Documentation in One Place**

An intelligent document extraction system with AI-powered schema generation, template-based extraction, and multiple implementation options. This system provides parallel implementations using different orchestration approaches while maintaining feature parity.

---

## Table of Contents

1. [Overview & Quick Start](#overview--quick-start)
2. [Features](#features)
3. [Installation & Setup](#installation--setup)
4. [Running the Applications](#running-the-applications)
5. [Usage Guide](#usage-guide)
6. [MCP Server Integration](#mcp-server-integration)
7. [Architecture Deep Dive](#architecture-deep-dive)
8. [Development Guide](#development-guide)
9. [Configuration](#configuration)
10. [Dependencies & Compatibility](#dependencies--compatibility)
11. [Troubleshooting](#troubleshooting)
12. [Testing](#testing)
13. [Examples & Workflows](#examples--workflows)

---

## Overview & Quick Start

### What is This?

This is a document extraction system that uses AI to convert unstructured documents (PDFs, images) into structured data (JSON, Excel). It features:

- **Three Parallel Implementations**: Plain (LlamaIndex), Agent (LangChain), and MCP Client
- **Dual Mode Operation**: AI-generated schemas or template-based extraction
- **Human-in-the-Loop**: Review and approve schemas before extraction
- **Multiple LLM Providers**: OpenAI or Groq
- **Rich Export Options**: JSON and multi-sheet Excel files

### Three Implementations Available

#### 1. Plain Implementation (Stable & Event-Driven)
- **Technology:** LlamaIndex Workflows
- **Architecture:** Event-driven state machine
- **UI:** `streamlit run streamlit_app.py`
- **Best for:** Production use, proven stability

#### 2. Agent Implementation (Transparent & Tool-Based)
- **Technology:** LangChain 1.0.2 ReAct Agent
- **Architecture:** Tool-calling with visible reasoning
- **UI:** `streamlit run streamlit_app_agent.py`
- **Best for:** Debugging, transparency, extensibility

#### 3. MCP Client (Client-Server Architecture)
- **Technology:** Model Context Protocol + Agent
- **Architecture:** Session-based workflow
- **UI:** `./run_mcp_client.sh`
- **Best for:** Scalable production with separated concerns

### 30-Second Quick Start

```bash
# 1. Install dependencies
python3 -m venv .venv --copies
source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure API keys
echo "LLAMA_CLOUD_API_KEY=your_key" > .env
echo "OPENAI_API_KEY=your_key" >> .env

# 3. Run the app (choose one)
./run_mcp_client.sh      # MCP Client (recommended)
./run_agent_app.sh       # Agent App
./run_plain_app.sh       # Plain App
```

---

## Features

### Core Features (All Implementations)

#### Dual Mode Operation
- **Auto Mode**: AI analyzes document and generates schema automatically
- **Manual Mode**: Use Excel templates for consistent structured extraction

#### Interactive Web UI (Streamlit)
- Visual schema editor with editable tables
- Real-time progress tracking
- Approval/rejection workflow with feedback
- Multi-format export (JSON, Excel)

#### Smart Caching
- Automatic caching of parsed documents in `parsed/` directory
- Instant schema iterations without re-parsing
- Filename-based cache lookup

#### Excel Export
- Multi-sheet workbooks (Metadata, Headers, Line Items, Raw Data)
- Professional formatting with auto-adjusted columns
- Perfect for business users and further analysis

#### Flexible Template System
- Create custom Excel templates with your field names
- Automatic line item detection (fields starting with "Item")
- Type inference from field names
- Reusable across similar documents

### Implementation-Specific Features

| Feature | Plain | Agent | MCP Client |
|---------|-------|-------|------------|
| Event-driven workflow | ✅ | ❌ | ❌ |
| Agent reasoning visibility | ❌ | ✅ | ✅ |
| Tool-level modularity | ❌ | ✅ | ✅ |
| Session management | ❌ | ❌ | ✅ |
| Client-server separation | ❌ | ❌ | ✅ |

---

## Installation & Setup

### Prerequisites

- Python 3.10+ (tested with 3.13.4)
- API keys for LlamaCloud and OpenAI/Groq

### Step 1: Create Virtual Environment

**Python 3.13 on macOS (Recommended Method):**
```bash
# Use --copies flag to avoid symlink issues
python3 -m venv .venv --copies

# Activate venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Bootstrap pip if needed
python -m ensurepip --upgrade
```

**Alternative for Python 3.10-3.12:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 2: Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt

# Verify installation
python -c "import streamlit; print(f'Streamlit {streamlit.__version__}')"
```

### Step 3: Configure Environment Variables

Create `.env` file in project root:

```bash
# Required: LlamaCloud API key
LLAMA_CLOUD_API_KEY=your_llama_cloud_key

# LLM Provider Configuration
LLM_PROVIDER=openai  # or groq
LLM_MODEL=gpt-5-nano  # or llama-3.3-70b-versatile for Groq

# API Keys (based on provider)
OPENAI_API_KEY=your_openai_key  # if using OpenAI
GROQ_API_KEY=your_groq_key      # if using Groq

# Optional: LlamaCloud Organization (if required)
LLAMA_CLOUD_PROJECT_ID=your_project_id
LLAMA_CLOUD_ORG_ID=your_org_id
```

### Step 4: Verify Installation

```bash
# Run diagnostic test
.venv/bin/python test_mcp_tools.py

# Expected output: All tests pass ✅
```

---

## Running the Applications

### Option 1: Helper Scripts (Recommended)

```bash
# MCP Client
./run_mcp_client.sh

# Agent App
./run_agent_app.sh

# Plain App
./run_plain_app.sh
```

**Advantages:**
- No need to activate venv
- Works even if venv activation has issues
- Portable across environments

### Option 2: Direct Streamlit Commands

```bash
# Activate venv first
source .venv/bin/activate

# Run your choice
streamlit run streamlit_app_mcp.py    # MCP Client
streamlit run streamlit_app_agent.py  # Agent App
streamlit run streamlit_app.py        # Plain App
```

### Option 3: CLI Mode (Plain Implementation Only)

```bash
# Basic usage
python src/plain/main.py docs/invoice.pdf

# With custom prompt
python src/plain/main.py docs/invoice.pdf --prompt "Extract all fields"

# With template (Manual mode)
python src/plain/main.py docs/invoice.pdf --template docs/template.xlsx
```

### Option 4: Programmatic Usage (Agent Implementation)

```python
from src.agent import create_extraction_agent

# Create agent
agent = create_extraction_agent(mode='auto', verbose=True)

# Run extraction
result = agent.invoke({
    'input': 'Extract all fields from the invoice',
    'file_path': 'docs/invoice.pdf',
    'mode': 'auto'
})

# Check status
if result.get('status') == 'awaiting_approval':
    schema = result['schema']
    # Display schema to user...
    agent.approve_schema(schema)
    result = agent.invoke({'input': 'Continue extraction'})

print('Extracted data:', result.get('extracted_data'))
```

---

## Usage Guide

### Auto Mode (AI-Generated Schema)

**Workflow:**
1. Upload your document (PDF or image)
2. AI analyzes and proposes a JSON schema
3. Review and edit the schema in visual tables
4. Approve or provide feedback for refinement
5. Extract data automatically
6. Export to JSON or Excel

**Step-by-Step:**

1. **Upload Document**
   - Click "Upload PDF or Image"
   - Supported formats: PDF, PNG, JPG, JPEG
   - File uploads to `uploads/` directory

2. **Select Mode & LLM**
   - Mode: Auto
   - Provider: OpenAI or Groq
   - Model: gpt-5-nano (or your preferred model)

3. **Enter Prompt**
   - Default: "Extract all the important information from the document."
   - Custom: "Extract invoice number, customer name, and line items"

4. **Start Extraction**
   - Click "Start Extraction" button
   - Wait for document parsing (5-10 seconds)
   - Wait for schema generation (3-5 seconds)

5. **Review Schema**
   - Schema displayed in editable tables:
     - **Document Fields (Scalar)**: Single-value fields
     - **Document Objects**: Nested structures
     - **Line Items (Arrays)**: Repeating items
   - Edit field names, types, required status
   - Add descriptions

6. **Approve or Reject**
   - **Approve**: Click "✅ Approve & Extract Data"
   - **Reject**: Provide feedback (e.g., "Add TaxAmount field, make Date required")

7. **View Results**
   - Extracted data shown in formatted tables
   - Document headers as key-value pairs
   - Line items as table rows

8. **Export**
   - JSON: Download structured JSON file
   - Excel: 4-sheet workbook with metadata

### Manual Mode (Template-Based)

**Workflow:**
1. Create an Excel template with field names
2. Upload both document and template
3. Schema auto-generated from template
4. Review and approve
5. Extract and export

**Creating Templates:**

1. **Excel File Structure**
   - First row: Field names
   - Use clear, descriptive names
   - Follow naming conventions

2. **Naming Conventions**
   ```
   *InvoiceNo | *Customer | *Date | ItemDescription | ItemQuantity | ItemAmount
   ```
   - `*` prefix = required field
   - `Item` prefix = line item field (will be grouped into arrays)
   - Spaces → underscores in property names

3. **Field Classification**
   - **Document fields**: No prefix or `*` for required
   - **Line item fields**: Prefix with `Item`
   - Example: `ItemDescription` becomes `Description` in line_items array

4. **Type Inference**
   - Automatic from field names:
   - `*No`, `ID`, `Number` → string
   - `Quantity`, `Rate`, `Amount` → number
   - `Date` → string (can specify format in description)

**Using Templates:**

1. **Upload Document**
   - Same as Auto mode

2. **Select Manual Mode**
   - Mode: Manual
   - Template upload field appears

3. **Upload Template**
   - Click "Upload Template (Excel)"
   - Select your .xlsx file
   - Template validation happens automatically

4. **Review Generated Schema**
   - Schema matches template structure
   - Required fields marked from `*` prefix
   - Line items grouped from `Item` prefix

5. **Approve & Extract**
   - Same as Auto mode

**Example Templates:**

**Invoice Template:**
```
*InvoiceNo | *Customer | *InvoiceDate | *DueDate | ItemDescription | ItemQuantity | ItemRate | *ItemAmount
```

**Receipt Template:**
```
*MerchantName | *Date | *Total | ItemName | ItemQuantity | ItemPrice
```

**Purchase Order Template:**
```
*PONumber | *Vendor | *OrderDate | *DeliveryDate | ItemPartNumber | ItemDescription | ItemQuantity | ItemUnitPrice
```

### Iterative Refinement

Both modes support iterative schema refinement:

1. **Initial Schema Generation**
   - AI or template creates first draft

2. **Review & Reject**
   - Find issues or missing fields
   - Reject with specific feedback

3. **Regeneration**
   - Agent incorporates feedback
   - Creates improved schema

4. **Repeat Until Satisfied**
   - Can reject multiple times
   - Each iteration improves schema

**Example Feedback:**
- "Add a TaxAmount field as a required number"
- "Change Date format to YYYY-MM-DD"
- "Split customer name into FirstName and LastName"
- "Add shipping address as a nested object"

---

## MCP Server Integration

The Model Context Protocol (MCP) server exposes the Agent implementation for use with Claude Desktop, Continue.dev, or custom AI assistants.

### Architecture

```
MCP Client (Claude Desktop, etc.)
    ↓
MCP Protocol (JSON-RPC over stdio)
    ↓
mcp_server.py (FastMCP + SessionManager)
    ↓
src/agent/ (LangChain ReAct Agent)
    ↓
ApprovalAgentWrapper (human-in-the-loop)
    ↓
LangChain Tools:
  - parse_document
  - generate_schema_ai
  - generate_schema_template
  - extract_data
  - validate_schema
```

### Configuration for Claude Desktop

Add to `~/.config/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "document-extraction-agent": {
      "command": "python",
      "args": ["/absolute/path/to/document_wf_v1/mcp_server.py"],
      "env": {
        "LLAMA_CLOUD_API_KEY": "your_key_here",
        "OPENAI_API_KEY": "your_key_here",
        "LLM_PROVIDER": "openai",
        "LLM_MODEL": "gpt-5-nano"
      }
    }
  }
}
```

**Important:**
- Use **absolute paths** (not relative)
- Restart Claude Desktop after changes

### Available MCP Tools

#### 1. `start_document_extraction`

Initialize extraction workflow.

**Parameters:**
- `file_path` (string, required): Absolute path to document
- `mode` (string): "auto" or "manual" (default: "auto")
- `prompt` (string): Extraction instructions
- `template_path` (string): Path to Excel template (if mode="manual")
- `llm_provider` (string): "openai" or "groq"
- `llm_model` (string): Model identifier
- `verbose` (boolean): Show agent logs

**Returns:**
```json
{
  "session_id": "abc-123",
  "status": "awaiting_approval",
  "schema": {...},
  "parsed_text": "...",
  "agent_logs": [...]
}
```

#### 2. `approve_extraction_schema`

Approve schema and extract data.

**Parameters:**
- `session_id` (string, required): From start_document_extraction
- `approved_schema` (object, required): Schema to use (may be edited)

**Returns:**
```json
{
  "status": "completed",
  "extracted_data": {...}
}
```

#### 3. `reject_extraction_schema`

Reject and regenerate schema.

**Parameters:**
- `session_id` (string, required): Session ID
- `feedback` (string, required): What to change

**Returns:**
```json
{
  "status": "awaiting_approval",
  "schema": {...}
}
```

### MCP Streamlit Client

A visual interface that consumes the MCP server:

```bash
./run_mcp_client.sh
```

**Features:**
- Visual schema editor (like Agent app)
- MCP session management
- All approval workflow features
- Client-server architecture

**When to Use:**
- Need visual UI with MCP backend
- Want to scale to remote MCP servers
- Prefer client-server separation
- Production with scalable backend

---

## Architecture Deep Dive

### Project Structure

```
document_wf_v1/
├── src/
│   ├── plain/                       # Plain Implementation
│   │   ├── extraction_workflow.py   # Event-driven workflow
│   │   ├── config.py                # API client factories (shared)
│   │   ├── cache_utils.py           # Parse cache (shared)
│   │   ├── template_schema_generator.py  # Template processor (shared)
│   │   ├── excel_export.py          # Export utilities (shared)
│   │   └── main.py                  # CLI entry point
│   │
│   ├── agent/                       # Agent Implementation
│   │   ├── agent.py                 # Agent creation
│   │   ├── executor.py              # ApprovalAgentWrapper
│   │   ├── prompts.py               # System prompts
│   │   └── tools/                   # LangChain tools
│   │       ├── parse_tool.py
│   │       ├── schema_ai_tool.py
│   │       ├── schema_template_tool.py
│   │       ├── extraction_tool.py
│   │       └── validation_tool.py
│   │
│   └── auto_schema_gen.py           # Async CLI runner
│
├── streamlit_app.py                 # Plain UI
├── streamlit_app_agent.py           # Agent UI
├── streamlit_app_mcp.py             # MCP Client UI
├── mcp_server.py                    # MCP Server
├── test_mcp_tools.py                # Diagnostic tool
│
├── run_plain_app.sh                 # Helper scripts
├── run_agent_app.sh
├── run_mcp_client.sh
│
├── docs/                            # Sample files
├── parsed/                          # Cache (gitignored)
├── uploads/                         # Uploads (gitignored)
│
├── requirements.txt
├── .env                             # Config (gitignored)
└── README.md                        # This file
```

### Plain Implementation Architecture

**Event-Driven Workflow:**

```
InputEvent → parse_file()
    ↓
ParsedContent → propose_schema()
    ↓
ProposedSchema → [HUMAN APPROVAL]
    ↓
ApprovedSchema → run_extraction()
    ↓
ExtractedData → Export
```

**Key Components:**

1. **IterativeExtractionWorkflow**
   - LlamaIndex Workflow class
   - Async event handlers
   - State management via WorkflowState

2. **Event Classes**
   - `InputEvent`: Start workflow
   - `ParsedContent`: Document parsed
   - `ProposedSchema`: Schema generated
   - `ApprovedSchema`: User response
   - `RunExtraction`: Run extraction
   - `ExtractedData`: Complete
   - `ProgressEvent`: Updates

3. **Workflow State**
   ```python
   file_path: str | None
   file_content: str | None
   current_schema: dict | None
   current_feedback: str | None
   original_prompt: str | None
   template_path: str | None
   ```

### Agent Implementation Architecture

**Tool-Calling Pattern:**

```
Agent receives task
    ↓
parse_document tool → markdown
    ↓
generate_schema_ai/template tool → schema
    ↓
ApprovalAgentWrapper → [PAUSE]
    ↓
approve_schema() or reject_schema()
    ↓
extract_data tool → structured data
```

**Key Components:**

1. **ApprovalAgentWrapper**
   - Intercepts schema generation
   - Pauses for human approval
   - Resumes with approved schema
   - Handles rejection feedback

2. **Agent State**
   ```python
   pending_schema: Optional[dict]
   approved_schema: Optional[dict]
   waiting_for_approval: bool
   rejection_feedback: Optional[str]
   last_messages: List
   last_error: Optional[str]
   ```

3. **LangChain Tools**
   - Each tool is `BaseTool` subclass
   - Clear descriptions for agent
   - Type-validated inputs
   - Shared services from `config.py`

### Shared Components

Both implementations share:

**1. Configuration (`src/plain/config.py`)**
```python
get_parse_client()     # LlamaParse
get_extract_client()   # LlamaExtract
get_openai_client()    # OpenAI
get_groq_client()      # Groq
get_model_config()     # LLM settings
```

**2. Cache System (`src/plain/cache_utils.py`)**
```python
get_cached_parse()         # Check cache
save_parsed_content()      # Save to cache
clear_cache()              # Clear all
```

**3. Template Processor (`src/plain/template_schema_generator.py`)**
```python
TemplateSchemaGenerator    # Class
generate_schema_from_template()  # Function
```

**4. Export Utilities (`src/plain/excel_export.py`)**
```python
export_to_excel()          # Create .xlsx
```

### Data Flow

**Plain Implementation - Auto Mode:**
```
1. Upload → LlamaParse → Cache (parsed/{file}.md)
2. Cache → OpenAI → JSON Schema
3. User Review → Edit Schema
4. Approve → LlamaExtract → Structured Data
5. Display → Export (JSON/Excel)
```

**Agent Implementation - Auto Mode:**
```
1. Upload → parse_document tool → LlamaParse → Cache
2. Cache → generate_schema_ai tool → OpenAI → Schema
3. ApprovalAgentWrapper → Pause → Return schema
4. User Review → approve_schema() or reject_schema()
5. extract_data tool → LlamaExtract → Data
6. Display → Export
```

**MCP Client:**
```
1. Upload → start_document_extraction (MCP tool)
   ↓
2. MCP Server → Agent → parse + generate schema
   ↓
3. Return session_id + schema
   ↓
4. User Review → approve_extraction_schema (MCP tool)
   ↓
5. MCP Server → Agent → extract data
   ↓
6. Return extracted data
```

---

## Development Guide

### Adding Features

#### Plain Implementation

1. **New Workflow Step**
   ```python
   # In src/plain/extraction_workflow.py
   @step
   async def my_new_step(self, ev: SomeEvent) -> NextEvent:
       # Implementation
       return NextEvent(...)
   ```

2. **Update UI**
   ```python
   # In streamlit_app.py
   if workflow_state == "my_new_state":
       # Display new UI
       pass
   ```

#### Agent Implementation

1. **New Tool**
   ```python
   # Create src/agent/tools/my_tool.py
   from langchain_core.tools import BaseTool
   from pydantic import BaseModel, Field
   from typing import Type

   class MyToolInput(BaseModel):
       param: str = Field(description="...")

   class MyTool(BaseTool):
       name: str = "my_tool"
       description: str = "..."
       args_schema: Type[BaseModel] = MyToolInput

       def _run(self, param: str) -> str:
           # Implementation
           return result
   ```

2. **Register Tool**
   ```python
   # In src/agent/tools/__init__.py
   from src.agent.tools.my_tool import MyTool

   def get_all_tools():
       return [..., MyTool()]
   ```

3. **Update Prompts (if needed)**
   ```python
   # In src/agent/prompts.py
   # Add instructions for new tool
   ```

#### MCP Server

1. **New MCP Tool**
   ```python
   # In mcp_server.py
   @mcp.tool()
   async def my_new_tool(param: str) -> Dict[str, Any]:
       """Tool description for MCP."""
       # Implementation
       return {"result": "..."}
   ```

### Modifying Schema Generation

**Auto Mode (Both implementations):**

Plain: Edit `propose_schema()` in `src/plain/extraction_workflow.py:182`
Agent: Edit `GenerateSchemaAITool` in `src/agent/tools/schema_ai_tool.py`

**Template Mode (Shared):**

Edit `TemplateSchemaGenerator` in `src/plain/template_schema_generator.py`

### Modifying Export Format

Edit `src/plain/excel_export.py` (shared by all):
- Add new sheets
- Change formatting
- Add transformations

### Changing UI Layout

**Plain UI:** `streamlit_app.py`
**Agent UI:** `streamlit_app_agent.py`
**MCP Client UI:** `streamlit_app_mcp.py`

Shared display functions in `streamlit_app.py`:
- `parse_schema_structure()`
- `reconstruct_schema_from_structure()`
- `display_schema_visual()`
- `display_extracted_data()`

### Coding Standards

**Style:**
- 4-space indentation
- Type hints on public functions
- Module-level docstrings
- `snake_case` for functions/modules
- `PascalCase` for classes

**Tools:**
```bash
# Format code
isort src/
black src/

# Check types
mypy src/
```

**Testing:**
```bash
# Run tests
pytest tests/

# With coverage
pytest --cov=src tests/
```

---

## Configuration

### Environment Variables

**Required:**
```bash
LLAMA_CLOUD_API_KEY=xxx    # LlamaCloud (parse + extract)
```

**LLM Provider (choose one):**
```bash
# OpenAI
LLM_PROVIDER=openai
LLM_MODEL=gpt-5-nano
OPENAI_API_KEY=xxx

# OR Groq
LLM_PROVIDER=groq
LLM_MODEL=llama-3.3-70b-versatile
GROQ_API_KEY=xxx
```

**Optional:**
```bash
LLAMA_CLOUD_PROJECT_ID=xxx
LLAMA_CLOUD_ORG_ID=xxx
MCP_MAX_SESSIONS=100
MCP_SESSION_TIMEOUT_MINUTES=60
```

### LLM Model Configuration

**OpenAI Models:**
- `gpt-5-nano` (recommended, fast, cost-effective)
- `gpt-4o` (more capable)
- `gpt-4o-mini` (balanced)

**Groq Models:**
- `moonshotai/kimi-k2-instruct-0905` (recommended)
- `llama-3.3-70b-versatile`
- `mixtral-8x7b-32768`

**Configuration in Code:**
```python
# src/plain/config.py
MODEL_CONFIGS = {
    "gpt-5-nano": {
        "temperature": 0.1,
        "max_tokens": 4000,
        "context_window": 128000
    },
    # ...
}
```

### Cache Management

**Location:** `parsed/` directory

**Clear Cache:**
```bash
rm -rf parsed/

# Or in Python
from src.plain.cache_utils import clear_cache
clear_cache()
```

**Cache Structure:**
```
parsed/
├── invoice_001.pdf.md     # Markdown from LlamaParse
├── receipt_042.png.md
└── ...
```

---

## Dependencies & Compatibility

### Core Dependencies

**LangChain 1.0.2 (Agent Implementation):**
```
langchain==1.0.2
langchain-core==1.0.1
langchain-openai==1.0.1
langchain-community==0.4.0
langgraph==1.0.1
langgraph-prebuilt==1.0.1
```

**LlamaIndex (Plain Implementation):**
```
llama-index-workflows
llama-cloud-services
```

**Shared:**
```
openai>=1.66.3,<2.0.0
streamlit
openpyxl
jsonschema
python-dotenv
fastmcp>=1.2.0
```

### LangChain 1.0 Compatibility

**Compatible Packages:**
- `langchain-groq==1.0.0`
- `langchain-google-genai==3.0.0`
- `langchain-text-splitters==1.0.0`
- `langchain-chroma==1.0.0`

**Incompatible (Not Updated Yet):**
- `langchain-experimental` (requires <1.0.0)
- `langchain-unstructured` (requires <0.4.0)
- `langgraph-supervisor` (requires <0.4.0)

### Base Dependency Constraints

```
openai>=1.66.3,<2.0.0       # Required by llama-index
httpx>=0.24.0,<0.25.0       # Required by supabase
websockets>=11,<13          # Required by realtime
cachetools>=2.0.0,<6.0      # Required by google-auth
```

### Installation Strategies

**Option 1: Clean Installation (Recommended)**
```bash
rm -rf .venv
python3 -m venv .venv --copies
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Option 2: Force Reinstall**
```bash
source .venv/bin/activate
pip uninstall -y langchain langchain-core langchain-openai
pip install --no-cache-dir -r requirements.txt
```

**Option 3: Use pip-tools**
```bash
pip install pip-tools
pip-compile requirements.txt --resolver=backtracking
pip-sync requirements-lock.txt
```

### Verify Installation

```bash
# Check versions
pip list | grep -E "(langchain|langgraph|openai)"

# Check for conflicts
pip check

# Expected: "No broken requirements found."
```

---

## Troubleshooting

### Virtual Environment Issues

**Problem:** `ModuleNotFoundError: No module named 'streamlit'` or `'pip'`

**Root Cause:** Python 3.13 on macOS has symlink issues

**Solution 1: Recreate with --copies (Recommended)**
```bash
rm -rf .venv
python3 -m venv .venv --copies
source .venv/bin/activate
python -m ensurepip --upgrade
pip install -r requirements.txt
```

**Solution 2: Use Helper Scripts**
```bash
./run_mcp_client.sh    # Works without activation
```

**Solution 3: Direct Python Path**
```bash
.venv/bin/python -m streamlit run streamlit_app.py
```

### Module Not Found Errors

**Error:** `ModuleNotFoundError: No module named 'src'`

**Solution:**
```bash
# Run from project root
cd /path/to/document_wf_v1
./run_mcp_client.sh
```

**Error:** `ModuleNotFoundError: No module named 'langchain'`

**Solution:**
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Cache Issues

**Problem:** Document always re-parses

**Solutions:**
- Check filename matches exactly (case-sensitive)
- Verify `parsed/` directory exists
- Clear cache: `rm -rf parsed/`

### Template Issues

**Problem:** Manual mode fails

**Solutions:**
- Ensure .xlsx format (not .xls)
- Headers must be in row 1
- Use "Item" prefix for line items
- Check for special characters in field names

### MCP Client Issues

**Problem:** "MCP operation error"

**Solutions:**
- Check .env file has all required keys
- Verify file paths are absolute
- Check network connectivity
- Review terminal logs

**Problem:** Agent did not pause for approval

**Solutions:**
- Check file format (PDF, PNG, JPG, JPEG only)
- Verify API keys are valid
- Check agent logs (enable verbose mode)
- Ensure document uploaded correctly

### Agent-Specific Issues

**Problem:** Agent not stopping for approval

**Debug:**
```python
# Enable verbose mode
agent = create_extraction_agent(mode='auto', verbose=True)
```

**Check:**
- Tool names contain "schema" and "generate"
- Tool response has valid JSON
- ApprovalAgentWrapper detects schema

**Problem:** Tool call failures

**Solutions:**
- Check tool implementation in `src/agent/tools/`
- Verify shared services from `src/plain/config.py`
- Check API keys in `.env`
- Review agent logs in UI

### Python Version Issues

**Problem:** Python 3.13 venv broken

**Solution:** Always use `--copies` flag
```bash
python3 -m venv .venv --copies
```

**Problem:** Need different Python version

**Solutions:**
```bash
# Use specific version
/opt/homebrew/bin/python3.12 -m venv .venv --copies

# Or use pyenv
pyenv install 3.12.0
pyenv local 3.12.0
python -m venv .venv
```

### Common Error Messages

**"ERROR: pip's dependency resolver..."**
- Follow Clean Installation strategy above

**"ImportError: cannot import name 'X' from 'langchain_core'"**
- Package expects different langchain-core version
- Update package or downgrade stack

**"No module named 'langchain_experimental'"**
- Package not compatible with LangChain 1.0
- Use alternative or downgrade

### Diagnostic Commands

```bash
# Check setup
.venv/bin/python test_mcp_tools.py

# List installed packages
pip list | grep -E "(langchain|streamlit)"

# Check for conflicts
pip check

# Verify imports
.venv/bin/python -c "import streamlit, langchain, fastmcp; print('OK')"

# Check Python version
python3 --version

# Check venv structure
ls -la .venv/lib/
```

---

## Testing

### Quick Tests

**Test MCP Tools:**
```bash
.venv/bin/python test_mcp_tools.py
```

**Test Plain Implementation:**
```bash
streamlit run streamlit_app.py
# Upload docs/IMG_4693.JPEG
# Click "Start Extraction"
```

**Test Agent Implementation:**
```bash
streamlit run streamlit_app_agent.py
# Upload a document
# Enable "Show agent reasoning"
```

**Test MCP Client:**
```bash
./run_mcp_client.sh
# Upload a document
# Check MCP session tracking
```

### CLI Testing

**Plain Implementation:**
```bash
# Auto mode
python src/plain/main.py docs/invoice.pdf

# With prompt
python src/plain/main.py docs/invoice.pdf --prompt "Extract vendor and items"

# Manual mode
python src/plain/main.py docs/invoice.pdf --template docs/template.xlsx
```

### Programmatic Testing

**Agent:**
```python
from src.agent import create_extraction_agent

agent = create_extraction_agent(mode='auto', verbose=True)
result = agent.invoke({
    'input': 'Extract all fields',
    'file_path': 'docs/invoice.pdf'
})

assert result['status'] == 'awaiting_approval'
assert 'schema' in result
```

**MCP Server:**
```python
import asyncio
from mcp_server import start_document_extraction

async def test():
    result = await start_document_extraction.fn(
        file_path='docs/invoice.pdf',
        mode='auto'
    )
    assert result['status'] == 'awaiting_approval'

asyncio.run(test())
```

### Cache Testing

```bash
# First run - parses document
python src/plain/main.py docs/invoice.pdf

# Second run - uses cache (faster)
python src/plain/main.py docs/invoice.pdf

# Clear and test
rm -rf parsed/
python src/plain/main.py docs/invoice.pdf
```

### Unit Tests

**Structure:**
```
tests/
├── plain/
│   ├── test_workflow.py
│   └── test_schema_generator.py
├── agent/
│   ├── test_tools.py
│   └── test_executor.py
└── conftest.py
```

**Run Tests:**
```bash
# All tests
pytest tests/

# Specific test
pytest tests/plain/test_workflow.py

# With coverage
pytest --cov=src tests/

# With verbose
pytest -v tests/
```

---

## Examples & Workflows

### Example 1: Invoice Extraction (Auto Mode)

```bash
# Start app
./run_mcp_client.sh

# In browser:
# 1. Upload: docs/invoice.pdf
# 2. Mode: Auto
# 3. Provider: OpenAI / gpt-5-nano
# 4. Prompt: "Extract invoice header and line items"
# 5. Click: Start Extraction
# 6. Review schema, approve
# 7. Download Excel
```

**Expected Schema:**
```json
{
  "type": "object",
  "properties": {
    "InvoiceNo": {"type": "string"},
    "Customer": {"type": "string"},
    "Date": {"type": "string"},
    "Total": {"type": "number"},
    "line_items": {
      "type": "array",
      "items": {
        "properties": {
          "Description": {"type": "string"},
          "Quantity": {"type": "number"},
          "Rate": {"type": "number"},
          "Amount": {"type": "number"}
        }
      }
    }
  }
}
```

### Example 2: Template-Based Extraction

**1. Create Template (invoice_template.xlsx):**
```
*InvoiceNo | *Customer | *Date | *Total | ItemDescription | ItemQuantity | ItemRate | ItemAmount
```

**2. Run Extraction:**
```bash
./run_plain_app.sh

# In browser:
# 1. Upload: invoice.pdf
# 2. Mode: Manual
# 3. Upload: invoice_template.xlsx
# 4. Click: Start Extraction
# 5. Schema auto-generated from template
# 6. Approve and extract
```

### Example 3: Batch Processing

```python
from src.agent import create_extraction_agent
from pathlib import Path

# Create agent once
agent = create_extraction_agent(mode='auto', verbose=False)

# Process multiple documents
documents = Path('docs/').glob('*.pdf')

for doc in documents:
    # Start extraction
    result = agent.invoke({
        'input': 'Extract all fields',
        'file_path': str(doc)
    })

    # Auto-approve (for trusted documents)
    if result['status'] == 'awaiting_approval':
        schema = result['schema']
        agent.approve_schema(schema)
        result = agent.invoke({'input': 'Continue'})

    # Save results
    with open(f'{doc.stem}_extracted.json', 'w') as f:
        json.dump(result['extracted_data'], f, indent=2)
```

### Example 4: Custom Schema Refinement

```python
# Start extraction
result = agent.invoke({
    'input': 'Extract invoice data',
    'file_path': 'invoice.pdf'
})

# Get initial schema
schema = result['schema']

# Add custom fields
schema['properties']['TaxAmount'] = {'type': 'number'}
schema['properties']['PaymentTerms'] = {'type': 'string'}
schema['required'].extend(['TaxAmount'])

# Approve with modifications
agent.approve_schema(schema)
result = agent.invoke({'input': 'Continue'})
```

### Example 5: MCP Integration with Claude Desktop

**Conversation Flow:**

```
User: "Extract data from invoice.pdf"

Claude: [Calls start_document_extraction]
        "I've parsed the invoice and generated a schema.
         Here's what I found:
         - InvoiceNo (string, required)
         - Customer (string, required)
         - Date (string)
         - line_items (array)

         Does this look correct?"

User: "Add TaxAmount field"

Claude: [Calls reject_extraction_schema with feedback]
        "I've updated the schema to include:
         - TaxAmount (number, required)

         Is this better?"

User: "Yes, approve"

Claude: [Calls approve_extraction_schema]
        "Extraction complete! Here's the data:
         InvoiceNo: INV-001
         Customer: Acme Corp
         TaxAmount: 125.50
         ..."
```

### Example 6: Error Recovery

```python
from src.agent import create_extraction_agent

agent = create_extraction_agent(mode='auto', verbose=True)

try:
    result = agent.invoke({
        'input': 'Extract from invoice.pdf',
        'file_path': 'invoice.pdf'
    })

    if result['status'] == 'error':
        print(f"Error: {result['error']}")

        # Retry with different approach
        result = agent.invoke({
            'input': 'Extract just the header fields',
            'file_path': 'invoice.pdf'
        })

except Exception as e:
    print(f"Fatal error: {e}")
    # Log and alert
```

---

## Cache Management

### Cache Structure

**Location:** `parsed/` directory (project root)

**Format:**
```
parsed/
├── invoice_001.pdf.md      # Markdown from LlamaParse
├── receipt_042.png.md
├── document.jpeg.md
└── ...
```

**Cache File Contents:**
- Markdown text extracted from PDF/image
- Used by both Plain and Agent implementations
- Filename-based lookup

### Cache Operations

**Clear All Cache:**
```bash
rm -rf parsed/

# Or in Python
from src.plain.cache_utils import clear_cache
clear_cache()
```

**Check Cache:**
```python
from src.plain.cache_utils import get_cached_parse

content = get_cached_parse("invoice.pdf")
if content:
    print("Cache hit!")
else:
    print("Cache miss - will parse")
```

**Manual Cache:**
```python
from src.plain.cache_utils import save_parsed_content

save_parsed_content("document.pdf", "# Markdown content...")
```

### Cache Benefits

- **Speed**: Parsing takes 5-10 seconds, cache is instant
- **Cost**: Avoids repeated API calls to LlamaParse
- **Iterations**: Edit schema without re-parsing
- **Consistency**: Same parse result across attempts

### When to Clear Cache

- Document content changed
- Testing different parsing parameters
- Cache corruption suspected
- Storage cleanup needed

---

## Excel Export Format

### Sheet Structure

Excel exports contain **4 sheets**:

#### 1. Metadata
- Export timestamp
- Document filename
- Field counts
- Schema version

#### 2. Document Headers
- Flattened key-value pairs
- All scalar document fields
- Nested objects flattened with dot notation

Example:
```
Field               | Value
--------------------|------------------
InvoiceNo           | INV-001
Customer            | Acme Corp
Date                | 2025-01-15
seller.name         | ABC Company
seller.address      | 123 Main St
```

#### 3. Line Items
- Tabular data
- All repeating items
- One row per item

Example:
```
Description | Quantity | Rate   | Amount
------------|----------|--------|--------
Widget A    | 5        | 50.00  | 250.00
Widget B    | 10       | 25.00  | 250.00
```

#### 4. Raw Data
- Complete JSON
- All fields preserved
- Nested structure intact
- For reference and debugging

### Formatting

- Auto-adjusted column widths
- Headers in bold
- Professional styling
- Date formatting where detected

### Customization

Edit `src/plain/excel_export.py` to:
- Add new sheets
- Change styling
- Add formulas
- Modify transformations

---

## File Locations

### User Data
- **Uploaded files**: `uploads/` (gitignored)
- **Cached parses**: `parsed/` (gitignored)
- **Excel templates**: `docs/` (versioned)

### Source Code
- **Core workflows**: `src/plain/`, `src/agent/`
- **Web UIs**: `streamlit_app*.py`
- **MCP server**: `mcp_server.py`
- **Tests**: `tests/`

### Configuration
- **Environment**: `.env` (gitignored)
- **Requirements**: `requirements.txt`
- **Git rules**: `.gitignore`

---

## Which Implementation Should I Use?

### Use Plain Implementation When:
✅ You want proven stability for production
✅ Event-driven architecture fits your mental model
✅ You don't need visibility into decision-making
✅ You prefer simpler execution flow
✅ You're comfortable with LlamaIndex Workflows

### Use Agent Implementation When:
✅ You need transparent reasoning and debugging
✅ You want to see each step as explicit tool call
✅ You plan to extend with custom tools
✅ You're building on LangChain ecosystem
✅ You want fine-grained control over behavior
✅ You need to explain extraction logic to stakeholders

### Use MCP Client When:
✅ You want visual UI with MCP backend
✅ You need MCP session management
✅ You plan to scale to remote servers
✅ You want client-server separation
✅ You prefer scalable production architecture

### Migration Strategy

All implementations are production-ready:

1. **Run in parallel** (already supported)
2. **Compare outputs** with same documents
3. **Switch users gradually** based on use case
4. **Maintain multiple** for different scenarios

No code changes needed - all share core services.

---

## Support

For issues:

**General:**
- Check this README first
- Run diagnostic: `.venv/bin/python test_mcp_tools.py`
- Review terminal logs
- Check browser console (F12)

**Plain Implementation:**
- Check `src/plain/extraction_workflow.py`
- Review event handlers

**Agent Implementation:**
- Check `src/agent/executor.py`
- Enable verbose mode
- Review agent logs

**MCP Client:**
- Check MCP server logs
- Verify session management
- Review tool invocations

**Report Issues:**
Include:
- Python version
- OS version
- Error message (full traceback)
- Steps to reproduce
- Document type being processed

---

## License

See LICENSE file for details.

---

## Summary

This is a complete document extraction system with three implementations (Plain, Agent, MCP), supporting both AI-generated and template-based schemas, with human-in-the-loop approval, caching, and rich export options.

**Quick Start:**
```bash
python3 -m venv .venv --copies
source .venv/bin/activate
pip install -r requirements.txt
echo "LLAMA_CLOUD_API_KEY=your_key" > .env
echo "OPENAI_API_KEY=your_key" >> .env
./run_mcp_client.sh
```

**Get Help:**
- Run diagnostics: `.venv/bin/python test_mcp_tools.py`
- Check logs in terminal
- Review error messages carefully
- Use helper scripts: `./run_*.sh`

**Key Files:**
- `streamlit_app_mcp.py` - MCP Client (recommended)
- `streamlit_app_agent.py` - Agent with reasoning
- `streamlit_app.py` - Plain stable version
- `mcp_server.py` - MCP Server for Claude Desktop
- `test_mcp_tools.py` - Diagnostic tool

---

**End of Documentation**

This README consolidates all project documentation in one comprehensive file. For specific issues, use the Table of Contents to jump to relevant sections.

**Version:** 1.0
**Last Updated:** October 28, 2025
**Total Lines:** ~2,500
