# Document Extraction - Parallel Implementations

This project now supports **two parallel implementations** for document extraction:

1. **Plain Implementation** (`/src/plain/`) - Original LlamaIndex Workflows
2. **Agent Implementation** (`/src/agent/`) - New LangChain 1.0.2 ReAct Agent

Both implementations share the same core functionality but use different orchestration approaches.

---

## Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Setup Environment

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key
LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key
LLAMA_CLOUD_PROJECT_ID=your_project_id
LLAMA_CLOUD_ORG_ID=your_org_id
```

---

## Running the Applications

### Plain Implementation (LlamaIndex Workflows)

```bash
streamlit run streamlit_app.py
```

**Features:**
- Event-driven workflow
- Async execution
- Simple, linear flow
- Proven and stable

### Agent Implementation (LangChain ReAct)

```bash
streamlit run streamlit_app_agent.py
```

**Features:**
- Tool-calling agent
- Visible reasoning process
- Flexible decision-making
- Agent transparency (view tool calls)

---

## Architecture Comparison

### Plain Implementation (`/src/plain/`)

```
Workflow Steps:
  Parse → Generate Schema → [HUMAN APPROVAL] → Extract → Export

Implementation:
  - LlamaIndex Workflows
  - Event-based state machine
  - ProposedSchema/ApprovedSchema events
  - Async handlers
```

**Files:**
- `src/plain/extraction_workflow.py` - Main workflow
- `src/plain/config.py` - Service clients
- `src/plain/cache_utils.py` - Parsing cache
- `src/plain/template_schema_generator.py` - Template processor
- `src/plain/excel_export.py` - Export utilities

### Agent Implementation (`/src/agent/`)

```
Agent Flow:
  Tools → ReAct Agent → [HUMAN APPROVAL via Executor] → Continue

Implementation:
  - LangChain 1.0.2 ReAct Agent
  - Tool-calling pattern
  - Custom ApprovalAgentExecutor
  - Transparent reasoning
```

**Files:**
- `src/agent/tools/` - 5 LangChain tools (parse, schema_ai, schema_template, extract, validate)
- `src/agent/agent.py` - Agent creation
- `src/agent/executor.py` - Custom executor with approval
- `src/agent/prompts.py` - ReAct prompts

---

## Tool Descriptions (Agent Implementation)

The agent has access to these tools:

### 1. `parse_document`
- **Purpose:** Parse PDF/image to markdown
- **Input:** `file_path` (absolute path)
- **Output:** Parsed markdown text
- **Uses:** LlamaParse (from `/src/plain/`)

### 2. `generate_schema_ai`
- **Purpose:** Generate JSON schema using AI
- **Input:** `document_text`, `prompt`
- **Output:** JSON Schema Draft 7
- **Uses:** OpenAI GPT-4 (from `/src/plain/config.py`)

### 3. `generate_schema_template`
- **Purpose:** Generate schema from Excel template
- **Input:** `template_path` (.xlsx file)
- **Output:** JSON Schema
- **Uses:** TemplateSchemaGenerator (from `/src/plain/`)

### 4. `extract_data`
- **Purpose:** Extract structured data with schema
- **Input:** `file_path`, `schema` (approved)
- **Output:** Extracted data dictionary
- **Uses:** LlamaExtract (from `/src/plain/config.py`)

### 5. `validate_schema`
- **Purpose:** Validate JSON schema format
- **Input:** `schema` dict
- **Output:** `{valid: bool, errors: list}`
- **Uses:** jsonschema library

---

## Shared Components

Both implementations share:

- **Cache System** (`parsed/` directory)
  - Caches LlamaParse results
  - Speeds up repeated extractions
  - Keyed by filename

- **Export Utilities** (`src/plain/excel_export.py`)
  - Excel export (4 sheets)
  - JSON export
  - Works with both implementations

- **Configuration** (`src/plain/config.py`)
  - API client factories
  - Environment variable management
  - Shared by tools

---

## When to Use Which Implementation?

### Use Plain Implementation When:
✅ You want a proven, stable workflow
✅ You prefer event-driven architecture
✅ You don't need to see agent reasoning
✅ You want simpler execution flow

### Use Agent Implementation When:
✅ You want transparent decision-making
✅ You need to debug extraction logic
✅ You prefer tool-based modularity
✅ You want to extend with more tools
✅ You're building on LangChain ecosystem

---

## Feature Parity

Both implementations support:

| Feature | Plain | Agent |
|---------|-------|-------|
| Auto mode (AI schema) | ✅ | ✅ |
| Manual mode (template) | ✅ | ✅ |
| Human approval workflow | ✅ | ✅ |
| Schema editing | ✅ | ✅ |
| Dual mode support | ✅ | ✅ |
| Cache system | ✅ | ✅ |
| Excel export | ✅ | ✅ |
| JSON export | ✅ | ✅ |
| Agent reasoning visibility | ❌ | ✅ |
| Tool-level modularity | ❌ | ✅ |

---

## Development

### Adding a New Tool (Agent Implementation)

1. Create tool file in `src/agent/tools/`:

```python
from langchain_core.tools import BaseTool
from pydantic import Field, BaseModel
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

2. Register in `src/agent/tools/__init__.py`:

```python
from src.agent.tools.my_tool import MyTool

def get_all_tools():
    return [
        # ... existing tools
        MyTool(),
    ]
```

3. Tool is now available to agent automatically!

### Modifying Workflow (Plain Implementation)

Edit `src/plain/extraction_workflow.py`:

```python
@step
async def my_new_step(self, ev: SomeEvent) -> NextEvent:
    # Add workflow logic
    return NextEvent(...)
```

---

## Testing

### Test Plain Implementation

```bash
# Run Streamlit app
streamlit run streamlit_app.py

# Or use CLI
python src/plain/main.py docs/invoice.pdf
```

### Test Agent Implementation

```bash
# Run Streamlit app
streamlit run streamlit_app_agent.py

# Or test programmatically
python -c "
from src.agent import create_extraction_agent
agent = create_extraction_agent(mode='auto')
result = agent.invoke({
    'input': 'Extract from test.pdf',
    'file_path': 'test.pdf'
})
print(result)
"
```

---

## Troubleshooting

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'src.plain'`

**Solution:** Run from project root:
```bash
cd /path/to/document_wf_v1
streamlit run streamlit_app.py
```

### Agent Not Stopping for Approval

**Check:** Ensure the executor detects schema generation:
- Look for tools with "schema" and "generate" in name
- Verify observation contains valid JSON

**Debug:** Enable `verbose=True` in agent creation:
```python
agent = create_extraction_agent(mode='auto', verbose=True)
```

### Cache Issues

**Clear cache:**
```bash
rm -rf parsed/
```

Or use UI button in sidebar.

---

## Dependencies

### LangChain 1.0.x (Agent Implementation)
```
langchain==1.0.2
langchain-openai==1.0.1
langchain-core==1.0.1
langchain-community==0.4.0
```

### LlamaIndex (Plain Implementation)
```
llama-index-workflows
llama-cloud-services
```

### Shared
```
openai
streamlit
openpyxl
jsonschema
python-dotenv
```

---

## Migration Path

To migrate from Plain to Agent:

1. **Run both in parallel** (already set up)
2. **Compare outputs** with same documents
3. **Gradually switch** by replacing Plain app with Agent
4. **Deprecate Plain** when confident in Agent

No code changes needed for migration - both UIs are compatible.

---

## License

Same as original project.

---

## Support

For issues:
- Plain implementation: Check `src/plain/extraction_workflow.py`
- Agent implementation: Check `src/agent/executor.py`
- Both: Check Streamlit logs and agent reasoning output
