# Document Extraction Workflow

An intelligent document extraction system with AI-powered schema generation and template-based extraction. This project provides **two parallel implementations** with different orchestration approaches.

## 🚀 Two Implementations Available

This project supports two parallel implementations for document extraction:

### 1. **Plain Implementation** (Stable & Event-Driven)
- **Technology:** LlamaIndex Workflows
- **Architecture:** Event-driven state machine
- **UI:** `streamlit run streamlit_app.py`
- **Best for:** Production use, proven stability, simple execution flow

### 2. **Agent Implementation** (Transparent & Tool-Based)
- **Technology:** LangChain 1.0.2 ReAct Agent
- **Architecture:** Tool-calling pattern with visible reasoning
- **UI:** `streamlit run streamlit_app_agent.py`
- **Best for:** Debugging, transparency, extending with custom tools

Both implementations share the same core functionality (dual mode, caching, export) but use different orchestration approaches. See [README_AGENT.md](README_AGENT.md) for detailed comparison.

---

## Features (Both Implementations)

- **Dual Mode Operation**
  - 🤖 **Auto Mode**: AI generates schema automatically from document analysis
  - 📋 **Manual Mode**: Use Excel templates for consistent structured extraction

- **Interactive Web UI** (Streamlit)
  - Visual schema editor with editable tables
  - Real-time progress tracking
  - Approval/rejection workflow with feedback
  - Multi-format export (JSON, Excel)

- **Smart Caching**
  - Automatic caching of parsed documents
  - Instant schema iterations without re-parsing
  - Filename-based cache lookup

- **Excel Export**
  - Multi-sheet workbooks (Metadata, Headers, Line Items, Raw Data)
  - Professional formatting with auto-adjusted columns
  - Perfect for business users and further analysis

- **Flexible Template System**
  - Create custom Excel templates with your field names
  - Automatic line item detection
  - Type inference from field names
  - Reusable across similar documents

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Running the Application

**Choose Your Implementation:**

**Option 1: Plain Implementation (LlamaIndex Workflows)**
```bash
# Streamlit UI (Recommended)
streamlit run streamlit_app.py

# Command Line
python src/plain/main.py docs/invoice.pdf
```

**Option 2: Agent Implementation (LangChain ReAct)**
```bash
# Streamlit UI with Agent Reasoning
streamlit run streamlit_app_agent.py

# Programmatic Use
python -c "
from src.agent import create_extraction_agent
agent = create_extraction_agent(mode='auto')
result = agent.invoke({
    'input': 'Extract from docs/invoice.pdf',
    'file_path': 'docs/invoice.pdf'
})
print(result)
"
```

> 💡 **New to the project?** Start with the Plain Implementation (`streamlit_app.py`) for a stable, proven workflow. Switch to the Agent Implementation when you need transparency or want to extend with custom tools.

## Usage

### Auto Mode

1. Upload your document (PDF or image)
2. AI analyzes and proposes a JSON schema
3. Review and edit the schema in visual tables
4. Approve or provide feedback for refinement
5. Extract data automatically
6. Export to JSON or Excel

### Manual Mode

1. Create an Excel template with field names in row 1:
   ```
   *InvoiceNo | *Customer | *Date | ItemDescription | ItemQuantity | ItemRate | ItemAmount
   ```
2. Upload both document and template
3. Schema generated from template headers
4. Review and approve schema
5. Extract and export data

**Template Conventions:**
- Prefix with `*` for required fields
- Prefix with `Item` for line item fields (e.g., ItemQuantity)
- Spaces converted to underscores in property names

## Architecture Overview

### Project Structure

```
src/
├── plain/                       # Plain Implementation (LlamaIndex Workflows)
│   ├── extraction_workflow.py   # Event-driven workflow orchestration
│   ├── config.py                # Shared API client factories
│   ├── cache_utils.py           # Parsing cache (shared by both)
│   ├── template_schema_generator.py # Excel → JSON schema
│   ├── excel_export.py          # Export utilities (shared by both)
│   └── main.py                  # CLI entry point
│
├── agent/                       # Agent Implementation (LangChain ReAct)
│   ├── agent.py                 # Agent creation and setup
│   ├── executor.py              # Custom approval wrapper
│   ├── prompts.py               # System prompts for agent
│   └── tools/                   # LangChain tools
│       ├── parse_tool.py        # Document parsing tool
│       ├── schema_ai_tool.py    # AI schema generation
│       ├── schema_template_tool.py # Template schema generation
│       ├── extraction_tool.py   # Data extraction tool
│       └── validation_tool.py   # Schema validation tool
│
└── auto_schema_gen.py           # Async CLI runner

streamlit_app.py                 # Plain implementation UI
streamlit_app_agent.py           # Agent implementation UI
docs/                            # Sample documents and templates
parsed/                          # Shared cache directory
```

### Plain Implementation Workflow

This describes the **Plain Implementation** (LlamaIndex Workflows). For the **Agent Implementation**, see [README_AGENT.md](README_AGENT.md).

**Workflow Steps:**

```
1. Parse Document (LlamaParse)
   ↓
2. Generate Schema
   ├─ Auto: OpenAI analyzes document
   └─ Manual: Read from Excel template
   ↓
3. User Approval Loop
   ├─ Approve → Continue
   └─ Reject → Provide feedback → Regenerate
   ↓
4. Extract Data (LlamaExtract)
   ↓
5. Display & Export
   ├─ Streamlit UI tables
   ├─ JSON download
   └─ Excel download (4 sheets)
```

## Configuration

Create a `.env` file in the project root:

```env
# Required API Keys
OPENAI_API_KEY=your-openai-key
LLAMA_CLOUD_API_KEY=your-llama-cloud-key
LLAMA_CLOUD_PROJECT_ID=your-project-id
LLAMA_CLOUD_ORG_ID=your-org-id

# Optional: Model Configuration
OPENAI_MODEL=gpt-5-nano
```

## Cache Management

Parsed documents are cached in the `parsed/` directory:

```bash
# Clear all cached documents
rm -rf parsed/

# View cache contents
ls -la parsed/
```

Cache enables fast schema iterations - edit and re-extract without re-parsing the document.

## Excel Export Format

Exported Excel files contain 4 sheets:

1. **Metadata**: Export timestamp, field counts, document info
2. **Document Headers**: Flattened key-value pairs for document-level fields
3. **Line Items**: Tabular data with all line item fields
4. **Raw Data**: Complete JSON for reference and debugging

## Example Templates

**Invoice Template** (`docs/Invoice Template.xlsx`):
```
*InvoiceNo | *Customer | *InvoiceDate | *DueDate | ItemDescription | ItemQuantity | ItemRate | *ItemAmount
```

**Receipt Template**:
```
*MerchantName | *Date | *Total | ItemName | ItemQuantity | ItemPrice
```

Create templates for:
- Invoices
- Receipts
- Purchase Orders
- Bills of Lading
- Contracts
- Forms

## Development

### Project Structure

- `src/extraction_workflow.py`: Core workflow with event handlers
- `src/config.py`: Centralized configuration
- `src/cache_utils.py`: Document parsing cache
- `src/template_schema_generator.py`: Excel → JSON schema conversion
- `src/excel_export.py`: Data export functionality
- `streamlit_app.py`: Web UI interface

### Key Technologies

- **LlamaIndex Workflows**: Event-driven workflow orchestration
- **LlamaParse**: Advanced document parsing with OCR
- **LlamaExtract**: Structured data extraction
- **OpenAI GPT**: Schema generation and validation
- **Streamlit**: Interactive web interface
- **OpenPyXL**: Excel file creation

## Which Implementation Should I Use?

### Use Plain Implementation When:
✅ You want a proven, stable workflow
✅ You prefer event-driven architecture
✅ You don't need to see internal decision-making
✅ You want simpler execution flow
✅ You're deploying to production

### Use Agent Implementation When:
✅ You want transparent decision-making (see agent reasoning)
✅ You need to debug extraction logic step-by-step
✅ You prefer tool-based modularity
✅ You want to extend with custom tools easily
✅ You're building on LangChain ecosystem
✅ You want human-in-the-loop approval with full context

### Feature Parity

Both implementations support the same core features:

| Feature | Plain | Agent |
|---------|-------|-------|
| Auto mode (AI schema) | ✅ | ✅ |
| Manual mode (template) | ✅ | ✅ |
| Human approval workflow | ✅ | ✅ |
| Schema editing | ✅ | ✅ |
| Cache system | ✅ | ✅ |
| Excel export | ✅ | ✅ |
| JSON export | ✅ | ✅ |
| Agent reasoning visibility | ❌ | ✅ |
| Tool-level modularity | ❌ | ✅ |

For detailed comparison and agent architecture, see [README_AGENT.md](README_AGENT.md).

## Development

### Adding Features

**For Plain Implementation:**
- See `CLAUDE.md` for detailed workflow development
- Modify `src/plain/extraction_workflow.py` to add workflow steps
- Edit `src/plain/config.py` for configuration changes
- Update `streamlit_app.py` for UI modifications

**For Agent Implementation:**
- See `README_AGENT.md` for agent development guide
- Add new tools in `src/agent/tools/` directory
- Modify `src/agent/prompts.py` to change agent behavior
- Update `streamlit_app_agent.py` for agent UI
- Tools automatically become available to agent once registered

**Shared Components:**
Both implementations share:
- `src/plain/cache_utils.py` - Caching system
- `src/plain/excel_export.py` - Export utilities
- `src/plain/template_schema_generator.py` - Template processing
- `src/plain/config.py` - API client factories

## Troubleshooting

**Import Errors:**
```
ModuleNotFoundError: No module named 'src'
```
→ Run from project root, not from src/ directory

**Cache Not Working:**
```
Document re-parses every time
```
→ Check filename matches exactly (case-sensitive)

**Template Errors:**
```
Manual mode: schema generation failed
```
→ Ensure .xlsx format, headers in row 1, "Item" prefix for line items

## License

See LICENSE file for details.

## Support

For issues and questions:
1. **Plain Implementation:** Check `CLAUDE.md` for detailed workflow documentation
2. **Agent Implementation:** Check `README_AGENT.md` for agent architecture and tools
3. **Templates:** Review `TEMPLATE_USAGE.md` for template help
4. **Development:** Check `AGENTS.md` for coding guidelines
5. Check issue tracker for known problems

## Additional Documentation

- **`CLAUDE.md`** - Detailed plain implementation workflow guide
- **`README_AGENT.md`** - Comprehensive agent implementation comparison
- **`AGENTS.md`** - Development guidelines for both implementations
- **`QUICKSTART.md`** - Quick start guide for getting up and running
- **`TEMPLATE_USAGE.md`** - Excel template creation guide
