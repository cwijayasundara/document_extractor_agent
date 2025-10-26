# Quick Start Guide - LangChain Agent Implementation

## ✅ Implementation Complete!

Your project now has **two parallel document extraction implementations**:

### 1. Plain (Original) - `streamlit_app.py`
- LlamaIndex Workflows
- Event-driven architecture
- Proven and stable

### 2. Agent (New) - `streamlit_app_agent.py`
- LangChain 1.0.2 ReAct Agent
- Tool-calling architecture
- Transparent reasoning

---

## 🚀 Run the Agent Implementation

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `langchain==1.0.2`
- `langchain-openai==1.0.1`
- `langchain-core==1.0.1`
- `langchain-community==0.4.0`
- Plus existing dependencies

### Step 2: Configure Environment

Ensure your `.env` file has:

```env
OPENAI_API_KEY=your_openai_api_key
LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key
LLAMA_CLOUD_PROJECT_ID=your_project_id
LLAMA_CLOUD_ORG_ID=your_org_id
```

### Step 3: Run the Agent App

```bash
streamlit run streamlit_app_agent.py
```

The app will open at `http://localhost:8501`

---

## 📋 How to Use

### Auto Mode (AI-Generated Schema)

1. **Upload Document**
   - Click "Upload PDF or Image"
   - Select your document (PDF, PNG, JPG, JPEG)

2. **Enter Prompt** (optional)
   - Default: "Extract all the important information from the invoice."
   - Customize for specific needs

3. **Run Agent**
   - Click "🤖 Run Agent Extraction"
   - Agent will:
     - Parse document to markdown
     - Generate schema using GPT-4
     - **PAUSE for your approval**

4. **Review & Approve Schema**
   - View proposed schema in visual tables
   - Edit field names, types, or descriptions
   - Click "✅ Approve & Extract Data"

5. **Get Results**
   - Agent extracts data using approved schema
   - View extracted data in structured format
   - Download as JSON or Excel

### Manual Mode (Template-Based)

1. **Upload Document** (same as Auto)

2. **Upload Excel Template**
   - Create .xlsx with field names in row 1
   - Prefix line item fields with "Item" (e.g., ItemQuantity)

3. **Run Agent**
   - Click "🤖 Run Agent with Template"
   - Agent generates schema from template
   - Review and approve

4. **Extract & Export** (same as Auto)

---

## 🤖 Agent Features

### Visible Reasoning

Check the box **"Show agent reasoning & tool calls"** to see:
- Which tools the agent calls
- Tool inputs and outputs
- Agent's decision-making process

Example log:
```
Step 1: Tool: parse_document
Input: {"file_path": "/path/to/invoice.pdf"}
Output: Parsed markdown text...

Step 2: Tool: generate_schema_ai
Input: {"document_text": "...", "prompt": "..."}
Output: {"type": "object", "properties": {...}}

APPROVAL: Schema approved by user

Step 3: Tool: extract_data
Input: {"file_path": "...", "schema": {...}}
Output: Extracted data dictionary
```

### Human-in-the-Loop Approval

The agent **automatically pauses** after generating a schema:
- ✅ You can edit the schema
- ✅ You can approve to continue
- ✅ You can reject with feedback
- ✅ Agent will regenerate if rejected

---

## 🔧 Architecture Overview

```
src/
├── plain/                    # Original implementation
│   ├── extraction_workflow.py
│   ├── config.py
│   ├── cache_utils.py
│   ├── template_schema_generator.py
│   └── excel_export.py
│
└── agent/                    # NEW: LangChain implementation
    ├── tools/                # 5 LangChain tools
    │   ├── parse_tool.py
    │   ├── schema_ai_tool.py
    │   ├── schema_template_tool.py
    │   ├── extraction_tool.py
    │   └── validation_tool.py
    ├── agent.py              # ReAct agent creation
    ├── executor.py           # Custom executor with approval
    └── prompts.py            # System prompts

streamlit_app.py              # Plain UI (original)
streamlit_app_agent.py        # Agent UI (NEW)
```

### Tool Details

| Tool | Purpose | Input | Output |
|------|---------|-------|--------|
| `parse_document` | Parse PDF/image | `file_path` | Markdown text |
| `generate_schema_ai` | AI schema gen | `document_text`, `prompt` | JSON schema |
| `generate_schema_template` | Template schema | `template_path` | JSON schema |
| `extract_data` | Extract data | `file_path`, `schema` | Data dict |
| `validate_schema` | Validate schema | `schema` | Validation result |

---

## 📊 Comparison: Plain vs Agent

| Feature | Plain (`streamlit_app.py`) | Agent (`streamlit_app_agent.py`) |
|---------|---------------------------|----------------------------------|
| Architecture | LlamaIndex Workflows | LangChain ReAct Agent |
| Reasoning Visibility | Hidden | ✅ Visible |
| Tool Calls | Internal | ✅ Visible |
| Modularity | Workflow steps | ✅ Independent tools |
| Extensibility | Moderate | ✅ Easy (add tools) |
| Approval | Event-based | ✅ Executor-based |
| Performance | Fast | Slightly slower (reasoning) |
| Debugging | Limited | ✅ Full transparency |

**Both support:**
- Auto/Manual modes
- Schema editing
- Excel/JSON export
- Caching

---

## 🧪 Testing

### Test with Sample Document

```bash
# 1. Start agent app
streamlit run streamlit_app_agent.py

# 2. Upload a document (e.g., docs/invoice.pdf)

# 3. Enable "Show agent reasoning"

# 4. Run extraction

# 5. Watch agent work:
#    - Parse document
#    - Generate schema
#    - Wait for approval
#    - Extract data
```

### Compare with Plain Implementation

```bash
# Terminal 1: Run plain
streamlit run streamlit_app.py --server.port 8501

# Terminal 2: Run agent
streamlit run streamlit_app_agent.py --server.port 8502

# Upload same document to both
# Compare results
```

---

## ❓ FAQ

### Q: Which implementation should I use?

**A:**
- **Agent** if you want transparency and extensibility
- **Plain** if you want proven stability and speed

### Q: Can I use both?

**A:** Yes! They run in parallel with no conflicts.

### Q: Do they share data?

**A:** Yes, they share:
- Cache (`parsed/` directory)
- Upload directory
- Configuration (`.env`)

### Q: Can I add custom tools?

**A:** Yes! In agent implementation:
1. Create tool in `src/agent/tools/my_tool.py`
2. Add to `get_all_tools()` in `__init__.py`
3. Agent automatically uses it

### Q: Does caching work the same?

**A:** Yes, both implementations use the same cache system in `parsed/`.

---

## 🐛 Troubleshooting

### Agent doesn't pause for approval

**Check:** Tool detection in `executor.py`
```python
# Ensure tool name contains "schema" and "generate"
```

**Debug:** Enable verbose mode:
```python
agent = create_extraction_agent(mode='auto', verbose=True)
```

### Import errors

**Error:** `ModuleNotFoundError: No module named 'src.agent'`

**Fix:** Run from project root:
```bash
cd /path/to/document_wf_v1
streamlit run streamlit_app_agent.py
```

### LangChain version conflicts

**Check:** Ensure exact versions:
```bash
pip list | grep langchain
```

**Fix:** Reinstall:
```bash
pip install --force-reinstall langchain==1.0.2 langchain-openai==1.0.1
```

---

## 📚 Next Steps

1. **Try both implementations** with your documents
2. **Compare outputs** and performance
3. **Choose preferred** implementation
4. **Extend agent** with custom tools if needed
5. **Provide feedback** on what works best

---

## 📖 Documentation

- Full architecture: `README_AGENT.md`
- Original docs: `README.md`
- Tool development: See `README_AGENT.md` → "Adding a New Tool"

---

Enjoy your new LangChain agent-based document extraction! 🎉
