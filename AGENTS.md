# Document Extraction Agents

## Project Structure & Module Organization
- `src/plain/` – LlamaIndex Workflows implementation of the iterative extraction pipeline. Includes `extraction_workflow.py` for the event-driven workflow, `template_schema_generator.py` for Excel-driven schemas, `excel_export.py` for workbook output, `cache_utils.py` for parse caching, and `config.py` for API client factories. The CLI helper in `main.py` wraps the same workflow for scripts.
- `src/agent/` – LangChain 1.0.2 ReAct agent stack. `agent.py` wires the agent graph, `executor.py` provides the approval-aware wrapper, `prompts.py` stores the system prompt, and `tools/` hosts the tool definitions used by the agent.
- `src/auto_schema_gen.py` – Async CLI harness that runs the plain workflow end-to-end with the sample invoice and manual approval loop.
- `streamlit_app.py` – Streamlit UI built on the plain workflow. Houses shared UI helpers such as `parse_schema_structure`, `reconstruct_schema_from_structure`, `display_schema_visual`, and `display_extracted_data`.
- `streamlit_app_agent.py` – Streamlit UI for the LangChain agent. Reuses the rendering helpers from `streamlit_app.py` and adds agent reasoning toggles and state management.
- `docs/` – Sample PDFs, images, and Excel templates for local testing. Uploads staged through the UI land in `uploads/` (gitignored).
- `parsed/` – Repository-level cache for extracted outputs (referenced by `src/plain/config.py`). The plain workflow also keeps markdown parses in `src/plain/parsed/` via `cache_utils.py`; clean both when resetting state.
- `README.md` covers the plain workflow, while `README_AGENT.md` dives deeper into the agent orchestration.

## Build, Run, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` – set up an isolated runtime.
- `pip install -r requirements.txt` – install workflow dependencies.
- `cp .env.example .env` and populate `OPENAI_API_KEY`, `LLAMA_CLOUD_API_KEY`, `LLAMA_CLOUD_PROJECT_ID`, and `LLAMA_CLOUD_ORG_ID` for both stacks (see `src/plain/config.py`).
- Streamlit UIs:
  - Plain workflow: `streamlit run streamlit_app.py`
  - Agent workflow: `streamlit run streamlit_app_agent.py`
- CLI workflows:
  - `python -m src.auto_schema_gen` – run the plain workflow against the sample invoice with an interactive approval prompt.
  - Launch a Python shell and `from src.agent import create_extraction_agent` to embed the LangChain agent in other services (see `README_AGENT.md` for invocation examples).

## Coding Style & Naming Conventions
- Use 4-space indentation, type hints on public functions, and module-level docstrings mirroring the existing code.
- Keep module and function names in `snake_case`, classes in `PascalCase`, and continue suffixing orchestration entry points with descriptive names (e.g., `ApprovalAgentWrapper`).
- Prefer dependency injection through constructor or function parameters instead of global state. Reuse the factories in `src/plain/config.py` for API clients.
- Run `isort` followed by `black` on touched files (`isort src` and `black src`).

### Streamlit UI Notes (`streamlit_app.py`, `streamlit_app_agent.py`)
- Keep schema parsing and reconstruction helpers pure; they are imported by both apps.
- Preserve container names/descriptions when rebuilding JSON schemas so arrays/objects retain context after edits.
- Use `display_extracted_data` for nested table rendering and maintain the camelCase → Title Case formatting.
- Update user-facing copy alongside functional changes; prefer `st.caption` for inline explanations and keep the agent reasoning toggle in sync with new diagnostics.

### LangChain Agent Tooling (`src/agent/tools/`)
- `ParseDocumentTool` – parses PDF/image content to markdown using LlamaParse (shared config).
- `GenerateSchemaAITool` – creates JSON schemas with OpenAI models.
- `GenerateSchemaTemplateTool` – builds schemas from Excel templates via `TemplateSchemaGenerator`.
- `ExtractDataTool` – executes LlamaExtract against the approved schema.
- `ValidateSchemaTool` – checks JSON Schema compliance before extraction.

## Testing Guidelines
- Place pytest suites under `tests/`, mirroring the `src/plain` and `src/agent` layout (`tests/plain/`, `tests/agent/`, etc.).
- Use fixtures to load documents from `docs/` and clean up temporary outputs in `parsed/` (and `src/plain/parsed/`) after assertions.
- Mark cross-stack flows with `@pytest.mark.integration` and run locally with `pytest -q` before opening a pull request.

## Commit & Pull Request Guidelines
- Write commit subjects in the imperative mood ("Add approval step logging"), keeping the subject under 72 characters and grouping related changes.
- Summarize workflow impact in pull request descriptions, note required API keys, and link tracking issues where possible.
- Attach screenshots or terminal logs when behavior changes the Streamlit UI or extraction results and ensure the test suite passes before requesting review.

## Security & Configuration Tips
- Store secrets exclusively in `.env` (never in the repo) and document any new variables inside `src/plain/config.py`.
- Verify that artifacts in `parsed/` (and cached markdown in `src/plain/parsed/`) do not include sensitive customer data before sharing logs or examples.
- When changing caching behavior or download destinations, update both Streamlit apps so clean-up and export paths stay aligned.
