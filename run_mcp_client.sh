#!/bin/bash
# Helper script to run the Streamlit MCP Client
# Usage: ./run_mcp_client.sh

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run streamlit using the venv python
"$SCRIPT_DIR/.venv/bin/python" -m streamlit run "$SCRIPT_DIR/streamlit_app_mcp.py" "$@"
