#!/bin/bash
# Helper script to run the Plain Streamlit App (LlamaIndex Workflows)
# Usage: ./run_plain_app.sh

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run streamlit using the venv python
"$SCRIPT_DIR/.venv/bin/python" -m streamlit run "$SCRIPT_DIR/streamlit_app.py" "$@"
