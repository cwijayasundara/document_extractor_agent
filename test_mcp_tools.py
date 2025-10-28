#!/usr/bin/env python3
"""Test MCP server tools to diagnose issues.

Usage:
    python test_mcp_tools.py
    # or
    .venv/bin/python test_mcp_tools.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))


def check_environment():
    """Check if required environment variables are set."""
    print("=" * 60)
    print("CHECKING ENVIRONMENT VARIABLES")
    print("=" * 60)

    required_vars = ["LLAMA_CLOUD_API_KEY"]
    llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if llm_provider == "openai":
        required_vars.append("OPENAI_API_KEY")
    elif llm_provider == "groq":
        required_vars.append("GROQ_API_KEY")

    all_ok = True
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: {'*' * 10}{value[-4:] if len(value) > 4 else '****'}")
        else:
            print(f"❌ {var}: NOT SET")
            all_ok = False

    # Optional vars
    optional_vars = ["LLM_PROVIDER", "LLM_MODEL"]
    print("\nOptional variables:")
    for var in optional_vars:
        value = os.getenv(var, "not set")
        print(f"   {var}: {value}")

    return all_ok


def check_imports():
    """Check if all required modules can be imported."""
    print("\n" + "=" * 60)
    print("CHECKING IMPORTS")
    print("=" * 60)

    modules = [
        "streamlit",
        "fastmcp",
        "langchain",
        "langchain_core",
        "langchain_openai",
        "langchain_groq",
        "src.agent.agent",
        "src.plain.config",
        "mcp_server",
    ]

    all_ok = True
    for module in modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            all_ok = False

    return all_ok


async def test_mcp_tool_import():
    """Test if MCP tools can be imported and called."""
    print("\n" + "=" * 60)
    print("TESTING MCP TOOL IMPORTS")
    print("=" * 60)

    try:
        from mcp_server import (
            start_document_extraction,
            approve_extraction_schema,
            reject_extraction_schema
        )
        print("✅ MCP tools imported successfully")
        print(f"   - start_document_extraction: {type(start_document_extraction)}")
        print(f"   - approve_extraction_schema: {type(approve_extraction_schema)}")
        print(f"   - reject_extraction_schema: {type(reject_extraction_schema)}")
        return True
    except Exception as e:
        print(f"❌ MCP tool import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_creation():
    """Test if agent can be created."""
    print("\n" + "=" * 60)
    print("TESTING AGENT CREATION")
    print("=" * 60)

    try:
        from src.agent import create_extraction_agent

        # Try to create agent (might fail if API keys are missing)
        agent = create_extraction_agent(mode="auto", verbose=False)
        print("✅ Agent created successfully")
        print(f"   Type: {type(agent)}")
        return True
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_asyncio():
    """Test asyncio functionality."""
    print("\n" + "=" * 60)
    print("TESTING ASYNCIO")
    print("=" * 60)

    async def dummy_async():
        return {"status": "ok", "message": "Async works!"}

    try:
        result = asyncio.run(dummy_async())
        print(f"✅ asyncio.run() works: {result}")
        return True
    except Exception as e:
        print(f"❌ asyncio.run() failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all diagnostic tests."""
    print("\n" + "=" * 60)
    print("MCP TOOLS DIAGNOSTIC TEST")
    print("=" * 60 + "\n")

    # Load .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ .env file loaded\n")
    except Exception as e:
        print(f"⚠️ .env file not loaded: {e}\n")

    # Run tests
    results = {
        "Environment": check_environment(),
        "Imports": check_imports(),
        "MCP Tools": asyncio.run(test_mcp_tool_import()),
        "Agent Creation": test_agent_creation(),
        "Asyncio": test_asyncio(),
    }

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED - MCP tools should work!")
    else:
        print("❌ SOME TESTS FAILED - Check errors above")
        print("\nCommon fixes:")
        print("1. Missing .env file: Create .env with API keys")
        print("2. Missing packages: pip install -r requirements.txt")
        print("3. Wrong directory: Run from project root")
    print("=" * 60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
