#!/usr/bin/env bash
# Launch Xu-Lab MCP server with correct cwd & env
cd "/Users/isaacvanorman/AI-local/xu-lab-decks"
export PYTHONPATH="src"
export XU_DECKS_ROOT="/Users/isaacvanorman/AI-local/xu-lab-decks/data/samples"
export XU_DECKS_SCHEMA="/Users/isaacvanorman/AI-local/xu-lab-decks/outputs/schema_records"
export XU_DECKS_SCHEMA_INDEX="/Users/isaacvanorman/AI-local/xu-lab-decks/outputs/index_schema"
export XU_DECKS_MANIFEST="outputs/manifest.jsonl"
exec "/Users/isaacvanorman/AI-local/xu-lab-decks/.venv/bin/python" -m mcp_server.main