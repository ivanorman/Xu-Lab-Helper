#!/usr/bin/env bash
# Launch Xu-Lab MCP server with correct cwd & env
cd "/Users/isaacvanorman/AI-local/xu-lab-data"
export PYTHONPATH="src"
export XU_DECKS_ROOT="/Users/isaacvanorman/AI-local/xu-lab-data/data"
export XU_DECKS_SCHEMA="/Users/isaacvanorman/AI-local/xu-lab-data/outputs/schema_records"
export XU_DECKS_SCHEMA_INDEX="/Users/isaacvanorman/AI-local/xu-lab-data/outputs/index_schema"
export XU_DECKS_MANIFEST="outputs/manifest.jsonl"
exec "/Users/isaacvanorman/AI-local/xu-lab-data/.venv/bin/python" -m mcp_server.main