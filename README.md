# Xu Lab Decks — Lucien Setup Guide

This repository contains the code and configuration to run Lucien, an MCP-based assistant for managing and searching the Xu Lab’s slide decks. Lucien parses PowerPoint presentations, extracts text and images, generates captions and structured metadata, and builds a searchable database.

---

## Repository structure

```
xu-lab-decks/
├── configs/              # Prompts and config files
├── data/
│   └── samples/          # Put .pptx slide decks here
├── outputs/
│   ├── images/           # Temporary slide image exports
│   ├── schema_records/   # Schema JSONs
│   └── index_schema/     # Search index
├── scripts/
│   └── xu_mcp.sh         # Launch script for MCP server
├── src/                  # Python source (mcp_server, parsing, indexing)
├── schema.md             # Schema template for structured metadata
├── requirements.txt
└── mcp.json              # MCP manifest
```

---

## Environment setup

```bash
git clone https://github.com/ivanorman/Xu-Lab-Helper.git
cd Xu-Lab-Helper
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Adding slide decks

Place all `.pptx` files you want Lucien to process into `data/samples/`.  
You can organize by subfolders; Lucien scans recursively.

---

## Launch script

Lucien runs through an MCP server started by `scripts/xu_mcp.sh`.  
Edit this file so that all paths match your machine:

```bash
#!/usr/bin/env bash
cd /Users/<your-username>/AI-local/xu-lab-decks
export PYTHONPATH=src
export XU_DECKS_ROOT=$(pwd)/data/samples
export XU_DECKS_SCHEMA=$(pwd)/outputs/schema_records
export XU_DECKS_SCHEMA_INDEX=$(pwd)/outputs/index_schema
export XU_DECKS_MANIFEST=$(pwd)/outputs/manifest.jsonl
exec .venv/bin/python -m mcp_server.main
```

Then make it executable:

```bash
chmod +x scripts/xu_mcp.sh
```

---

## Connecting to Lucien

In the Lucien desktop app or VSCode MCP client:

1. Open the MCP connections panel.
2. Add a new server with:
   - **Name:** `xu_lab_decks`
   - **Command:** `/absolute/path/to/scripts/xu_mcp.sh`
   - **Args:** `--start`
3. Save and restart Lucien.

Lucien should now recognize `xu_lab_decks` as a connected tool.

---

## Building the corpus

Once connected, tell Lucien:

```
Rebuild and reindex the corpus.
```

Lucien will:
1. Check for new or changed decks.
2. Parse slides and extract text and images.
3. Caption images and enrich schema.
4. Build an index for searching.

If you add new slides later, use:
```
Update corpus and reindex.
```

---

## Searching

Lucien supports flexible natural-language queries:

- “Find slides on twisted MoTe2 transport.”
- “Show Raman and PL data for WSe2.”
- “Summarize CrSBr electrical results.”

Lucien expands known aliases (like `tMoTe2`, `MoTe₂`, `twisted MoTe2`), searches slide text and captions, and lists matching decks for you to open or refine.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: mcp_server.manifest` | Ensure `src/mcp_server/__init__.py` exists and `PYTHONPATH=src` |
| Stuck indexing | Disable iCloud/OneDrive sync for this directory |
| MCP connection error `ENOENT` | Check that the `xu_mcp.sh` path is correct |
| `Connection closed` | Usually due to a missing manifest or Python import issue |

---

## Notes

- `data/samples` and `outputs/images` are excluded from git commits but kept as folders.
- For a clean rebuild, remove contents of `outputs/schema_records` and `outputs/index_schema`.
- Avoid running from synced cloud directories; use a local path instead.

---

## Workspace prompt

Lucien’s behavior and tool use can be customized with a workspace prompt.  
A pre-written prompt is included in `configs/`. To use it:

1. Open Lucien’s workspace setup panel.
2. Copy the contents of the prompt file from `configs/`.
3. Paste it into the workspace prompt field in Lucien.
4. Save and restart Lucien for the settings to take effect.
