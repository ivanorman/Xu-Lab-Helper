# src/mcp_server/main.py
from __future__ import annotations
import os, json, hashlib, glob, subprocess, re, shutil
from typing import Iterator, Dict, List, Tuple, Set
from datetime import datetime
from difflib import SequenceMatcher

from mcp.server.fastmcp import FastMCP
from tqdm import tqdm
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

# ----------------------
# MCP instance
# ----------------------
mcp = FastMCP("xu-lab-data")

# ----------------------
# Defaults / Paths
# ----------------------
DEF_ROOT   = os.environ.get("XU_DATA_ROOT", "data")
DEF_MAN    = os.environ.get("XU_DATA_MANIFEST", "outputs/manifest.jsonl")
DEF_SLIDES = os.environ.get("XU_DATA_SLIDES", "outputs/jsonl")
DEF_IMAGES = os.environ.get("XU_DATA_IMAGES", "outputs/images")
DEF_SCHEMA = os.environ.get("XU_DATA_SCHEMA", "outputs/schema_records")
DEF_SCHEMA_INDEX = os.environ.get("XU_DATA_SCHEMA_INDEX", "outputs/index_schema")
SCHEMA_VERSION = "deck-schema-v1"
STATE_PATH = os.environ.get("XU_DATA_STATE", "outputs/STATE.json")
ALIASES_PATH = os.environ.get("XU_DATA_ALIASES", "outputs/aliases.json")

@mcp.tool()
def ingest_presentation(path: str):
    """
    Scans a PowerPoint presentation, extracts text and images, and creates a summary.
    This is a placeholder and does not yet have full functionality.
    """
    # In a real implementation, this would involve parsing the pptx file.
    # For now, we'll just create a dummy schema file.
    basename = os.path.splitext(os.path.basename(path))[0]
    schema = {
        "basename": basename,
        "title": "Placeholder Title",
        "summary": "This is a placeholder summary of the presentation.",
        "device": "Placeholder Device",
        "figures": []
    }
    # Create a dummy schema file.
    with open(f"outputs/schemas/{basename}_schema.json", "w") as f:
        json.dump(schema, f, indent=2)
    return f"Successfully ingested presentation and created schema for {basename}"

@mcp.tool()
def inspect_mat_file(file_path: str):
    """
    Inspects a .mat file and returns a dictionary of variables and their shapes.
    """
    try:
        mat_data = loadmat(file_path)
        return {k: v.shape for k, v in mat_data.items() if not k.startswith('__')}
    except FileNotFoundError:
        return f"Error: File not found at {file_path}"
    except Exception as e:
        return f"Error reading .mat file: {e}"

@mcp.tool()
def ingest_data_files(path: str = "data/Data"):
    """
    Scans data files in a directory and creates a JSON file with metadata for each file,
    including variable shapes and descriptions from .mat files.
    """
    metadata = []
    for root, _, files in os.walk(path):
        for filename in files:
            file_path = os.path.join(root, filename)
            entry = {"file_path": file_path, "filename": filename}
            if filename.endswith(".mat"):
                try:
                    mat_data = loadmat(file_path)
                    entry["file_type"] = "matlab_data"
                    variables = {k: v.shape for k, v in mat_data.items() if not k.startswith('__')}
                    entry["variables"] = variables
                    if 'description' in mat_data:
                        entry["description"] = mat_data["description"][0]
                except Exception as e:
                    entry["error"] = f"Could not read .mat file: {e}"
            elif filename.endswith(".m"):
                try:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        content = f.read(1000)
                        entry["file_type"] = "matlab_script"
                        entry["script_content"] = content
                except Exception as e:
                    entry["error"] = f"Could not read .m file: {e}"
            else:
                entry["file_type"] = "other"

            metadata.append(entry)

    with open("outputs/data_metadata.jsonl", "w") as f:
        for entry in metadata:
            # Need to handle numpy shapes for JSON serialization
            f.write(json.dumps(entry, default=lambda x: str(x) if isinstance(x, np.ndarray) else x) + "\n")
    return f"Successfully ingested {len(metadata)} data files."

@mcp.tool()
def find_data(keywords: List[str], k: int = 5):
    """
    Searches for relevant data files by computing a relevance score based on a list of keywords.
    This is a fuzzy search that handles typos and returns a ranked list of results.
    """
    results_with_scores = []
    if not keywords:
        return "Please provide a list of keywords to search for."

    lower_keywords = [kw.lower() for kw in keywords]

    try:
        with open("outputs/data_metadata.jsonl", "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    # Create a single searchable text blob from all values in the entry
                    searchable_blob = " ".join(str(v) for v in entry.values()).lower()
                    words_in_blob = set(searchable_blob.split())

                    total_score = 0
                    for kw in lower_keywords:
                        # Find the best match for the keyword in the entire blob
                        best_match_score_for_kw = 0
                        if words_in_blob:
                           best_match_score_for_kw = max(SequenceMatcher(None, kw, word).ratio() for word in words_in_blob)
                        total_score += best_match_score_for_kw
                    
                    if total_score > 0:
                        results_with_scores.append({"score": total_score, "entry": entry})

                except (json.JSONDecodeError, AttributeError):
                    continue
    except FileNotFoundError:
        return "The data metadata file ('outputs/data_metadata.jsonl') was not found. Please run the `ingest_data_files` tool first."
    
    # Sort results by score in descending order
    results_with_scores.sort(key=lambda x: x['score'], reverse=True)

    # Return just the entries, without the scores
    return [item['entry'] for item in results_with_scores[:k]]

@mcp.tool()
def plot_data(file_path: str, x_variable: str, y_variable: str, z_variable: str = None, plot_parameters: dict = None):
    """
    Loads data from a .mat file and generates a plot, returning the path to the saved image.

    Args:
        file_path: Path to the .mat file.
        x_variable: Name of the variable for the x-axis.
        y_variable: Name of the variable for the y-axis.
        z_variable: Name of the variable for the color data in a 2D plot.
        plot_parameters: A dictionary of optional parameters for plot customization.
            Supported parameters:
            - 'xlim': [min, max]
            - 'ylim': [min, max]
            - 'title': str
            - 'xlabel': str
            - 'ylabel': str
            - 'log_scale_x': bool
            - 'log_scale_y': bool
            - 'cmap': str (for 2D plots)
    """
    try:
        mat_data = loadmat(file_path)
    except FileNotFoundError:
        return f"Error: File not found at {file_path}"
    except Exception as e:
        return f"Error reading .mat file: {e}"

    # Squeeze all data arrays to remove singleton dimensions
    for key in mat_data:
        if isinstance(mat_data[key], np.ndarray):
            mat_data[key] = np.squeeze(mat_data[key])

    if x_variable not in mat_data or y_variable not in mat_data:
        return f"Error: One or both variables '{x_variable}', '{y_variable}' not found in file. Available variables: {list(mat_data.keys())}"

    x_data = mat_data[x_variable]
    y_data = mat_data[y_variable]

    fig, ax = plt.subplots()

    # Handle 2D plots
    if z_variable:
        if z_variable not in mat_data:
            return f"Error: Z-variable '{z_variable}' not found in file."
        z_data = mat_data[z_variable]
        im = ax.pcolormesh(x_data, y_data, z_data.T, cmap=plot_parameters.get('cmap', 'viridis'))
        fig.colorbar(im, ax=ax)
    # Handle 1D plots
    else:
        ax.plot(x_data, y_data)

    if plot_parameters:
        if 'xlim' in plot_parameters: ax.set_xlim(plot_parameters['xlim'])
        if 'ylim' in plot_parameters: ax.set_ylim(plot_parameters['ylim'])
        if 'title' in plot_parameters: ax.set_title(plot_parameters['title'])
        if 'xlabel' in plot_parameters: ax.set_xlabel(plot_parameters['xlabel'])
        if 'ylabel' in plot_parameters: ax.set_ylabel(plot_parameters['ylabel'])
        if plot_parameters.get('log_scale_x'): ax.set_xscale('log')
        if plot_parameters.get('log_scale_y'): ax.set_yscale('log')

    ax.set_xlabel(x_variable if 'xlabel' not in (plot_parameters or {}) else plot_parameters['xlabel'])
    ax.set_ylabel(y_variable if 'ylabel' not in (plot_parameters or {}) else plot_parameters['ylabel'])
    ax.set_title(f'{y_variable} vs. {x_variable}' if 'title' not in (plot_parameters or {}) else plot_parameters['title'])
    ax.grid(True)

    # Save the plot
    os.makedirs("outputs/plots", exist_ok=True)
    plot_filename = f"{os.path.splitext(os.path.basename(file_path))[0]}.png"
    plot_path = os.path.join("outputs/plots", plot_filename)
    plt.savefig(plot_path)
    plt.close(fig)

    return {"plot_path": plot_path, "message": f"Plot saved to {plot_path}"}

if __name__ == "__main__":
    mcp.run()
