import os
import glob
from typing import List, Optional
from pathlib import Path

from .schemas import ContextPayload
from .scorer import score_files
from .packer import pack_context

__all__ = ["build_context"]

def _get_workspace_files(workspace_path: str) -> List[str]:
    """Gathers relevant source files from the workspace, ignoring common patterns."""
    # A more robust implementation would respect .gitignore
    files = []
    # For now, let's just grab py, js, ts files.
    supported_extensions = ["**/*.py", "**/*.js", "**/*.ts"]
    
    for ext in supported_extensions:
        pattern = os.path.join(workspace_path, ext)
        files.extend(glob.glob(pattern, recursive=True))
    
    # Filter out virtual environments and node_modules
    filtered_files = [
        f for f in files 
        if '/.venv/' not in f and '/venv/' not in f and '/node_modules/' not in f
    ]
    return filtered_files

def build_context(
    workspace_path: str, 
    cursor_file: str, 
    budget: int = 8000, 
    language_hint: Optional[str] = None
) -> ContextPayload:
    """
    Selects and compresses workspace file content for LLM context.

    This function analyzes the workspace, scores files based on their relevance
    to the current editing context, and packs them into a token budget.

    Args:
        workspace_path: The absolute path to the workspace root.
        cursor_file: The absolute path to the file currently being edited.
        budget: The maximum number of tokens for the context. Defaults to 8000.
        language_hint: Optional language hint (currently unused, for future enhancement).

    Returns:
        A ContextPayload containing selected file snippets and metadata.
    """
    # 1. Discover all files in the workspace
    all_files = _get_workspace_files(workspace_path)
    
    cursor_file_abs = str(Path(cursor_file).resolve())
    if cursor_file_abs not in [str(Path(f).resolve()) for f in all_files]:
        all_files.append(cursor_file_abs)

    # 2. Score files for relevance
    scored_files = score_files(workspace_path, cursor_file_abs, all_files)

    # 3. Pack files into budget
    context_payload = pack_context(scored_files, budget)

    return context_payload
