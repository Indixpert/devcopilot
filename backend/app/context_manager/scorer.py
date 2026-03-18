import os
import re
import logging
from typing import List, Dict, Set
from pathlib import Path
from collections import deque

from git import Repo, GitCommandError, InvalidGitRepositoryError

from .schemas import ScoredFile

# NOTE: This implementation uses regex for symbol and import extraction for simplicity
# and to avoid complex build dependencies for tree-sitter grammars.
# A more robust implementation should use tree-sitter as requested.
# This would involve:
# 1. Setting up tree-sitter language grammars (e.g., python, typescript).
# 2. Replacing _extract_symbols_simple with a function that traverses the AST
#    and extracts identifiers from function/class/variable declarations.
# 3. Replacing _extract_imports_simple with a function that uses AST nodes
#    for import/require statements to build a more accurate import graph.

logging.basicConfig(level=logging.INFO)

def _extract_symbols_simple(content: str) -> Set[str]:
    """Extracts function and class names from Python code using regex."""
    symbols = set(re.findall(r"^(?:class|def)\s+([a-zA-Z0-9_]+)", content, re.MULTILINE))
    return symbols

def _extract_imports_simple(content: str, file_path: str, workspace_path: str) -> List[str]:
    """Extracts imported modules from Python code using regex."""
    imports = []
    # For now, this is a simplified implementation focusing on relative imports
    # A robust solution needs to handle Python's complex import resolution rules.
    relative_imports = re.findall(r"^from\s+(\.+)(\w*)", content, re.MULTILINE)
    for dots, module in relative_imports:
        try:
            level = len(dots)
            base_path = Path(file_path).parent
            for _ in range(level - 1):
                base_path = base_path.parent
            
            imp_path = base_path / f"{module}.py"
            if module and imp_path.exists():
                imports.append(str(imp_path.resolve()))
            elif (base_path / module).is_dir():
                imp_path = base_path / module / "__init__.py"
                if imp_path.exists():
                    imports.append(str(imp_path.resolve()))
        except Exception as e:
            logging.debug(f"Could not resolve relative import '{''.join(dots)}{module}' in {file_path}: {e}")

    return imports

def _get_recency_scores(workspace_path: str, all_files: List[str]) -> Dict[str, float]:
    """Calculates recency scores for files based on last git commit date."""
    try:
        repo = Repo(workspace_path, search_parent_directories=True)
        file_mod_times = {}
        for file_path in all_files:
            try:
                last_commit = next(repo.iter_commits(paths=file_path, max_count=1), None)
                file_mod_times[file_path] = last_commit.committed_date if last_commit else 0
            except Exception:
                file_mod_times[file_path] = 0

        non_zero_times = [t for t in file_mod_times.values() if t > 0]
        if not non_zero_times:
            return {f: 0.0 for f in all_files}

        min_time, max_time = min(non_zero_times), max(non_zero_times)
        time_range = float(max_time - min_time) if max_time > min_time else 1.0

        return {
            f: ((t - min_time) / time_range) if t > 0 else 0.0
            for f, t in file_mod_times.items()
        }
    except (InvalidGitRepositoryError, GitCommandError, ValueError):
        logging.warning("Could not read git history. Recency scores will be 0.")
        return {f: 0.0 for f in all_files}

def _build_import_graph(workspace_path: str, all_files: List[str]) -> Dict[str, List[str]]:
    """Builds a graph of imports between files in the workspace."""
    graph = {f: [] for f in all_files}
    abs_path_files = {str(Path(f).resolve()) for f in all_files}

    for file_path in all_files:
        if not file_path.endswith('.py'):
            continue
        try:
            content = Path(file_path).read_text(errors='ignore')
            imports = _extract_imports_simple(content, file_path, workspace_path)
            for imp_path in imports:
                if imp_path in abs_path_files:
                    graph[file_path].append(imp_path)
        except Exception as e:
            logging.warning(f"Failed to parse imports for {file_path}: {e}")
    return graph

def _get_distance_scores(import_graph: Dict[str, List[str]], cursor_file: str) -> Dict[str, float]:
    """Calculates scores based on import graph distance from the cursor file."""
    distances = {f: float('inf') for f in import_graph}
    if cursor_file not in import_graph:
        return {f: 0.0 for f in import_graph}

    distances[cursor_file] = 0
    queue = deque([cursor_file])
    visited = {cursor_file}

    # BFS to find distances from cursor_file (both imports and dependents)
    while queue:
        current_file = queue.popleft()
        
        # Files that import current_file
        for f, imports in import_graph.items():
            if current_file in imports and f not in visited:
                distances[f] = distances[current_file] + 1
                visited.add(f)
                queue.append(f)

        # Files that current_file imports
        for imp in import_graph.get(current_file, []):
            if imp not in visited:
                distances[imp] = distances[current_file] + 1
                visited.add(imp)
                queue.append(imp)

    max_dist = max((d for d in distances.values() if d != float('inf')), default=0)
    if max_dist == 0: return {f: 1.0 if f == cursor_file else 0.0 for f in import_graph}

    return {
        f: (1.0 - (d / max_dist)) if d != float('inf') else 0.0
        for f, d in distances.items()
    }

def score_files(workspace_path: str, cursor_file: str, all_files: List[str]) -> List[ScoredFile]:
    """Scores files based on relevance to the cursor file."""
    cursor_file_abs = str(Path(cursor_file).resolve())
    all_files_abs = [str(Path(f).resolve()) for f in all_files]
    workspace_path_abs = str(Path(workspace_path).resolve())

    recency_scores = _get_recency_scores(workspace_path_abs, all_files_abs)
    import_graph = _build_import_graph(workspace_path_abs, all_files_abs)
    distance_scores = _get_distance_scores(import_graph, cursor_file_abs)

    try:
        cursor_content = Path(cursor_file_abs).read_text(errors='ignore')
        cursor_symbols = _extract_symbols_simple(cursor_content)
    except Exception: cursor_symbols = set()

    scored_files = []
    for file_path in all_files_abs:
        try:
            content = Path(file_path).read_text(errors='ignore')
            symbols = _extract_symbols_simple(content)
            
            intersection = len(cursor_symbols.intersection(symbols))
            union = len(cursor_symbols.union(symbols))
            symbol_score = intersection / union if union > 0 else 0.0

            score = (
                symbol_score * 0.5 +
                distance_scores.get(file_path, 0.0) * 0.3 +
                recency_scores.get(file_path, 0.0) * 0.2
            )
            
            if file_path == cursor_file_abs: score = 2.0 # Boost current file

            scored_files.append(ScoredFile(file_path=file_path, score=score, content=content))
        except Exception as e:
            logging.warning(f"Could not score file {file_path}: {e}")
            scored_files.append(ScoredFile(file_path=file_path, score=0.0, content=""))

    return scored_files
