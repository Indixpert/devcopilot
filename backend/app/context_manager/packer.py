from typing import List
import tiktoken

from .schemas import ScoredFile, ContextPayload, FileSnippet

# Using cl100k_base as it's common for OpenAI models.
ENCODING = tiktoken.get_encoding("cl100k_base")

def pack_context(scored_files: List[ScoredFile], budget: int) -> ContextPayload:
    """
    Greedily packs scored files into a context budget.

    It iterates through files sorted by score in descending order, adding them
    to the context until the budget is exhausted. If a file exceeds the
    remaining budget, it's truncated to fit.

    Args:
        scored_files: A list of ScoredFile objects.
        budget: The token budget.

    Returns:
        A ContextPayload containing the packed files.
    """
    packed_files: List[FileSnippet] = []
    total_tokens = 0
    truncated = False

    # Sort files by score, descending
    sorted_files = sorted(scored_files, key=lambda f: f.score, reverse=True)

    for file in sorted_files:
        if total_tokens >= budget:
            truncated = True
            break

        remaining_budget = budget - total_tokens
        
        file_tokens = ENCODING.encode(file.content)
        num_file_tokens = len(file_tokens)

        if num_file_tokens <= remaining_budget:
            packed_files.append(FileSnippet(file_path=file.file_path, content=file.content))
            total_tokens += num_file_tokens
        else:
            # Truncate the file to fit the remaining budget
            truncated_tokens = file_tokens[:remaining_budget]
            truncated_content = ENCODING.decode(truncated_tokens)
            
            if not truncated_content.strip():
                truncated = True
                break

            packed_files.append(FileSnippet(file_path=file.file_path, content=truncated_content))
            actual_tokens_added = len(ENCODING.encode(truncated_content))
            total_tokens += actual_tokens_added
            truncated = True
            break  # Stop after the first truncation

    return ContextPayload(files=packed_files, total_tokens=total_tokens, truncated=truncated)
