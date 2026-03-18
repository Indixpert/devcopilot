from dataclasses import dataclass
from typing import List

@dataclass
class FileSnippet:
    file_path: str
    content: str

@dataclass
class ContextPayload:
    files: List[FileSnippet]
    total_tokens: int
    truncated: bool

@dataclass
class ScoredFile:
    file_path: str
    score: float
    content: str
