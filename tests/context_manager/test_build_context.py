import pytest
import tempfile
from pathlib import Path
from git import Repo

from backend.app.context_manager import build_context

@pytest.fixture
def mock_workspace():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace_path = Path(tmpdir)
        
        (workspace_path / "file1.py").write_text("from . import file2\n\nclass A:\n    pass")
        (workspace_path / "file2.py").write_text("from . import file1\n\nclass B:\n    pass") # Circular import
        (workspace_path / "file3.py").write_text("class C:\n    pass")
        (workspace_path / "unrelated.txt").write_text("some text")

        repo = Repo.init(workspace_path)
        repo.index.add(["file1.py", "file2.py", "file3.py", "unrelated.txt"])
        repo.index.commit("initial commit")

        yield str(workspace_path)

@pytest.fixture(autouse=True)
def mock_tiktoken(monkeypatch):
    class MockEncoder:
        def encode(self, text, **kwargs): return text.split()
        def decode(self, tokens, **kwargs): return " ".join(tokens)
    class MockTiktoken:
        def get_encoding(self, encoding_name): return MockEncoder()
    monkeypatch.setattr("backend.app.context_manager.packer.tiktoken", MockTiktoken())

def test_empty_workspace():
    with tempfile.TemporaryDirectory() as tmpdir:
        cursor_file = Path(tmpdir) / "new.py"
        cursor_file.write_text("print('hello')")
        
        payload = build_context(tmpdir, str(cursor_file), budget=1000)
        
        assert len(payload.files) == 1
        assert Path(payload.files[0].file_path).name == "new.py"
        assert not payload.truncated

def test_circular_imports(mock_workspace):
    # This test ensures that the scorer doesn't get stuck in an infinite loop
    # with circular imports and that it can still score files.
    cursor_file = str(Path(mock_workspace) / "file1.py")
    
    payload = build_context(mock_workspace, cursor_file, budget=100)
    
    # The scorer should not crash. If it runs without error, the test passes.
    assert payload is not None
    # We expect file1.py (cursor) and file2.py (imported) to be highly scored.
    file_paths = [f.file_path for f in payload.files]
    assert str(Path(cursor_file).resolve()) in file_paths
    assert str(Path(mock_workspace) / "file2.py") in file_paths
