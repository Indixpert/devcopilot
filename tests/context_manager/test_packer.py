import pytest
from backend.app.context_manager.packer import pack_context
from backend.app.context_manager.schemas import ScoredFile

@pytest.fixture(autouse=True)
def mock_tiktoken(monkeypatch):
    class MockEncoder:
        def encode(self, text, **kwargs):
            # Simple mock: 1 word = 1 token
            return text.split()
        def decode(self, tokens, **kwargs):
            return " ".join(tokens)
    
    class MockTiktoken:
        def get_encoding(self, encoding_name):
            return MockEncoder()

    monkeypatch.setattr("backend.app.context_manager.packer.tiktoken", MockTiktoken())

def test_pack_context_empty_input():
    payload = pack_context([], 100)
    assert payload.files == []
    assert payload.total_tokens == 0
    assert not payload.truncated

def test_pack_context_exact_fit():
    files = [
        ScoredFile(file_path="a.py", score=1.0, content="hello world"),  # 2 tokens
        ScoredFile(file_path="b.py", score=0.9, content="foo bar baz"),  # 3 tokens
    ]
    payload = pack_context(files, 5)
    assert len(payload.files) == 2
    assert payload.total_tokens == 5
    assert not payload.truncated
    assert payload.files[0].file_path == "a.py"
    assert payload.files[1].file_path == "b.py"

def test_pack_context_overflow_and_truncation():
    files = [
        ScoredFile(file_path="a.py", score=1.0, content="one two three"),  # 3 tokens
        ScoredFile(file_path="b.py", score=0.9, content="four five six"),  # 3 tokens
    ]
    payload = pack_context(files, 5)
    assert len(payload.files) == 2
    assert payload.files[0].file_path == "a.py"
    assert payload.files[0].content == "one two three"
    assert payload.files[1].file_path == "b.py"
    assert payload.files[1].content == "four five"
    assert payload.total_tokens == 5
    assert payload.truncated

def test_pack_context_single_large_file():
    files = [
        ScoredFile(file_path="large.py", score=1.0, content="one two three four five six")  # 6 tokens
    ]
    payload = pack_context(files, 4)
    assert len(payload.files) == 1
    assert payload.files[0].file_path == "large.py"
    assert payload.files[0].content == "one two three four"
    assert payload.total_tokens == 4
    assert payload.truncated

def test_pack_context_truncation_ordering():
    # Ensure higher scored file is packed first, even if it causes truncation
    files = [
        ScoredFile(file_path="small.py", score=0.8, content="one two"),  # 2 tokens
        ScoredFile(file_path="large.py", score=1.0, content="one two three four five six")  # 6 tokens
    ]
    payload = pack_context(files, 4)
    assert len(payload.files) == 1
    assert payload.files[0].file_path == "large.py"
    assert payload.files[0].content == "one two three four"
    assert payload.total_tokens == 4
    assert payload.truncated
