import pytest
from faker import Faker
import tempfile
import os
from pathlib import Path
import tracemalloc

# This assumes PYTHONPATH is set up to find the backend module
from backend.app.context_manager import build_context

fake = Faker()

def create_fake_workspace(tmpdir, num_files):
    workspace_path = Path(tmpdir)
    files = []
    for i in range(num_files):
        file_path = workspace_path / f"module_{i}.py"
        
        imports = ""
        if i > 0:
            imports += f"from .module_{i-1} import MyClass_{i-1}\n"
        if i > 1 and i % 5 == 0: # Add more complex imports occasionally
            imports += f"from .module_{i-2} import MyClass_{i-2}\n"

        content = f"""{imports}
import os
import sys

class MyClass_{i}:
    def method_{i}_a(self):
        print('{fake.sentence()}')
        return {i}

    def method_{i}_b(self, val):
        return f'value: {{val}}'
"""
        
        with open(file_path, "w") as f:
            f.write(content)
        files.append(str(file_path))
    
    from git import Repo
    repo = Repo.init(workspace_path)
    repo.index.add(files)
    repo.index.commit("initial commit")

    # Make some files more recent
    for i in range(num_files // 10):
        with open(files[i], "a") as f:
            f.write("\n# recent edit")
        repo.index.add([files[i]])
        repo.index.commit(f"update file {i}")

    return str(workspace_path), files[num_files // 2] # cursor in the middle

@pytest.mark.parametrize("num_files", [50, 100, 200])
def test_build_context_benchmark(benchmark, num_files):
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace_path, cursor_file = create_fake_workspace(tmpdir, num_files)

        def run():
            build_context(workspace_path, cursor_file)

        benchmark(run)


@pytest.mark.performance
def test_build_context_200_files_with_profiling(benchmark):
    """
    Benchmark for 200 files with memory profiling.
    The P99 latency assertion should be handled by CI parsing benchmark results.
    Example: `pytest --benchmark-json=report.json` and a script asserts on stats in the report.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace_path, cursor_file = create_fake_workspace(tmpdir, 200)

        tracemalloc.start()

        result = benchmark(build_context, workspace_path, cursor_file)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"\nWorkspace size: 200 files")
        print(f"Peak memory usage: {peak / 10**6:.2f} MB")
        
        assert result is not None
        # A basic check that context was built
        assert result.total_tokens > 0
