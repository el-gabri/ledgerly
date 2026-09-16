"""Build and install a clean wheel, then exercise it outside the checkout.

Run with the development dependencies installed: python scripts/check_wheel.py
No package downloads or optional model calls are made by this check.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile


def run(command: list[str], cwd: Path, env: dict[str, str]) -> None:
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"Command failed: {command!r}\n{result.stdout}\n{result.stderr}")


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    env = {
        **os.environ,
        "LEDGERLY_LLM_MODE": "offline", "LEDGERLY_EMBEDDINGS": "tfidf",
        "LANGCHAIN_TRACING_V2": "false", "LANGSMITH_TRACING": "false",
        "PIP_NO_INDEX": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    }
    with tempfile.TemporaryDirectory(prefix="ledgerly-wheel-") as directory:
        work = Path(directory).resolve()
        # All build/install output and automatic cleanup stay under this temp root.
        assert work.is_relative_to(Path(tempfile.gettempdir()).resolve())
        source = work / "source"
        source.mkdir()
        for name in ("pyproject.toml", "README.md"):
            shutil.copy2(repository / name, source / name)
        shutil.copytree(repository / "ledgerly", source / "ledgerly",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        wheels = work / "wheels"
        wheels.mkdir()
        run([sys.executable, "-c",
             "from setuptools.build_meta import build_wheel; "
             f"build_wheel({str(wheels)!r})"], source, env)
        wheel, = wheels.glob("*.whl")
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            assert "ledgerly/data/accounts.json" in names
            assert len([name for name in names if name.startswith("ledgerly/data/kb_docs/")
                        and name.endswith(".md")]) == 15
        installed = work / "installed"
        run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-index",
             "--target", str(installed), str(wheel)], work, env)
        probe = f"""
import sys
from pathlib import Path
sys.path.insert(0, {str(installed)!r})
import ledgerly
assert Path(ledgerly.__file__).resolve().is_relative_to(Path({str(installed)!r}))
from ledgerly.graph import build_app, run_turn
app = build_app()
account = run_turn(app, 'wheel-account', "What's my balance?")
assert account['messages'][-1].agent == 'account'
assert '1,284.50' in account['messages'][-1].content
kb = run_turn(app, 'wheel-kb', 'When do transfer limits reset?')
assert kb['messages'][-1].agent == 'kb'
assert 'midnight UTC' in kb['messages'][-1].content
assert kb['draft'].citations == ['transfer-limits']
"""
        run([sys.executable, "-I", "-c", probe], work, env)
        print(f"Wheel smoke passed: {wheel.name}; account fixture and 15 KB articles installed.")


if __name__ == "__main__":
    main()
