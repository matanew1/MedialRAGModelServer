"""Test configuration for pytest.

Ensures the project root is on sys.path so that 'import app.*' works
when running tests from the repository root.
"""
import sys
from pathlib import Path
import types

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Provide a lightweight stub for the 'rag' module so importing app.main does not
# require heavy dependencies (faiss, sentence-transformers) during HTTP contract tests.
if 'rag' not in sys.modules:
    rag_stub = types.ModuleType('rag')

    def _stub_answer(q: str):  # pragma: no cover - trivial
        return 'stub-answer'

    def _stub_answer_with_debug(q: str):  # pragma: no cover - trivial
        return 'stub-answer', {'dbg': True}

    rag_stub.get_answer = _stub_answer
    rag_stub.get_answer_with_debug = _stub_answer_with_debug
    sys.modules['rag'] = rag_stub
