from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.real_trace.analysis import ensure_real_trace_exists, load_real_trace_from_csv


class TestRealTraceDispatch(unittest.TestCase):
    def test_ensure_real_trace_exists_generates_qft_trace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "qft_n2.csv"
            output_path = ensure_real_trace_exists(str(path))
            trace = load_real_trace_from_csv(str(output_path))
            self.assertTrue(output_path.exists())
            self.assertTrue(trace)


if __name__ == "__main__":
    unittest.main()
