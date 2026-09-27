"""CPU-only regression tests for window lookup in long-running streams."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_methods(relative_path, class_name, names):
    path = ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(methods) == len(names)
    # Extract the actual methods without importing CUDA/model dependencies.
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *methods], type_ignores=[])
    namespace = {}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


WINDOWS = load_methods("deploy/xvideo/models/pipeline.py", "Pipeline", {"_get_chunk_windows", "_get_last_chunk_window"})
Pipeline = type("Pipeline", (), {k: WINDOWS[k] for k in ("_get_chunk_windows", "_get_last_chunk_window")})
NEXT = load_methods("deploy/xvideo/serving/joyomni_streaming.py", "JoyOmniV2VStreamingSession", {"_next_selected_chunk_ids"})["_next_selected_chunk_ids"]


class StreamingWindowTests(unittest.TestCase):
    def test_matches_existing_batch_windows(self):
        for total in [1, 2, 3, 4, 5, 7, 8, 9, 17, 63, 128, 257]:
            for chunk in [1, 2, 4, 8]:
                for window in [1, 2, 3, 8, 512]:
                    for sink in [False, True]:
                        args = (total, chunk, window, sink)
                        with self.subTest(args=args):
                            self.assertEqual(Pipeline._get_last_chunk_window(*args), Pipeline._get_chunk_windows(*args)[-1])

    def test_work_is_bounded_after_millions_of_chunks(self):
        visited = []
        def bounded_range(*args):
            result = range(*args)
            self.assertLessEqual(len(result), 8, "Streaming lookup enumerated session history")
            visited.append(len(result))
            return result
        WINDOWS["range"] = bounded_range
        try:
            for total in [50_000, 1_000_000, 1_000_000_000]:
                for sink in [False, True]:
                    result = Pipeline._get_last_chunk_window(total, 1, 8, sink)
                    self.assertEqual(result["chunk_idx"], total - 1)
                    self.assertEqual(result["selected_chunk_ids"][-1], total - 1)
                    self.assertEqual(len(result["selected_chunk_ids"]), 8)
                    self.assertEqual(result["selected_chunk_ids"][0], 0 if sink else total - 8)
        finally:
            del WINDOWS["range"]
        self.assertEqual(len(visited), 6)

    def test_streaming_eviction_uses_direct_lookup(self):
        def fail_history(*args, **kwargs):
            raise AssertionError("Streaming must not build historical windows")
        session = SimpleNamespace(pipeline=SimpleNamespace(_get_last_chunk_window=Pipeline._get_last_chunk_window, _get_chunk_windows=fail_history), chunk_idx=50_000, chunk_size=1, local_window_size=2, global_sink_chunk=True)
        self.assertEqual(NEXT(session), [0, 50_001])
        self.assertEqual(NEXT(session, 1_000_000), [0, 1_000_001])

    def test_rejects_empty_or_invalid_window(self):
        for args in [(0, 1, 2, True), (10, 0, 2, False), (10, 1, 0, True)]:
            with self.assertRaises(ValueError):
                Pipeline._get_last_chunk_window(*args)


if __name__ == "__main__":
    unittest.main()
