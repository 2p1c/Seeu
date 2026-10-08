from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.inference.trt.budget import claim_gpu, current_claim, drop_claims
from app.inference.trt.cache import read_text_cache, save_text_cache
from app.inference.trt.plan import PLANS, deployment_status, slot_mb, use_engine


class PlanTest(unittest.TestCase):
    def test_each_model_fits_in_the_slot(self) -> None:
        limit = slot_mb()
        for item in PLANS:
            self.assertLessEqual(item.budget_mb, limit, item.name)

    def test_runtime_choices(self) -> None:
        chosen = {item.name: item.runtime for item in PLANS}
        self.assertEqual(chosen["siglip2-base"], "tensorrt")
        self.assertEqual(chosen["dinov3-vits16"], "tensorrt")
        self.assertEqual(chosen["sam2.1-tiny"], "pytorch")
        self.assertEqual(chosen["grounding-dino-tiny"], "pytorch")
        self.assertEqual(chosen["qwen3-vl-2b"], "ollama")

    def test_status_marks_missing_engines(self) -> None:
        status = deployment_status()
        self.assertEqual(status["policy"], "exclusive")
        dinov3 = next(item for item in status["models"] if item["name"] == "dinov3-vits16")
        sam = next(item for item in status["models"] if item["name"] == "sam2.1-tiny")
        self.assertIs(dinov3["engine_ready"], False)
        self.assertIsNone(sam["engine_ready"])

    def test_engine_needs_cuda_and_the_switch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            engine = Path(tmp) / "model.engine"
            engine.write_bytes(b"engine")
            with patch.dict(os.environ, {"ROOMIND_TENSORRT": "0"}):
                self.assertFalse(use_engine(engine))
            with (
                patch.dict(os.environ, {"ROOMIND_TENSORRT": "1"}),
                patch("app.inference.trt.plan.cuda_available", return_value=True),
            ):
                self.assertTrue(use_engine(engine))
            with (
                patch.dict(os.environ, {"ROOMIND_TENSORRT": "1"}),
                patch("app.inference.trt.plan.cuda_available", return_value=False),
            ):
                self.assertFalse(use_engine(engine))


class CacheTest(unittest.TestCase):
    def test_label_change_invalidates_siglip_cache(self) -> None:
        from app.inference.siglip.labels import PROMPTS, ROOM_LABELS

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "text.npz"
            features = np.ones((len(ROOM_LABELS), 4), dtype=np.float32)
            save_text_cache(path, features, 1.5, np.float32(0.1), ROOM_LABELS, PROMPTS)
            cached = read_text_cache(path, ROOM_LABELS, PROMPTS)
            self.assertIsNotNone(cached)
            self.assertEqual(cached[1], 1.5)
            self.assertIsNone(read_text_cache(path, ROOM_LABELS + ("extra",), PROMPTS))
            self.assertIsNone(read_text_cache(path, ROOM_LABELS, PROMPTS + ("a photo of a extra",)))


class BudgetTest(unittest.TestCase):
    def tearDown(self) -> None:
        drop_claims()

    def test_switch_releases_the_previous_model(self) -> None:
        calls: list[str] = []

        def release_siglip() -> None:
            calls.append("siglip")

        with patch("app.inference.vlm.adapter.unload_model", side_effect=lambda: calls.append("ollama")):
            claim_gpu("siglip", release_siglip)
            claim_gpu("siglip", release_siglip)
            claim_gpu("sam", lambda: calls.append("sam"), unload_ollama=False)
            claim_gpu("dino", lambda: calls.append("dino"))
        self.assertEqual(calls, ["ollama", "siglip", "sam", "ollama"])
        self.assertEqual(current_claim(), "dino")


class LeaseTest(unittest.TestCase):
    def test_second_process_reports_the_holder(self) -> None:
        from app.inference.trt.lease import GpuBusy, GpuLease

        with tempfile.TemporaryDirectory() as tmp:
            lock = str(Path(tmp) / "gpu.lock")
            env = os.environ.copy()
            env["ROOMIND_GPU_LOCK"] = lock
            code = textwrap.dedent(
                """
                import time
                from app.inference.trt.lease import GpuLease
                with GpuLease("trt-export", timeout_s=2):
                    time.sleep(30)
                """
            )
            proc = subprocess.Popen(
                [sys.executable, "-c", code],
                env=env,
                cwd=str(Path(__file__).resolve().parents[1]),
            )
            try:
                deadline = time.time() + 5
                while time.time() < deadline:
                    if Path(lock).is_file() and Path(lock).read_text(encoding="utf-8").strip() == "trt-export":
                        break
                    if proc.poll() is not None:
                        self.fail(f"holder exited {proc.returncode}")
                    time.sleep(0.05)
                else:
                    self.fail("holder did not acquire the lease")
                with patch.dict(os.environ, {"ROOMIND_GPU_LOCK": lock}):
                    with self.assertRaises(GpuBusy) as caught:
                        with GpuLease("perception", timeout_s=0.3):
                            pass
                self.assertIn("trt-export", str(caught.exception))
            finally:
                proc.terminate()
                proc.wait(timeout=5)


class ExportCliTest(unittest.TestCase):
    def test_list_prints_and_build_requires_cuda(self) -> None:
        from app.inference.trt.export import main

        self.assertEqual(main(["--list"]), 0)
        with patch("app.inference.trt.export.cuda_available", return_value=False):
            self.assertEqual(main(["--only", "siglip"]), 1)


class TensorrtImportTest(unittest.TestCase):
    def test_system_dir_only_if_package_exists(self) -> None:
        from app.inference.trt.engine import _system_tensorrt_dir

        found = _system_tensorrt_dir()
        if found is not None:
            self.assertTrue((found / "tensorrt").is_dir())


if __name__ == "__main__":
    unittest.main()
