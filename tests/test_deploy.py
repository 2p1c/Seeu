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
from app.inference.trt.plan import PLANS, deployment_status, select_yolo_weights, slot_mb


class PlanTest(unittest.TestCase):
    def test_each_model_fits_in_the_slot(self) -> None:
        limit = slot_mb()
        for item in PLANS:
            self.assertLessEqual(item.budget_mb, limit, item.name)

    def test_runtime_choices(self) -> None:
        chosen = {item.name: item.runtime for item in PLANS}
        self.assertEqual(chosen["yolo26s"], "tensorrt")
        self.assertEqual(chosen["siglip2-base"], "tensorrt")
        self.assertEqual(chosen["dinov3-vits16"], "tensorrt")
        self.assertEqual(chosen["sam2.1-tiny"], "pytorch")
        self.assertEqual(chosen["grounding-dino-tiny"], "pytorch")
        self.assertEqual(chosen["qwen3-vl-2b"], "ollama")

    def test_status_marks_missing_engines(self) -> None:
        status = deployment_status()
        self.assertEqual(status["policy"], "exclusive")
        yolo = next(item for item in status["models"] if item["name"] == "yolo26s")
        sam = next(item for item in status["models"] if item["name"] == "sam2.1-tiny")
        self.assertIs(yolo["engine_ready"], False)
        self.assertIsNone(sam["engine_ready"])

    def test_yolo_engine_needs_cuda_and_the_switch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pt = Path(tmp) / "yolo26s.pt"
            engine = pt.with_suffix(".engine")
            pt.write_bytes(b"pt")
            engine.write_bytes(b"engine")
            with patch.dict(os.environ, {"ROOMIND_TENSORRT": "0"}):
                self.assertEqual(select_yolo_weights(pt), pt)
            with (
                patch.dict(os.environ, {"ROOMIND_TENSORRT": "1"}),
                patch("app.inference.trt.plan.cuda_available", return_value=True),
            ):
                self.assertEqual(select_yolo_weights(pt), engine)
            with (
                patch.dict(os.environ, {"ROOMIND_TENSORRT": "1"}),
                patch("app.inference.trt.plan.cuda_available", return_value=False),
            ):
                self.assertEqual(select_yolo_weights(pt), pt)


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

        def release_yolo() -> None:
            calls.append("yolo")

        with patch("app.inference.vlm.adapter.unload_model", side_effect=lambda: calls.append("ollama")):
            claim_gpu("yolo", release_yolo)
            claim_gpu("yolo", release_yolo)
            claim_gpu("sam", lambda: calls.append("sam"), unload_ollama=False)
            claim_gpu("dino", lambda: calls.append("dino"))
        self.assertEqual(calls, ["ollama", "yolo", "sam", "ollama"])
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
                with GpuLease("yolo-camera", timeout_s=2):
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
                    if Path(lock).is_file() and Path(lock).read_text(encoding="utf-8").strip() == "yolo-camera":
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
                self.assertIn("yolo-camera", str(caught.exception))
            finally:
                proc.terminate()
                proc.wait(timeout=5)


class ExportCliTest(unittest.TestCase):
    def test_list_prints_and_build_requires_cuda(self) -> None:
        from app.inference.trt.export import main

        self.assertEqual(main(["--list"]), 0)
        with patch("app.inference.trt.export.cuda_available", return_value=False):
            self.assertEqual(main(["--only", "yolo"]), 1)


if __name__ == "__main__":
    unittest.main()
