from __future__ import annotations

import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from app.inference.sam.masks import SegmentInstance, bbox_from_mask, mask_to_rle
from app.inference.scene.codec import crop_image, jpeg_base64, render_labeled_image
from app.inference.scene import progress
from app.inference.scene.schema import SceneObject
from app.inference.scene.select import select_instances
from app.inference.siglip.schema import ClassScore


class MaskCodecTest(unittest.TestCase):
    def test_rle_and_bbox(self) -> None:
        mask = np.zeros((2, 2), dtype=bool)
        mask[0, 0] = True
        self.assertEqual(bbox_from_mask(mask), [0, 0, 0, 0])
        self.assertEqual(mask_to_rle(mask), {"size": [2, 2], "counts": [0, 1, 3]})

    def test_empty_mask_has_no_box(self) -> None:
        self.assertIsNone(bbox_from_mask(np.zeros((3, 3), dtype=bool)))


class SelectTest(unittest.TestCase):
    def test_keeps_larger_box_when_overlap_is_high(self) -> None:
        large = SimpleNamespace(bbox=[0, 0, 19, 19])
        small = SimpleNamespace(bbox=[1, 1, 18, 18])
        other = SimpleNamespace(bbox=[30, 30, 39, 39])
        kept = select_instances(
            [small, other, large],
            image_area=80 * 80,
            max_objects=12,
            min_area_ratio=0,
        )
        self.assertEqual([item.bbox for item in kept], [[0, 0, 19, 19], [30, 30, 39, 39]])

    def test_drops_specks(self) -> None:
        speck = SimpleNamespace(bbox=[0, 0, 2, 2])
        kept = select_instances([speck], image_area=1000 * 1000, max_objects=12)
        self.assertEqual(kept, [])


class CropTest(unittest.TestCase):
    def test_crop_includes_both_edges(self) -> None:
        image = Image.new("RGB", (10, 8), (0, 0, 0))
        image.putpixel((1, 2), (255, 0, 0))
        image.putpixel((3, 4), (0, 255, 0))
        crop = crop_image(image, [1, 2, 3, 4])
        self.assertEqual(crop.size, (3, 3))
        self.assertEqual(crop.getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(crop.getpixel((2, 2)), (0, 255, 0))
        self.assertTrue(jpeg_base64(crop))


class LabeledImageTest(unittest.TestCase):
    def test_box_edge_uses_the_label_color(self) -> None:
        image = Image.new("RGB", (20, 20), (10, 20, 30))
        painted = render_labeled_image(image, [([2, 2, 8, 8], "0 chair")])
        self.assertEqual(painted.size, (20, 20))
        self.assertEqual(painted.getpixel((2, 2)), (20, 108, 84))
        self.assertEqual(painted.getpixel((19, 19)), (10, 20, 30))


class SceneSchemaTest(unittest.TestCase):
    def test_class_alias(self) -> None:
        payload = SceneObject(
            id=0,
            mask={"size": [1, 1], "counts": [0, 1]},
            bounding_box=[0, 0, 0, 0],
            embedding=[0.1],
            crop="abc",
            object_class=[ClassScore(name="chair", score=0.5)],
            description="一把椅子",
        ).model_dump(by_alias=True)
        self.assertEqual(payload["class"][0]["name"], "chair")
        self.assertNotIn("object_class", payload)


class ScenePipelineTest(unittest.TestCase):
    def test_perceive_assembles_one_object(self) -> None:
        from app.inference.scene.service import SceneService

        mask = np.zeros((40, 40), dtype=bool)
        mask[0:10, 0:10] = True
        work = Image.new("RGB", (40, 40), (10, 20, 30))

        class FakeSAM:
            def instances(self, image, filename, points_per_batch, points_per_crop, max_size):
                return work, [SegmentInstance(mask=mask, score=0.9, bbox=[0, 0, 9, 9])]

            def release(self) -> None:
                return None

        class FakeSiglip:
            def classify(self, images, top_k=3):
                self.stage = progress.snapshot()["stage"]
                return [[ClassScore(name="chair", score=0.81)] for _ in images]

            def release(self) -> None:
                return None

        class FakeDinov3:
            def embed(self, images):
                return [[0.25, -0.5] for _ in images]

            def release(self) -> None:
                return None

        class FakeVLM:
            def describe_object(self, scene, crop, bbox):
                self.bbox = bbox
                self.crop_size = crop.size
                self.progress = progress.snapshot()
                return "画面左上角有一把椅子"

        vlm = FakeVLM()
        siglip = FakeSiglip()
        image = BytesIO()
        work.save(image, format="PNG")
        with (
            patch("app.inference.scene.service.unload_model"),
            patch("app.inference.scene.service.ollama_memory_mb", return_value=(10.0, 8.0)),
        ):
            state = SceneService(FakeSAM(), siglip, FakeDinov3(), vlm).perceive(
                image.getvalue(),
                "room.png",
                "2026-09-23T14:00:00+08:00",
                min_area_ratio=0,
            )
        profile_path = Path(state.scene_path).with_name("room_profile.log")
        try:
            body = state.model_dump(by_alias=True)
            self.assertEqual(body["timestamp"], "2026-09-23T14:00:00+08:00")
            self.assertEqual(body["width"], 40)
            self.assertEqual(body["height"], 40)
            self.assertEqual(len(body["objects"]), 1)
            item = body["objects"][0]
            self.assertEqual(item["bounding_box"], [0, 0, 9, 9])
            self.assertEqual(item["embedding"], [0.25, -0.5])
            self.assertEqual(item["class"][0]["name"], "chair")
            self.assertEqual(item["description"], "画面左上角有一把椅子")
            self.assertEqual(item["mask"]["size"], [40, 40])
            self.assertTrue(item["crop"])
            self.assertEqual(vlm.bbox, [0, 0, 9, 9])
            self.assertEqual(vlm.crop_size, (10, 10))
            text = profile_path.read_text(encoding="utf-8")
            self.assertIn("sam", text)
            self.assertIn("vlm-0", text)
            self.assertIn("8.0", text)
            self.assertEqual(siglip.stage, "siglip")
            self.assertEqual(vlm.progress["stage"], "vlm")
            self.assertEqual(vlm.progress["index"], 0)
            self.assertEqual(vlm.progress["total"], 1)
            done = progress.snapshot()
            self.assertEqual(done["stage"], "done")
            self.assertEqual(done["details"]["sam"], "1 个掩码")
            self.assertFalse(done["running"])
            for name in ("sam", "siglip", "dinov3", "vlm"):
                self.assertGreaterEqual(done["seconds"][name], 0)
            self.assertEqual(done["partial"]["sam"]["masks"], 1)
            self.assertEqual(done["partial"]["sam"]["overlay"], "room_overlay.jpg")
            live = done["partial"]["objects"][0]
            self.assertEqual(live["class"][0]["name"], "chair")
            self.assertEqual(live["embedding_dim"], 2)
            self.assertEqual(live["description"], "画面左上角有一把椅子")
            saved = Image.open(state.image_path)
            self.assertEqual(saved.size, (40, 40))
            saved.close()
        finally:
            Path(state.scene_path).unlink(missing_ok=True)
            Path(state.image_path).unlink(missing_ok=True)
            profile_path.unlink(missing_ok=True)


class LatestSceneTest(unittest.TestCase):
    def test_picks_the_newest_scene_file(self) -> None:
        import json
        import os
        import tempfile
        import time

        from app.inference.scene.service import latest_saved_scene

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            older = root / "old_scene.json"
            newer = root / "new_scene.json"
            older.write_text(json.dumps({"filename": "old"}), encoding="utf-8")
            time.sleep(0.02)
            newer.write_text(json.dumps({"filename": "new"}), encoding="utf-8")
            os.utime(older, (1, 1))
            found = latest_saved_scene(root)
            self.assertEqual(found["filename"], "new")
            self.assertIsNone(latest_saved_scene(root / "missing"))


class SceneProgressTest(unittest.TestCase):
    def test_service_time_stops_and_sums(self) -> None:
        import time

        progress.begin()
        progress.enter("sam")
        time.sleep(0.03)
        progress.mark("sam", detail="1 个掩码")
        progress.leave("sam")
        frozen = progress.snapshot()["seconds"]["sam"]
        time.sleep(0.05)
        self.assertEqual(progress.snapshot()["seconds"]["sam"], frozen)

        progress.enter("vlm")
        time.sleep(0.03)
        progress.leave("vlm")
        progress.enter("vlm")
        time.sleep(0.03)
        progress.leave("vlm")
        progress.finish("2 个物体")
        self.assertGreaterEqual(progress.snapshot()["seconds"]["vlm"], 0.05)
        self.assertEqual(progress.snapshot()["seconds"]["sam"], frozen)

    def test_failure_keeps_the_active_stage(self) -> None:
        from app.inference.scene.service import SceneService

        class BrokenSAM:
            def instances(self, image, filename, points_per_batch, points_per_crop, max_size):
                raise RuntimeError("sam failed")

            def release(self) -> None:
                return None

        class Idle:
            def release(self) -> None:
                return None

        profile = Path(__file__).resolve().parents[1] / "tests" / "tmp" / "broken_profile.log"
        try:
            with (
                patch("app.inference.scene.service.unload_model"),
                self.assertRaises(RuntimeError),
            ):
                SceneService(BrokenSAM(), Idle(), Idle(), Idle()).perceive(b"not-an-image", "broken.png")
            body = progress.snapshot()
            self.assertEqual(body["stage"], "sam")
            self.assertEqual(body["error"], "sam failed")
            self.assertFalse(body["running"])
        finally:
            profile.unlink(missing_ok=True)


class ScoreListTest(unittest.TestCase):
    def test_tensor_scores_do_not_use_boolean_context(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        from app.inference.sam.generator import _score_list

        scores = torch.tensor([0.2, 0.9])
        values = _score_list(scores)
        self.assertEqual(len(values), 2)
        self.assertAlmostEqual(values[0], 0.2, places=4)
        self.assertAlmostEqual(values[1], 0.9, places=4)
        self.assertEqual(_score_list(None), [])


class OpenRgbTest(unittest.TestCase):
    def test_orientation_6_matches_display(self) -> None:
        from app.inference.image import open_rgb

        image = Image.new("RGB", (2, 4), (255, 0, 0))
        exif = image.getexif()
        exif[274] = 6
        buf = BytesIO()
        image.save(buf, format="JPEG", exif=exif)
        upright = open_rgb(buf.getvalue())
        self.assertEqual(upright.size, (4, 2))


if __name__ == "__main__":
    unittest.main()
