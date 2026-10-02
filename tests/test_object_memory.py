"""需要一个空的 PostgreSQL + pgvector。测试会清空表，所以只认 ROOMIND_TEST_DATABASE_URL，不碰正式库。

    docker run -d --rm --name roomind-db-test -p 127.0.0.1:55432:5432 \\
      -e POSTGRES_USER=roomind -e POSTGRES_PASSWORD=roomind -e POSTGRES_DB=roomind pgvector/pgvector:pg17
    ROOMIND_TEST_DATABASE_URL=postgresql://roomind:roomind@127.0.0.1:55432/roomind python -m unittest tests.test_object_memory
"""

from __future__ import annotations

import base64
import os
import unittest
from datetime import datetime
from unittest.mock import patch

from app.inference.scene.schema import SceneObject, SceneState
from app.inference.siglip.schema import ClassScore
from app.object_memory import EMBEDDING_DIM, _connect, _position, latest_frame, save_scene

TEST_URL = os.environ.get("ROOMIND_TEST_DATABASE_URL")
CROP = base64.b64encode(b"\xff\xd8fake-jpeg").decode("ascii")


def _scene(filename: str, labels: list[str]) -> SceneState:
    return SceneState(
        timestamp="2026-09-26T23:00:00+08:00",
        filename=filename,
        width=300,
        height=300,
        scene_path="",
        objects=[
            SceneObject(
                id=index,
                mask={"size": [300, 300], "counts": [0, 10]},
                bounding_box=[10, 10, 50, 50],
                embedding=[0.5] * EMBEDDING_DIM,
                crop=CROP,
                object_class=[ClassScore(name=label, score=0.9), ClassScore(name="other", score=0.1)],
                description=f"一个{label}",
            )
            for index, label in enumerate(labels)
        ],
    )


@unittest.skipUnless(TEST_URL, "没有设置 ROOMIND_TEST_DATABASE_URL")
class ObjectMemoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self._env = patch.dict(os.environ, {"ROOMIND_DATABASE_URL": TEST_URL})
        self._env.start()
        with _connect() as conn:
            conn.execute("TRUNCATE frames RESTART IDENTITY CASCADE")

    def tearDown(self) -> None:
        self._env.stop()

    def test_empty_database_has_no_frame(self) -> None:
        self.assertIsNone(latest_frame())

    def test_latest_frame_returns_the_newest_scene(self) -> None:
        save_scene(_scene("a.jpg", ["chair"]))
        frame_id = save_scene(_scene("b.jpg", ["sofa", "lamp"]))
        frame = latest_frame()
        self.assertEqual(frame["id"], frame_id)
        self.assertEqual(frame["filename"], "b.jpg")
        self.assertEqual(
            datetime.fromisoformat(frame["captured_at"]),
            datetime.fromisoformat("2026-09-26T23:00:00+08:00"),
        )
        self.assertEqual([o["label"] for o in frame["objects"]], ["sofa", "lamp"])
        self.assertEqual(frame["objects"][0]["position"], "画面左上")
        self.assertEqual(frame["objects"][0]["bounding_box"], [10, 10, 50, 50])
        self.assertEqual(frame["objects"][0]["classes"][0], {"name": "sofa", "score": 0.9})

    def test_embedding_is_a_vector_and_crop_is_bytes(self) -> None:
        save_scene(_scene("a.jpg", ["chair"]))
        with _connect() as conn:
            row = conn.execute(
                "SELECT vector_dims(embedding) AS dims, embedding <=> %s::vector AS distance, crop FROM objects",
                (str([0.5] * EMBEDDING_DIM),),
            ).fetchone()
        self.assertEqual(row["dims"], EMBEDDING_DIM)
        self.assertAlmostEqual(row["distance"], 0.0, places=6)
        self.assertEqual(bytes(row["crop"]), b"\xff\xd8fake-jpeg")

    def test_scene_without_objects_is_still_a_frame(self) -> None:
        save_scene(_scene("empty.jpg", []))
        self.assertEqual(latest_frame()["objects"], [])

    def test_deleting_a_frame_deletes_its_objects(self) -> None:
        frame_id = save_scene(_scene("a.jpg", ["chair", "sofa"]))
        with _connect() as conn:
            conn.execute("DELETE FROM frames WHERE id = %s", (frame_id,))
            left = conn.execute("SELECT count(*) AS n FROM objects").fetchone()["n"]
        self.assertEqual(left, 0)


class PositionTest(unittest.TestCase):
    def test_position(self) -> None:
        self.assertEqual(_position([10, 10, 50, 50], 300, 300), "画面左上")
        self.assertEqual(_position([140, 140, 160, 160], 300, 300), "画面中间")
        self.assertEqual(_position([140, 250, 160, 290], 300, 300), "画面下方")
        self.assertEqual(_position([250, 140, 290, 160], 300, 300), "画面右侧")
        self.assertEqual(_position([250, 250, 290, 290], 300, 300), "画面右下")


if __name__ == "__main__":
    unittest.main()
