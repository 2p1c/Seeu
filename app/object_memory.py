"""每个画面的物体记忆，存在 PostgreSQL + pgvector。只由感知服务直连，别的服务走 HTTP。"""

from __future__ import annotations

import base64
import os

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.inference.scene.schema import SceneState

DEFAULT_DATABASE_URL = "postgresql://roomind:roomind@127.0.0.1:5432/roomind"
EMBEDDING_DIM = 384  # DINOv3 ViT-S/16

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS frames (
    id          BIGSERIAL   PRIMARY KEY,
    captured_at TIMESTAMPTZ NOT NULL,
    filename    TEXT        NOT NULL,
    width       INTEGER     NOT NULL,
    height      INTEGER     NOT NULL
);
CREATE TABLE IF NOT EXISTS objects (
    id          BIGSERIAL   PRIMARY KEY,
    frame_id    BIGINT      NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
    idx         INTEGER     NOT NULL,
    label       TEXT        NOT NULL,
    classes     JSONB       NOT NULL,
    bbox        INTEGER[]   NOT NULL,
    mask        JSONB       NOT NULL,
    embedding   vector({EMBEDDING_DIM}) NOT NULL,
    crop        BYTEA       NOT NULL,
    description TEXT        NOT NULL,
    UNIQUE (frame_id, idx)
);
CREATE INDEX IF NOT EXISTS objects_label ON objects(label);
"""

_schema_ready = False


def database_url() -> str:
    return os.environ.get("ROOMIND_DATABASE_URL", DEFAULT_DATABASE_URL)


def _connect() -> psycopg.Connection:
    """每次调用开一个连接。感知服务一次只处理一个推理请求，用不着连接池。"""
    global _schema_ready
    conn = psycopg.connect(database_url(), row_factory=dict_row, connect_timeout=5)
    if not _schema_ready:
        conn.execute(SCHEMA)
        conn.commit()
        _schema_ready = True
    return conn


def save_scene(state: SceneState) -> int:
    """一个画面一条 frames 记录，画面里每个物体一条 objects 记录。返回 frame id。"""
    with _connect() as conn:
        frame_id = conn.execute(
            "INSERT INTO frames (captured_at, filename, width, height) VALUES (%s, %s, %s, %s) RETURNING id",
            (state.timestamp, state.filename, state.width, state.height),
        ).fetchone()["id"]
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO objects (frame_id, idx, label, classes, bbox, mask, embedding, crop, description)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s::vector, %s, %s)",
                [
                    (
                        frame_id,
                        obj.id,
                        obj.object_class[0].name if obj.object_class else "",
                        Jsonb([c.model_dump() for c in obj.object_class]),
                        obj.bounding_box,
                        Jsonb(obj.mask),
                        str(obj.embedding),
                        base64.b64decode(obj.crop),
                        obj.description,
                    )
                    for obj in state.objects
                ],
            )
    return frame_id


def latest_frame() -> dict | None:
    """最新一个画面和它的物体。不含 mask、向量和裁剪图，给 Agent 和页面用。"""
    with _connect() as conn:
        frame = conn.execute("SELECT * FROM frames ORDER BY id DESC LIMIT 1").fetchone()
        if frame is None:
            return None
        rows = conn.execute(
            "SELECT idx, label, classes, bbox, description FROM objects WHERE frame_id = %s ORDER BY idx",
            (frame["id"],),
        ).fetchall()
    return {
        **frame,
        "captured_at": frame["captured_at"].isoformat(),
        "objects": [
            {
                "id": row["idx"],
                "label": row["label"],
                "classes": row["classes"],
                "bounding_box": row["bbox"],
                "position": _position(row["bbox"], frame["width"], frame["height"]),
                "description": row["description"],
            }
            for row in rows
        ],
    }


def _position(bbox: list[int], width: int, height: int) -> str:
    """框中心落在九宫格的哪一格，比如"画面左上"。"""
    cx = (bbox[0] + bbox[2]) / 2 / max(width, 1)
    cy = (bbox[1] + bbox[3]) / 2 / max(height, 1)
    col = "左" if cx < 1 / 3 else "右" if cx > 2 / 3 else ""
    row = "上" if cy < 1 / 3 else "下" if cy > 2 / 3 else ""
    if not col and not row:
        return "画面中间"
    if not col:
        return f"画面{row}方"
    if not row:
        return f"画面{col}侧"
    return f"画面{col}{row}"
