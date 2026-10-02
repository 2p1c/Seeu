from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from app.inference.dinov3.service import Dinov3Service
from app.inference.sam.generator import (
    DEFAULT_MAX_SIZE,
    DEFAULT_POINTS_PER_BATCH,
    DEFAULT_POINTS_PER_CROP,
    SAVE_DIR,
)
from app.inference.sam.masks import mask_to_rle
from app.inference.sam.service import SAMService
from app.inference.scene.codec import crop_image, jpeg_base64, render_labeled_image, save_jpeg
from app.inference.scene import progress
from app.inference.scene.schema import SceneObject, SceneState
from app.inference.scene.select import (
    DEFAULT_MAX_OBJECTS,
    DEFAULT_MIN_AREA_RATIO,
    select_instances,
)
from app.inference.siglip.classifier import TOP_K
from app.inference.siglip.service import SiglipService
from app.inference.scene.profile import SceneProfile
from app.inference.vlm.adapter import OllamaQwen3VLAdapter, ollama_memory_mb, unload_model

log = logging.getLogger("roommind.scene")


def latest_saved_scene(directory: Path | None = None) -> dict | None:
    """最近一次写完的场景结果。同一张图会覆盖自己的文件，多张图按修改时间取最新。"""
    root = directory or SAVE_DIR
    files = [path for path in root.glob("*_scene.json") if path.is_file()]
    if not files:
        return None
    newest = max(files, key=lambda path: path.stat().st_mtime)
    return json.loads(newest.read_text(encoding="utf-8"))


def _safe_stem(stem: str) -> str:
    safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in stem).strip("_")
    return safe or "scene"


def _num(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.1f}"


def _publish_objects(selected, classes, embeddings, descriptions) -> None:
    rows = []
    for index, item in enumerate(selected):
        row = {"id": index, "bounding_box": list(item.bbox)}
        if classes is not None:
            row["class"] = [
                {"name": label.name, "score": round(float(label.score), 3)}
                for label in classes[index]
            ]
        if embeddings is not None:
            row["embedding_dim"] = len(embeddings[index])
        if descriptions is not None and descriptions[index]:
            row["description"] = descriptions[index]
        rows.append(row)
    progress.publish("objects", rows)


def _timestamp(value: str | None) -> str:
    if value and value.strip():
        return value.strip()
    return datetime.now().astimezone().isoformat(timespec="seconds")


class SceneService:
    def __init__(
        self,
        sam: SAMService,
        siglip: SiglipService,
        dinov3: Dinov3Service,
        vlm: OllamaQwen3VLAdapter,
        also_release: tuple = (),
    ) -> None:
        self.sam = sam
        self.siglip = siglip
        self.dinov3 = dinov3
        self.vlm = vlm
        self._also_release = also_release

    def perceive(
        self,
        image: bytes,
        filename: str | None = None,
        event_time: str | None = None,
        points_per_batch: int = DEFAULT_POINTS_PER_BATCH,
        points_per_crop: int = DEFAULT_POINTS_PER_CROP,
        max_size: int = DEFAULT_MAX_SIZE,
        max_objects: int = DEFAULT_MAX_OBJECTS,
        min_area_ratio: float = DEFAULT_MIN_AREA_RATIO,
    ) -> SceneState:
        if not image:
            raise ValueError("empty image")
        unload_model()
        self._release_torch()
        profile = SceneProfile(filename or "image")
        progress.begin()
        try:
            return self._perceive(
                image,
                filename=filename,
                event_time=event_time,
                points_per_batch=points_per_batch,
                points_per_crop=points_per_crop,
                max_size=max_size,
                max_objects=max_objects,
                min_area_ratio=min_area_ratio,
                profile=profile,
            )
        except Exception as exc:
            progress.fail(str(exc))
            raise
        finally:
            self._release_torch()
            self._write_profile(profile, filename)

    def _perceive(
        self,
        image: bytes,
        *,
        filename: str | None,
        event_time: str | None,
        points_per_batch: int,
        points_per_crop: int,
        max_size: int,
        max_objects: int,
        min_area_ratio: float,
        profile: SceneProfile,
    ) -> SceneState:
        progress.mark("sam")
        with profile.step("sam") as step:
            progress.enter("sam")
            try:
                work, instances = self.sam.instances(
                    image,
                    filename,
                    points_per_batch,
                    points_per_crop,
                    max_size,
                )
            finally:
                progress.leave("sam")
            step.detail = f"masks={len(instances)}"
        self.sam.release()
        progress.mark("sam", detail=f"{len(instances)} 个掩码")
        selected = select_instances(
            instances,
            image_area=work.width * work.height,
            max_objects=max_objects,
            min_area_ratio=min_area_ratio,
        )
        crops = [crop_image(work, item.bbox) for item in selected]
        log.info("scene objects=%d from masks=%d", len(selected), len(instances))
        stem = _safe_stem(Path(filename or "scene").stem)
        progress.publish(
            "sam",
            {
                "masks": len(instances),
                "selected": len(selected),
                "overlay": f"{stem}_overlay.jpg" if filename else "",
            },
        )

        progress.mark("siglip", detail=f"{len(selected)} 个物体")
        with profile.step("siglip") as step:
            progress.enter("siglip")
            try:
                classes = self.siglip.classify(crops, top_k=TOP_K) if crops else []
            finally:
                progress.leave("siglip")
            step.detail = f"objects={len(classes)}"
        self.siglip.release()
        _publish_objects(selected, classes, None, None)
        progress.mark("dinov3", detail=f"{len(crops)} 个物体")
        with profile.step("dinov3") as step:
            progress.enter("dinov3")
            try:
                embeddings = self.dinov3.embed(crops) if crops else []
            finally:
                progress.leave("dinov3")
            step.detail = f"objects={len(embeddings)}"
        self.dinov3.release()
        if not (len(selected) == len(crops) == len(classes) == len(embeddings)):
            raise RuntimeError("分类或向量数量与物体数量不一致")
        descriptions: list[str] = [""] * len(selected)
        _publish_objects(selected, classes, embeddings, descriptions)

        objects: list[SceneObject] = []
        if not selected:
            progress.mark("vlm", detail="0 个物体")
        for index, (item, crop, labels, embedding) in enumerate(
            zip(selected, crops, classes, embeddings)
        ):
            progress.mark(
                "vlm",
                detail=f"{index + 1}/{len(selected)}",
                index=index,
                total=len(selected),
            )
            with profile.step(f"vlm-{index}") as step:
                progress.enter("vlm")
                try:
                    description = self.vlm.describe_object(work, crop, item.bbox)
                    size_mb, vram_mb = ollama_memory_mb()
                    step.vram_override = vram_mb
                    step.detail = f"ollama_size_mb={_num(size_mb)} chars={len(description)}"
                finally:
                    progress.leave("vlm")
            descriptions[index] = description
            _publish_objects(selected, classes, embeddings, descriptions)
            objects.append(
                SceneObject(
                    id=index,
                    mask=mask_to_rle(item.mask),
                    bounding_box=item.bbox,
                    embedding=embedding,
                    crop=jpeg_base64(crop),
                    object_class=labels,
                    description=description,
                )
            )

        image_file = SAVE_DIR / f"{stem}_scene.jpg"
        save_jpeg(
            render_labeled_image(
                work,
                [
                    (
                        obj.bounding_box,
                        f"{obj.id} {obj.object_class[0].name if obj.object_class else obj.id}",
                    )
                    for obj in objects
                ],
            ),
            image_file,
        )
        state = SceneState(
            timestamp=_timestamp(event_time),
            filename=filename or "image",
            width=work.width,
            height=work.height,
            scene_path="",
            image_path=str(image_file),
            objects=objects,
        )
        scene_path = SAVE_DIR / f"{stem}_scene.json"
        scene_path.parent.mkdir(parents=True, exist_ok=True)
        state.scene_path = str(scene_path)
        scene_path.write_text(
            state.model_dump_json(by_alias=True, indent=2),
            encoding="utf-8",
        )
        log.info("scene saved %s objects=%d", scene_path, len(objects))
        progress.finish(f"{len(objects)} 个物体")
        return state

    def _write_profile(self, profile: SceneProfile, filename: str | None) -> None:
        stem = _safe_stem(Path(filename or "scene").stem)
        path = profile.write(SAVE_DIR / f"{stem}_profile.log")
        logging.getLogger("uvicorn.error").info("scene profile %s\n%s", path, profile.render().rstrip())

    def _release_torch(self) -> None:
        for service in (self.sam, self.siglip, self.dinov3, *self._also_release):
            release = getattr(service, "release", None)
            if release is not None:
                release()
