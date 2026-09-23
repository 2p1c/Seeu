from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.inference.siglip.schema import ClassScore


class SceneObject(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: int
    mask: dict[str, Any] = Field(description="COCO 未压缩 RLE。size 是 [高, 宽]，counts 按列展开")
    bounding_box: list[int] = Field(
        min_length=4,
        max_length=4,
        description="[x1, y1, x2, y2]，相对本次缩放后的画面，含两端像素",
    )
    embedding: list[float] = Field(description="DINOv3 ViT-S/16 的 pooler 向量")
    crop: str = Field(description="裁剪图的 JPEG base64")
    object_class: list[ClassScore] = Field(alias="class")
    description: str


class SceneState(BaseModel):
    timestamp: str
    filename: str
    width: int
    height: int
    scene_path: str
    objects: list[SceneObject]
