from pydantic import BaseModel, Field


class DinoObject(BaseModel):
    label: str = Field(description="匹配到的文本短语")
    score: float = Field(description="检测置信度")
    bbox: list[float] = Field(
        min_length=4,
        max_length=4,
        description="[x1, y1, x2, y2] 原图像素坐标",
    )


class DinoResult(BaseModel):
    detection_count: int = Field(description="检测框数量（Grounding DINO 输出框，不是 mask）")
    inference_ms: float = Field(description="推理耗时，毫秒")
    device: str
    dtype: str
    prompt: str = Field(description="规范化后的文本提示")
    box_threshold: float
    text_threshold: float
    vram_model_mb: float | None = Field(
        default=None,
        description="模型加载后已分配显存，MB；CPU 为 null",
    )
    vram_peak_mb: float | None = Field(
        default=None,
        description="本次推理峰值显存，MB；CPU 为 null",
    )
    objects: list[DinoObject]
    boxes_path: str = Field(description="bounding box JSON 路径")
    annotated_path: str = Field(description="带检测框的图片路径")
