from pydantic import BaseModel, Field


class SAMResult(BaseModel):
    mask_count: int = Field(description="生成的 mask 数量")
    inference_ms: float = Field(description="AMG 推理耗时，毫秒")
    device: str
    dtype: str
    points_per_batch: int
    points_per_crop: int
    vram_model_mb: float | None = Field(
        default=None,
        description="模型加载后已分配显存，MB；CPU 为 null",
    )
    vram_peak_mb: float | None = Field(
        default=None,
        description="本次推理峰值显存，MB；CPU 为 null",
    )
    overlay_path: str = Field(description="彩色叠加图路径")
    masks_path: str = Field(description="实例着色 mask 图路径")
