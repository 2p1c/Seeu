from pydantic import BaseModel, ConfigDict, Field


class DetectedObject(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    class_name: str = Field(alias="class", description="COCO80 类别名")
    bbox: list[float] = Field(
        min_length=4,
        max_length=4,
        description="[x1, y1, x2, y2] 原图像素坐标",
    )
    track_id: int | None = Field(default=None, description="ByteTrack id，单图检测为空")


class DetectionResult(BaseModel):
    timestamp: str = Field(description="检测时刻，ISO 8601")
    objects: list[DetectedObject]
    annotated_image: str | None = Field(
        default=None,
        description="带标注框的 JPEG，base64；终端打印时可省略",
    )
    saved_path: str | None = Field(
        default=None,
        description="标注图保存路径",
    )
