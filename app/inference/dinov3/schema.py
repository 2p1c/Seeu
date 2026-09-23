from pydantic import BaseModel, Field


class Dinov3Result(BaseModel):
    embedding: list[float] = Field(description="DINOv3 pooler 向量")
    dim: int
