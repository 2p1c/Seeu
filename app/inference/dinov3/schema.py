from pydantic import BaseModel, Field


class Dinov3Result(BaseModel):
    embedding: list[float] = Field(description="DINOv3 pooler 向量")
    dim: int


class Dinov3CompareResult(BaseModel):
    dim: int
    cosine_similarity: float = Field(description="两张图向量的余弦相似度，1 为同方向，0 为无关")
    embedding_a: list[float] = Field(description="第一张图的向量")
    embedding_b: list[float] = Field(description="第二张图的向量")
