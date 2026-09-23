from pydantic import BaseModel, Field


class ClassScore(BaseModel):
    name: str = Field(description="候选类别名")
    score: float = Field(description="SigLIP sigmoid 分数，各类别之间不是互斥概率")


class SiglipResult(BaseModel):
    classes: list[ClassScore]
