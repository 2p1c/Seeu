from pydantic import BaseModel, Field


class VLMResult(BaseModel):
    objects: list[str] = Field(description="物体类别")
    description: str = Field(description="画面描述")
