from pydantic import BaseModel, Field


class VLMObservation(BaseModel):
    description: str = Field(description="根据给定物体和位置，对物体本身的描述")


class ObjectDescription(BaseModel):
    description: str = Field(description="这件物品本身，以及它在画面中的位置")


class VLMResult(BaseModel):
    time: str
    objects: list[str] = Field(description="要描述的物体，可以是自然语言短语")
    location: str
    description: str
