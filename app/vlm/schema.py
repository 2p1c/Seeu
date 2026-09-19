from typing import Literal

from pydantic import BaseModel, Field

HouseholdObject = Literal[
    "person",
    "cat",
    "dog",
    "backpack",
    "umbrella",
    "handbag",
    "suitcase",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


class VLMObservation(BaseModel):
    description: str = Field(description="根据给定物体和位置，对物体本身的描述")


class VLMResult(BaseModel):
    time: str
    objects: list[HouseholdObject]
    location: str
    description: str
