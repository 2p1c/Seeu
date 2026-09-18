from typing import Literal, get_args

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

HOUSEHOLD_OBJECTS: tuple[str, ...] = get_args(HouseholdObject)


class VLMOllamaOutput(BaseModel):
    objects: dict[HouseholdObject, Literal[1, 2, 3]] = Field(
        default_factory=dict,
        description="日常用品到数量，每个类别 1 到 3",
        json_schema_extra={
            "propertyNames": {"enum": list(HOUSEHOLD_OBJECTS)},
            "additionalProperties": {"type": "integer", "enum": [1, 2, 3]},
        },
    )
    location: str = Field(description="位置描述")
    description: str = Field(description="物体本身的描述")


class VLMObservation(BaseModel):
    objects: list[HouseholdObject] = Field(description="日常用品类别")
    location: str = Field(description="位置描述")
    description: str = Field(description="物体本身的描述")


class VLMResult(BaseModel):
    time: str
    objects: list[HouseholdObject]
    location: str
    description: str
