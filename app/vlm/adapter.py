from ollama import chat

from app.vlm.schema import VLMResult

MODEL_NAME = "qwen3-vl:2b-instruct"


class OllamaQwen3VLAdapter:
    def analyze(self, image: bytes, prompt: str) -> VLMResult:
        response = chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "根据图片和用户要求作答。"
                        "objects 填能确认看到的物体类别；"
                        "description 填对画面的描述。"
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image],
                },
            ],
            format=VLMResult.model_json_schema(),
        )
        return VLMResult.model_validate_json(response.message.content)
