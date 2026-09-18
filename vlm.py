from ollama import chat

image_path = "./Living-Room-Neutral-Sectional-Sofa-Haven-Corner-Walnut-Lyon-Coffee-Table-brand-x7291_CV1.webp"

prompt = """
请分析这张图片。
告诉我图片中有哪些主要物体。
只描述你能够确认看到的内容
"""

response = chat(
    model = "qwen3-vl:2b-instruct",
    messages = [
        {
            "role": "user",
            "content": prompt,
            "images": [image_path],
        }
    ],
)

print(response.message.content)
