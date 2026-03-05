# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("image-text-to-text", model="lapa-llm/lapa-v0.1.2-instruct")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Чи слово в кутових дужках є евфемізмом? Поясни. Нападники <задвохсотилися>"}
        ]
    },
]
print(pipe(text=messages))

