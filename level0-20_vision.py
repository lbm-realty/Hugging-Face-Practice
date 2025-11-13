from transformers import pipeline

pipe = pipeline("image-text-to-text", model="llava-hf/llava-interleave-qwen-0.5b-hf")
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "pic1.jpg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
outputs = pipe(text=messages, max_new_tokens=60, return_full_text=False)

print(outputs)
