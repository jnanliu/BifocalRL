import os
from openai import OpenAI
messages = [
    {
        "role": "system", "content": "You are a highly intelligent AI assistant.",
    },
    {
        "role": "user", "content": "Is 13371337 a prime number?"
    },
]
client = OpenAI(
    base_url="https://api.x.ai/v1",
    api_key=os.getenv("XAI_API_KEY"),
)

completion = client.chat.completions.create(
    model="grok-3-mini-fast-high-beta",  # or "grok-3-mini-fast-beta"
    # reasoning_effort="high", # "high" or "low" or None for grok-3-beta
    messages=messages,
    temperature=0.7,
)
print("Reasoning Content:")
print(completion.choices[0].message.reasoning_content)
print("\nFinal Response:")
print(completion.choices[0].message.content)
print("\nNumber of completion tokens (input):")
print(completion.usage.completion_tokens)
print("\nNumber of reasoning tokens (input):")
print(completion.usage.completion_tokens_details.reasoning_tokens)
