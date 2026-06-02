import os
import time
from datetime import datetime
from pathlib import Path
from openai import OpenAI

PROJECT_DIR = Path(__file__).resolve().parent
PROMPT_OUTPUT = PROJECT_DIR / "gen_prompt_output.txt"

# Read config from environment
base_url = os.environ["NVIDIA_BASE_URL"]
api_key = os.environ["NVIDIA_API_KEY"]
model = os.environ["NVIDIA_MODEL"]

client = OpenAI(base_url=base_url, api_key=api_key)

# Record start time
start_time = time.time()
start_datetime = datetime.now()
print(f"Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")

system_prompt = """
Generate a prompt which used in text-to-image, to create an image to well describe the word.
Because abstract word maybe not easy to describe in a single scene, you can use multiple scenes
in one image in case it needs.
"""
word = "appreciate"
meaning = "thankful"
user_prompt = f"""
The word is '{word}', meaning is '{meaning}'.
"""

print(f"Model: {model}")
print("Generating prompts...")

response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
    temperature=0.7,
    max_tokens=512,
)

output_text = response.choices[0].message.content
tokens_used = response.usage.total_tokens

print("-" * 50)
print(output_text)
print("-" * 50)
print(f"Tokens used: {tokens_used}")

# Write word + generated prompt to shared file for z-image.py
with open(PROMPT_OUTPUT, "w") as f:
    f.write(word + "\n")
    f.write(output_text.strip() + "\n")
print(f"Prompt written to: {PROMPT_OUTPUT}")

# Record end time and print timing summary
end_time = time.time()
end_datetime = datetime.now()
duration_seconds = end_time - start_time
duration_minutes = duration_seconds / 60.0

print(f"End time: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Duration: {duration_minutes:.2f} minutes ({duration_seconds:.1f} seconds)")
