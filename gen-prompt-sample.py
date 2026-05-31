import os
import time
from datetime import datetime
from openai import OpenAI

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
You are a creative comic strip writer. Given a story sentence, output exactly 4 text prompts,
one per panel of a 4-panel comic strip. Each prompt should describe the visual scene for that
panel — characters, setting, emotion, and action. Keep each prompt concise (1-2 sentences).
Output ONLY the 4 prompts, each on its own line prefixed with "Panel N: ".
"""

user_prompt = """
Based on below sentence, setup 4 text for comic strip.
"She didn't appreciate how difficult the math competition would be until she started preparing."
"""

print(f"Model: {model}")
print("Generating comic strip prompts...")

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

# Record end time and print timing summary
end_time = time.time()
end_datetime = datetime.now()
duration_seconds = end_time - start_time
duration_minutes = duration_seconds / 60.0

print(f"End time: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Duration: {duration_minutes:.2f} minutes ({duration_seconds:.1f} seconds)")
