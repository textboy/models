import os
import time
import argparse
from datetime import datetime
from pathlib import Path
from openai import OpenAI

PROJECT_DIR = Path(__file__).resolve().parent
PROMPT_OUTPUT = PROJECT_DIR / "gen_prompt_output.txt"

SYSTEM_PROMPT = """
Generate a prompt which used in text-to-image, to create an image to well describe the word.
Because abstract word maybe not easy to describe in a single scene, you can use multiple scenes
in one image in case it needs.
"""


def _get_client():
    base_url = os.environ["NVIDIA_BASE_URL"]
    api_key = os.environ["NVIDIA_API_KEY"]
    return OpenAI(base_url=base_url, api_key=api_key), os.environ["NVIDIA_MODEL"]


def generate_prompt(word, meaning, client=None, model=None):
    """Generate a text-to-image prompt for the given word and meaning.

    Returns the generated prompt string.
    """
    if client is None or model is None:
        client, model = _get_client()

    user_prompt = f"The word is '{word}', meaning is '{meaning}'."

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=512,
    )

    output_text = response.choices[0].message.content
    tokens_used = response.usage.total_tokens

    return output_text.strip(), tokens_used


def main():
    parser = argparse.ArgumentParser(description="Generate a text-to-image prompt from a word")
    parser.add_argument("--word", required=True, help="The word to illustrate")
    parser.add_argument("--meaning", required=True, help="The meaning/definition of the word")
    parser.add_argument("--output", default=None, help="File to write the prompt (default: gen_prompt_output.txt)")
    args = parser.parse_args()

    start_time = time.time()
    start_datetime = datetime.now()
    print(f"Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")

    client, model = _get_client()
    print(f"Model: {model}")
    print(f"Word: {args.word}")
    print(f"Meaning: {args.meaning}")
    print("Generating prompt...")

    prompt_text, tokens_used = generate_prompt(args.word, args.meaning, client, model)

    print("-" * 50)
    print(prompt_text)
    print("-" * 50)
    print(f"Tokens used: {tokens_used}")

    output_path = Path(args.output) if args.output else PROMPT_OUTPUT
    with open(output_path, "w") as f:
        f.write(args.word + "\n")
        f.write(prompt_text + "\n")
    print(f"Prompt written to: {output_path}")

    end_time = time.time()
    end_datetime = datetime.now()
    duration_seconds = end_time - start_time
    duration_minutes = duration_seconds / 60.0

    print(f"End time: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration_minutes:.2f} minutes ({duration_seconds:.1f} seconds)")


if __name__ == "__main__":
    main()
