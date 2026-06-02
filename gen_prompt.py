import os
import time
import argparse
from datetime import datetime
from pathlib import Path
from openai import OpenAI

from log_utils import setup_logger

log = setup_logger("gen_prompt")

PROJECT_DIR = Path(__file__).resolve().parent
PROMPT_OUTPUT = PROJECT_DIR / "gen_prompt_output.txt"

SYSTEM_PROMPT = """
Generate a prompt which used in text-to-image, to create an image to well describe the word.
Because abstract word maybe not easy to describe in a single scene, you can use multiple scenes
in one image in case it needs.
"""

SYSTEM_PROMPT_DUAL = """You are a text-to-image prompt generator. Given a word and its meaning, generate TWO prompts:

1. WITH WORD — a scene where the word itself appears as visible, styled text/typography integrated into the image (e.g., on a sign, as lettering, as part of the composition)
2. NO WORD — a purely visual scene that conveys the meaning without ANY text, letters, or words appearing

Output exactly in this format:
---WITH-WORD---
<scene description with the word visible>
---NO-WORD---
<scene description without any text>
"""

MAX_RETRIES = 5
RETRY_DELAY_BASE = 2  # seconds


def _get_client():
    base_url = os.environ["NVIDIA_BASE_URL"]
    api_key = os.environ["NVIDIA_API_KEY"]
    return OpenAI(base_url=base_url, api_key=api_key), os.environ["NVIDIA_MODEL"]


def _call_api(client, model, system_prompt, user_prompt):
    """Call the LLM API with retry logic. Returns (content, tokens_used)."""
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
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

            if output_text is None:
                raise RuntimeError("API returned empty content")

            return output_text.strip(), tokens_used

        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAY_BASE * (2 ** attempt)
                log.warning("Retry %d/%d in %ds: %s", attempt + 1, MAX_RETRIES, delay, e)
                time.sleep(delay)

    raise RuntimeError(f"Failed after {MAX_RETRIES} attempts: {last_error}")


def generate_prompt(word, meaning, client=None, model=None):
    """Generate a single text-to-image prompt (backward-compatible).

    Returns (prompt_text, tokens_used).
    """
    if client is None or model is None:
        client, model = _get_client()
    user_prompt = f"The word is '{word}', meaning is '{meaning}'."
    return _call_api(client, model, SYSTEM_PROMPT, user_prompt)


def generate_prompts(word, meaning, client=None, model=None):
    """Generate two prompts for the word: one with the word visible, one without.

    Returns (prompt_with_word, prompt_no_word, tokens_used).
    """
    if client is None or model is None:
        client, model = _get_client()
    user_prompt = f"The word is '{word}', meaning is '{meaning}'."
    output_text, tokens = _call_api(client, model, SYSTEM_PROMPT_DUAL, user_prompt)

    # Parse the two prompts from the response
    try:
        parts = output_text.split("---NO-WORD---")
        with_word = parts[0].replace("---WITH-WORD---", "").strip()
        no_word = parts[1].strip() if len(parts) > 1 else ""
    except (IndexError, AttributeError):
        # Fallback: treat entire output as no-word prompt
        with_word = output_text
        no_word = output_text

    if not no_word:
        no_word = with_word

    return with_word, no_word, tokens


def main():
    parser = argparse.ArgumentParser(description="Generate a text-to-image prompt from a word")
    parser.add_argument("--word", required=True, help="The word to illustrate")
    parser.add_argument("--meaning", required=True, help="The meaning/definition of the word")
    parser.add_argument("--output", default=None, help="File to write the prompt (default: gen_prompt_output.txt)")
    args = parser.parse_args()

    start_time = time.time()
    start_datetime = datetime.now()
    log.info("Start time: %s", start_datetime.strftime("%Y-%m-%d %H:%M:%S"))

    client, model = _get_client()
    log.info("Model: %s", model)
    log.info("Word: %s", args.word)
    log.info("Meaning: %s", args.meaning)
    log.info("Generating prompt...")

    prompt_text, tokens_used = generate_prompt(args.word, args.meaning, client, model)

    separator = "-" * 50
    log.info(separator)
    log.info(prompt_text)
    log.info(separator)
    log.info("Tokens used: %s", tokens_used)

    output_path = Path(args.output) if args.output else PROMPT_OUTPUT
    with open(output_path, "w") as f:
        f.write(args.word + "\n")
        f.write(prompt_text + "\n")
    log.info("Prompt written to: %s", output_path)

    end_time = time.time()
    end_datetime = datetime.now()
    duration_seconds = end_time - start_time
    duration_minutes = duration_seconds / 60.0

    log.info("End time: %s", end_datetime.strftime("%Y-%m-%d %H:%M:%S"))
    log.info("Duration: %.2f minutes (%.1f seconds)", duration_minutes, duration_seconds)


if __name__ == "__main__":
    main()
