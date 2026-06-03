import time
import argparse
from datetime import datetime
from pathlib import Path
from llama_cpp import Llama

from log_utils import setup_logger

log = setup_logger("gen_prompt")

PROJECT_DIR = Path(__file__).resolve().parent
PROMPT_OUTPUT = PROJECT_DIR / "gen_prompt_output.txt"
DEFAULT_GGUF = str(PROJECT_DIR / "model" / "LFM2.5-8B-A1B-Q4_K_M.gguf")

SYSTEM_PROMPT = """\
Output a single text-to-image prompt that visually conveys the meaning of a given word through
a purely visual scene. The image must NOT contain any text, letters, words, or typography.
Use multiple scenes in one image if needed to illustrate abstract concepts.

IMPORTANT: Output ONLY the final prompt text. Do NOT include any reasoning, thinking,
or explanation. Just the prompt, nothing else.
"""

MAX_RETRIES = 3
RETRY_DELAY_BASE = 2  # seconds


def _get_llm(gguf_path=None):
    """Load the local GGUF model. Call once and reuse."""
    if gguf_path is None:
        gguf_path = DEFAULT_GGUF
    log.info("Loading LLM: %s", gguf_path)
    llm = Llama(
        model_path=gguf_path,
        n_ctx=4096,
        n_threads=4,
        verbose=False,
    )
    log.info("LLM loaded.")
    return llm


def _run_llm(llm, system_prompt, user_prompt):
    """Run local LLM inference with retry logic. Returns (content, tokens_used)."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = llm.create_chat_completion(
                messages=messages,
                temperature=0.7,
                max_tokens=2048,
            )

            output_text = response["choices"][0]["message"]["content"]
            tokens_used = response["usage"]["total_tokens"]

            if output_text is None:
                raise RuntimeError("LLM returned empty content")

            log.info("LLM call OK — %d tokens", tokens_used)
            log.debug("Response: %s", output_text.strip()[:200])

            # Strip <think>...</think> reasoning block (LFM 2.5 model)
            if "</think>" in output_text:
                output_text = output_text.split("</think>", 1)[1]
            else:
                # Strip <think> without closing tag — entire content is reasoning
                output_text = output_text.split("<think>", 1)[-1]

            return output_text.strip(), tokens_used

        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAY_BASE * (2 ** attempt)
                log.warning("Retry %d/%d in %ds: %s", attempt + 1, MAX_RETRIES, delay, e)
                time.sleep(delay)

    raise RuntimeError(f"Failed after {MAX_RETRIES} attempts: {last_error}")


def generate_prompt(word, meaning, llm=None):
    """Generate a single text-to-image prompt.

    Returns (prompt_text, tokens_used).
    """
    if llm is None:
        llm = _get_llm()
    log.info("Word: %s | Meaning: %s", word, meaning)
    user_prompt = f"The word is '{word}', meaning is '{meaning}'."
    prompt_text, tokens = _run_llm(llm, SYSTEM_PROMPT, user_prompt)
    log.info("Prompt: %s...", prompt_text[:120])
    return prompt_text, tokens


def main():
    parser = argparse.ArgumentParser(description="Generate a text-to-image prompt from a word")
    parser.add_argument("--word", default=None, help="The word to illustrate")
    parser.add_argument("--meaning", default=None, help="The meaning/definition of the word")
    parser.add_argument("--output", default=None, help="File to write the prompt (default: gen_prompt_output.txt)")
    parser.add_argument("--test", action="store_true", help="Run with built-in test word (exact)")
    args = parser.parse_args()

    if args.test:
        args.word = "exact"
        args.meaning = "precise and accurate in every detail; not approximate"
    elif not args.word or not args.meaning:
        parser.error("--word and --meaning are required (or use --test)")

    start_time = time.time()
    start_datetime = datetime.now()
    log.info("Start time: %s", start_datetime.strftime("%Y-%m-%d %H:%M:%S"))

    llm = _get_llm()
    log.info("Word: %s", args.word)
    log.info("Meaning: %s", args.meaning)
    log.info("Generating prompt...")

    prompt_text, tokens_used = generate_prompt(args.word, args.meaning, llm)

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
