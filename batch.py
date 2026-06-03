#!/usr/bin/env python3
"""Batch generate vocabulary images from IELTS JSON files.

Usage:
    python batch.py 0001 0011     # process 0001.json through 0011.json
    python batch.py 0001          # process 0001.json only
"""

import sys
import json
import time
from datetime import datetime
from pathlib import Path

from log_utils import setup_logger

log = setup_logger("batch")
import gen_prompt
import z_image

PROJECT_DIR = Path(__file__).resolve().parent
VOCAB_DIR = PROJECT_DIR / "pool" / "IELTS" / "vocabulary"
IMG_DIR = PROJECT_DIR / "img"


def load_json(file_number):
    """Load a vocabulary JSON file by its number string (e.g., '0001')."""
    path = VOCAB_DIR / f"{file_number}.json"
    if not path.exists():
        log.warning("%s not found, skipping.", path)
        return None
    with open(path) as f:
        return json.load(f)


def process_batch(start, end, device="mps"):
    """Process vocabulary files from start to end (inclusive)."""
    start_time = time.time()
    log.info("Batch started at: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info("Range: %s.json → %s.json", start, end)
    log.info("Device: %s", device)

    # Load model once
    pipe = z_image.load_pipeline(device)

    # Load local LLM once
    llm = gen_prompt._get_llm()
    log.info("LLM: %s", gen_prompt.DEFAULT_GGUF)

    total_words = 0
    total_images = 0
    total_skipped = 0
    total_errors = 0

    for num in range(int(start), int(end) + 1):
        file_number = f"{num:04d}"
        data = load_json(file_number)
        if data is None:
            continue

        log.info("")
        log.info("=" * 60)
        log.info("Processing: %s.json  (%d words)", file_number, len(data))
        log.info("=" * 60)

        for word_entry in data:
            word_id = word_entry["id"]
            word_text = word_entry["text"]

            for definition in word_entry["definitions"]:
                definition_id = definition["definitionId"]
                meanings = definition["meanings"]
                output_01 = f"{word_id}-{definition_id}-01.png"
                output_02 = f"{word_id}-{definition_id}-02.png"
                path_01 = IMG_DIR / output_01
                path_02 = IMG_DIR / output_02

                total_words += 1

                # Skip if both already generated
                if path_01.exists() and path_02.exists():
                    log.info("  SKIP %s / %s (already exist)", output_01, output_02)
                    total_skipped += 1
                    continue

                log.info("  [%s] %s...", word_text, meanings[:80])

                try:
                    # Step 1: Generate prompt via LLM (with retry built in)
                    prompt_text, tokens = gen_prompt.generate_prompt(
                        word_text, meanings, llm
                    )
                    log.info("    Prompt: %s...", prompt_text[:80])

                    # Step 2: Generate two images from the same prompt, different seeds
                    if not path_01.exists():
                        z_image.generate_image(
                            pipe, prompt_text, str(path_01), device, seed=42
                        )
                        log.info("    ✓ Saved: %s", output_01)
                        total_images += 1
                    else:
                        log.info("    SKIP %s (already exists)", output_01)

                    if not path_02.exists():
                        z_image.generate_image(
                            pipe, prompt_text, str(path_02), device, seed=43
                        )
                        log.info("    ✓ Saved: %s", output_02)
                        total_images += 1
                    else:
                        log.info("    SKIP %s (already exists)", output_02)

                    # Brief pause to avoid rate-limiting the API
                    time.sleep(1)

                except Exception as e:
                    log.error("    ✗ ERROR: %s", e)
                    total_errors += 1
                    continue

    # Summary
    elapsed = time.time() - start_time
    log.info("")
    log.info("=" * 60)
    log.info("Batch complete!")
    log.info("  Words processed:  %d", total_words)
    log.info("  Images generated: %d", total_images)
    log.info("  Skipped:          %d", total_skipped)
    log.info("  Errors:           %d", total_errors)
    log.info("  Duration:         %.1f minutes (%.0fs)", elapsed / 60, elapsed)
    log.info("Finished at: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    start = sys.argv[1]
    end = sys.argv[2] if len(sys.argv) > 2 else start

    # Validate format
    if not start.isdigit() or not end.isdigit() or len(start) != 4 or len(end) != 4:
        log.error("Arguments must be 4-digit file numbers (e.g., 0001, 0011)")
        sys.exit(1)

    if int(end) < int(start):
        log.error("end must be >= start")
        sys.exit(1)

    process_batch(start, end)


if __name__ == "__main__":
    main()
