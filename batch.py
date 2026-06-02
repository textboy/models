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

    # Create LLM client once
    client, model = gen_prompt._get_client()
    log.info("LLM Model: %s", model)

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
                output_word = f"{word_id}-{definition_id}-word.png"
                output_no_word = f"{word_id}-{definition_id}-no-word.png"
                path_word = IMG_DIR / output_word
                path_no_word = IMG_DIR / output_no_word

                total_words += 1

                # Skip if both already generated
                if path_word.exists() and path_no_word.exists():
                    log.info("  SKIP %s / %s (already exist)", output_word, output_no_word)
                    total_skipped += 1
                    continue

                log.info("  [%s] %s...", word_text, meanings[:80])

                try:
                    # Step 1: Generate two prompts in one API call
                    prompt_with, prompt_no, tokens = gen_prompt.generate_prompts(
                        word_text, meanings, client, model
                    )
                    log.info("    [with word]    %s...", prompt_with[:80])
                    log.info("    [no word]      %s...", prompt_no[:80])

                    # Step 2: Generate image with word
                    if not path_word.exists():
                        z_image.generate_image(
                            pipe, prompt_with, str(path_word), word_text, device,
                            with_word=True
                        )
                        log.info("    ✓ Saved: %s", output_word)
                        total_images += 1
                    else:
                        log.info("    SKIP %s (already exists)", output_word)

                    # Step 3: Generate image without word
                    if not path_no_word.exists():
                        z_image.generate_image(
                            pipe, prompt_no, str(path_no_word), word_text, device,
                            with_word=False
                        )
                        log.info("    ✓ Saved: %s", output_no_word)
                        total_images += 1
                    else:
                        log.info("    SKIP %s (already exists)", output_no_word)

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
