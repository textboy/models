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

import gen_prompt
import z_image

PROJECT_DIR = Path(__file__).resolve().parent
VOCAB_DIR = PROJECT_DIR / "pool" / "IELTS" / "vocabulary"
IMG_DIR = PROJECT_DIR / "img"


def load_json(file_number):
    """Load a vocabulary JSON file by its number string (e.g., '0001')."""
    path = VOCAB_DIR / f"{file_number}.json"
    if not path.exists():
        print(f"WARNING: {path} not found, skipping.")
        return None
    with open(path) as f:
        return json.load(f)


def process_batch(start, end, device="mps"):
    """Process vocabulary files from start to end (inclusive)."""
    start_time = time.time()
    print(f"Batch started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Range: {start}.json → {end}.json")
    print(f"Device: {device}")

    # Load model once
    pipe = z_image.load_pipeline(device)

    # Create LLM client once
    client, model = gen_prompt._get_client()
    print(f"LLM Model: {model}")

    total_words = 0
    total_images = 0
    total_skipped = 0
    total_errors = 0

    for num in range(int(start), int(end) + 1):
        file_number = f"{num:04d}"
        data = load_json(file_number)
        if data is None:
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {file_number}.json  ({len(data)} words)")
        print(f"{'='*60}")

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
                    print(f"  SKIP {output_word} / {output_no_word} (already exist)")
                    total_skipped += 1
                    continue

                print(f"  [{word_text}] {meanings[:80]}...")

                try:
                    # Step 1: Generate two prompts in one API call
                    prompt_with, prompt_no, tokens = gen_prompt.generate_prompts(
                        word_text, meanings, client, model
                    )
                    print(f"    [with word]    {prompt_with[:80]}...")
                    print(f"    [no word]      {prompt_no[:80]}...")

                    # Step 2: Generate image with word
                    if not path_word.exists():
                        z_image.generate_image(
                            pipe, prompt_with, str(path_word), word_text, device,
                            with_word=True
                        )
                        print(f"    ✓ Saved: {output_word}")
                        total_images += 1
                    else:
                        print(f"    SKIP {output_word} (already exists)")

                    # Step 3: Generate image without word
                    if not path_no_word.exists():
                        z_image.generate_image(
                            pipe, prompt_no, str(path_no_word), word_text, device,
                            with_word=False
                        )
                        print(f"    ✓ Saved: {output_no_word}")
                        total_images += 1
                    else:
                        print(f"    SKIP {output_no_word} (already exists)")

                    # Brief pause to avoid rate-limiting the API
                    time.sleep(1)

                except Exception as e:
                    print(f"    ✗ ERROR: {e}")
                    total_errors += 1
                    continue

    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Batch complete!")
    print(f"  Words processed:  {total_words}")
    print(f"  Images generated: {total_images}")
    print(f"  Skipped:          {total_skipped}")
    print(f"  Errors:           {total_errors}")
    print(f"  Duration:         {elapsed/60:.1f} minutes ({elapsed:.0f}s)")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    start = sys.argv[1]
    end = sys.argv[2] if len(sys.argv) > 2 else start

    # Validate format
    if not start.isdigit() or not end.isdigit() or len(start) != 4 or len(end) != 4:
        print("ERROR: Arguments must be 4-digit file numbers (e.g., 0001, 0011)")
        sys.exit(1)

    if int(end) < int(start):
        print("ERROR: end must be >= start")
        sys.exit(1)

    process_batch(start, end)


if __name__ == "__main__":
    main()
