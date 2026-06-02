import torch
import time
import sys
import argparse
from datetime import datetime
from pathlib import Path
from diffusers import ZImagePipeline, ZImageTransformer2DModel, GGUFQuantizationConfig

from log_utils import setup_logger

# Module-level logger — can be overridden by batch.py to redirect to batch log
log = setup_logger("z_image")

# ── step callback for progress logging ──────────────────────────────
_step_start = None
_num_steps = 0


def _on_step_end(pipe, step_index, timestep, callback_kwargs):
    global _step_start, _num_steps
    now = time.time()
    if step_index == 0:
        _step_start = now
    step = step_index + 1
    total = _num_steps
    pct = step * 100 // total
    elapsed = now - _step_start
    steps_per_min = step / (elapsed / 60) if elapsed > 0 else 0
    eta_total = (elapsed / step) * total if step > 0 else 0
    eta_remaining = eta_total - elapsed
    log.info(
        "  Step %d/%d (%d%%) | %.1f steps/min | elapsed %.0fs | ETA %.0fs",
        step, total, pct, steps_per_min, elapsed, eta_remaining,
    )
    return callback_kwargs

PROJECT_DIR = Path(__file__).resolve().parent
PROMPT_INPUT = PROJECT_DIR / "gen_prompt_output.txt"
DEFAULT_GGUF = str(PROJECT_DIR / "model" / "z-image-turbo-Q4_K_S.gguf")
DEFAULT_OUTPUT = str(PROJECT_DIR / "img" / "zimage.png")


def load_pipeline(device="mps", gguf_path=None):
    """Load the Z-Image-Turbo pipeline. Call once and reuse for batch generation."""
    if gguf_path is None:
        gguf_path = DEFAULT_GGUF

    log.info("Loading model...")
    transformer = ZImageTransformer2DModel.from_single_file(
        gguf_path,
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16,
    )

    pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )

    pipe = pipe.to(device)
    log.info("Model loaded on %s.", device)
    return pipe


def generate_image(pipe, prompt_text, output_path, device="mps", seed=42):
    """Generate an image from a prompt and save to output_path.

    Args:
        pipe: loaded ZImagePipeline
        prompt_text: the detailed image description (from gen_prompt)
        output_path: where to save the PNG
        device: torch device
        seed: random seed for reproducibility
    """
    category = "Generate an image without any words.\n"
    full_prompt = category + prompt_text

    log.info("Generating image (seed=%d, %s)...", seed, output_path)
    global _num_steps
    _num_steps = 9
    image = pipe(
        prompt=full_prompt,
        height=1024,
        width=1024,
        num_inference_steps=_num_steps,
        guidance_scale=0.0,
        generator=torch.Generator(device=device).manual_seed(seed),
        callback_on_step_end=_on_step_end,
    ).images[0]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    log.info("Image saved: %s", output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate an image with Z-Image-Turbo")
    parser.add_argument("--prompt-file", default=None, help="File containing word (line 1) + prompt text (rest)")
    parser.add_argument("--prompt", default=None, help="Prompt text directly (for batch usage)")
    parser.add_argument("--word", default="", help="Word for the category prefix")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output PNG path")
    parser.add_argument("--device", default="mps", help="Device: mps, cuda, or cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    start_time = time.time()
    start_datetime = datetime.now()
    log.info("Start time: %s", start_datetime.strftime("%Y-%m-%d %H:%M:%S"))

    # Determine prompt source
    if args.prompt:
        prompt_text = args.prompt
        word = args.word
    elif args.prompt_file:
        prompt_path = Path(args.prompt_file)
        if not prompt_path.exists():
            log.error("Prompt file not found: %s", prompt_path)
            log.error("Run gen_prompt.py first to generate it.")
            sys.exit(1)
        with open(prompt_path) as f:
            word = f.readline().strip()
            prompt_text = f.read().strip()
        if not word or not prompt_text:
            log.error("Prompt file %s is empty or malformed.", prompt_path)
            sys.exit(1)
    else:
        # Default: read from shared file
        if not PROMPT_INPUT.exists():
            log.error("Prompt file not found: %s", PROMPT_INPUT)
            log.error("Run gen_prompt.py first to generate it.")
            sys.exit(1)
        with open(PROMPT_INPUT) as f:
            word = f.readline().strip()
            prompt_text = f.read().strip()
        if not word or not prompt_text:
            log.error("Prompt file %s is empty or malformed.", PROMPT_INPUT)
            sys.exit(1)

    log.info("Word: %s", word)
    log.info("Prompt: %s...", prompt_text[:120])

    pipe = load_pipeline(args.device)
    generate_image(pipe, prompt_text, args.output, args.device, args.seed)
    log.info("Image saved to: %s", args.output)

    end_time = time.time()
    end_datetime = datetime.now()
    duration_seconds = end_time - start_time
    duration_minutes = duration_seconds / 60.0

    log.info("End time: %s", end_datetime.strftime("%Y-%m-%d %H:%M:%S"))
    log.info("Duration: %.2f minutes (%.1f seconds)", duration_minutes, duration_seconds)


if __name__ == "__main__":
    main()
