import torch
import time
import sys
import argparse
from datetime import datetime
from pathlib import Path
from diffusers import ZImagePipeline, ZImageTransformer2DModel, GGUFQuantizationConfig

PROJECT_DIR = Path(__file__).resolve().parent
PROMPT_INPUT = PROJECT_DIR / "gen_prompt_output.txt"
DEFAULT_GGUF = str(PROJECT_DIR / "model" / "z-image-turbo-Q4_K_S.gguf")
DEFAULT_OUTPUT = str(PROJECT_DIR / "img" / "zimage.png")


def load_pipeline(device="mps", gguf_path=None):
    """Load the Z-Image-Turbo pipeline. Call once and reuse for batch generation."""
    if gguf_path is None:
        gguf_path = DEFAULT_GGUF

    print("Loading model...")
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
    print(f"Model loaded on {device}.")
    return pipe


def generate_image(pipe, prompt_text, output_path, word="", device="mps", seed=42):
    """Generate an image from a prompt and save to output_path.

    Args:
        pipe: loaded ZImagePipeline
        prompt_text: the detailed image description (from gen_prompt)
        output_path: where to save the PNG
        word: optional word for the category prefix
        device: torch device
        seed: random seed for reproducibility
    """
    category = f"Generate an image without words {word}.\n" if word else ""
    full_prompt = category + prompt_text

    image = pipe(
        prompt=full_prompt,
        height=1024,
        width=1024,
        num_inference_steps=9,
        guidance_scale=0.0,
        generator=torch.Generator(device=device).manual_seed(seed),
    ).images[0]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
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
    print(f"Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")

    # Determine prompt source
    if args.prompt:
        prompt_text = args.prompt
        word = args.word
    elif args.prompt_file:
        prompt_path = Path(args.prompt_file)
        if not prompt_path.exists():
            print(f"ERROR: Prompt file not found: {prompt_path}")
            print("Run gen_prompt.py first to generate it.")
            sys.exit(1)
        with open(prompt_path) as f:
            word = f.readline().strip()
            prompt_text = f.read().strip()
        if not word or not prompt_text:
            print(f"ERROR: Prompt file {prompt_path} is empty or malformed.")
            sys.exit(1)
    else:
        # Default: read from shared file
        if not PROMPT_INPUT.exists():
            print(f"ERROR: Prompt file not found: {PROMPT_INPUT}")
            print("Run gen_prompt.py first to generate it.")
            sys.exit(1)
        with open(PROMPT_INPUT) as f:
            word = f.readline().strip()
            prompt_text = f.read().strip()
        if not word or not prompt_text:
            print(f"ERROR: Prompt file {PROMPT_INPUT} is empty or malformed.")
            sys.exit(1)

    print(f"Word: {word}")
    print(f"Prompt: {prompt_text[:120]}...")

    pipe = load_pipeline(args.device)
    generate_image(pipe, prompt_text, args.output, word, args.device, args.seed)
    print(f"Image saved to: {args.output}")

    end_time = time.time()
    end_datetime = datetime.now()
    duration_seconds = end_time - start_time
    duration_minutes = duration_seconds / 60.0

    print(f"End time: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration_minutes:.2f} minutes ({duration_seconds:.1f} seconds)")


if __name__ == "__main__":
    main()
