import torch
import time
from datetime import datetime
from diffusers import ZImagePipeline, ZImageTransformer2DModel, GGUFQuantizationConfig

local_gguf_path = "/users/mk/workspace/models/model/z-image-turbo-Q4_K_S.gguf"
image_output_path = "/users/mk/workspace/models/output/zimage.png"

# Record start time
start_time = time.time()
start_datetime = datetime.now()
print(f"Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")

# Load the quantized transformer model from your local GGUF file
print("Loading model...")
transformer = ZImageTransformer2DModel.from_single_file(
    local_gguf_path,
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16,
)
print("Model loaded.")

# 4. Create the pipeline, passing the loaded transformer
#    The original model ID is still needed for the pipeline's configuration
pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)

# 5. Move the pipeline to your device
#    For Mac, use "mps". For NVIDIA GPU, use "cuda".
#    If you don't have a compatible GPU, use "cpu" (will be very slow).
device = "mps"  # For Apple Silicon Macs
# device = "cuda" # For NVIDIA GPUs
pipe = pipe.to(device)

# --- Optional settings (uncomment if needed) ---
# pipe.enable_model_cpu_offload() # Use if you run out of memory

# --- Your Prompt ---
prompt = """
Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, 
red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, 
beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), 
bright yellow glow, above extended left palm. Soft-lit outdoor night background, 
silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights.
"""

# --- Generate the image ---
# The settings like num_inference_steps and guidance_scale are specific to Turbo models
print("Generating image...")
image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    num_inference_steps=9,  # This is correct for the Turbo model
    guidance_scale=0.0,     # Must be 0 for Turbo models
    generator=torch.Generator(device=device).manual_seed(42),
).images[0]

# --- Save the image ---
image.save(image_output_path)
print("Image saved as zimage.png")

# Record end time and print timing summary
end_time = time.time()
end_datetime = datetime.now()
duration_seconds = end_time - start_time
duration_minutes = duration_seconds / 60.0

print(f"End time: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Duration: {duration_minutes:.2f} minutes ({duration_seconds:.1f} seconds)")