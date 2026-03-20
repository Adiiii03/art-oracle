"""
Generate artwork images from concepts using SDXL-Turbo via HuggingFace diffusers.

Loads concepts from generated/concepts.json, builds optimized prompts,
and generates images on GPU using SDXL-Turbo (optimized for RTX 3060 12GB).

Usage:
    python scripts/generate_art.py                  # Generate all concepts
    python scripts/generate_art.py --index 9        # Generate concept #9 only
    python scripts/generate_art.py --index 1 3 7    # Generate specific concepts
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from diffusers import AutoPipelineForText2Image

GENERATED_DIR = Path(__file__).resolve().parent.parent / "generated"
CONCEPTS_FILE = GENERATED_DIR / "concepts.json"
IMAGES_DIR = GENERATED_DIR / "images"

MODEL_ID = "stabilityai/sdxl-turbo"


def slugify(text: str) -> str:
    """Convert text to a safe filename."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "_", text)
    return text[:80].strip("_")


def build_prompt(concept: dict) -> str:
    """Convert a concept into an optimized image generation prompt."""
    parts = []

    # Use the image_prompt field if it exists and is substantial
    image_prompt = concept.get("image_prompt", "").strip()
    if image_prompt and len(image_prompt) > 20:
        parts.append(image_prompt)
    else:
        # Fall back to building from description + metadata
        desc = concept.get("description", "").strip()
        if desc:
            # If description contains raw JSON, try to extract the actual description
            if desc.startswith("{"):
                try:
                    inner = json.loads(desc)
                    desc = inner.get("description", desc)
                except json.JSONDecodeError:
                    pass
            parts.append(desc)

    # Add medium for style guidance
    medium = concept.get("medium", "").strip()
    if medium and medium != "Mixed media":
        parts.append(medium)

    # Add cultural synthesis for thematic grounding
    synthesis = concept.get("cultural_synthesis", "").strip()
    if synthesis:
        parts.append(synthesis)

    prompt = ". ".join(parts) if parts else concept.get("title", "abstract artwork")

    # Add quality boosters for SDXL
    prompt += ". masterpiece, highly detailed, professional artwork, museum quality"

    # Truncate to stay within CLIP token limits (~77 tokens, ~300 chars is safe)
    if len(prompt) > 350:
        prompt = prompt[:347] + "..."

    return prompt


def load_pipeline():
    """Load SDXL-Turbo pipeline optimized for RTX 3060."""
    print(f"Loading {MODEL_ID}...")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    pipe = AutoPipelineForText2Image.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe = pipe.to("cuda")

    # Memory optimizations for 12GB VRAM
    pipe.enable_attention_slicing()

    print("Pipeline ready.\n")
    return pipe


def generate_image(pipe, prompt: str, seed: int = None):
    """Generate a single image from a prompt."""
    generator = torch.Generator(device="cuda")
    if seed is not None:
        generator.manual_seed(seed)
    else:
        generator.manual_seed(torch.randint(0, 2**32 - 1, (1,)).item())

    # SDXL-Turbo: designed for 1-4 steps, no CFG needed
    image = pipe(
        prompt=prompt,
        num_inference_steps=4,
        guidance_scale=0.0,
        width=512,
        height=512,
        generator=generator,
    ).images[0]

    return image


def main():
    parser = argparse.ArgumentParser(description="Generate art from concepts")
    parser.add_argument(
        "--index", type=int, nargs="+",
        help="1-based index(es) of concepts to generate (default: all)",
    )
    args = parser.parse_args()

    if not CONCEPTS_FILE.exists():
        print(f"Error: {CONCEPTS_FILE} not found. Run autonomous_oracle.py first.")
        sys.exit(1)

    concepts = json.loads(CONCEPTS_FILE.read_text(encoding="utf-8"))
    print(f"Loaded {len(concepts)} concepts")

    # Filter to requested indices
    if args.index:
        indices = [i - 1 for i in args.index if 1 <= i <= len(concepts)]
        if not indices:
            print(f"Error: invalid indices. Must be 1-{len(concepts)}.")
            sys.exit(1)
    else:
        indices = list(range(len(concepts)))

    if not torch.cuda.is_available():
        print("Error: CUDA not available. Need GPU for image generation.")
        sys.exit(1)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    pipe = load_pipeline()

    for idx in indices:
        concept = concepts[idx]
        title = concept.get("title", f"concept_{idx + 1}")
        prompt = build_prompt(concept)
        filename = slugify(title) + ".png"
        output_path = IMAGES_DIR / filename

        print(f"[{idx + 1}/{len(concepts)}] Generating: {title}")
        print(f"  Prompt: {prompt[:120]}...")

        image = generate_image(pipe, prompt, seed=42 + idx)
        image.save(output_path)
        print(f"  Saved: {output_path}\n")

    print(f"Done! Generated {len(indices)} image(s) in {IMAGES_DIR}")


if __name__ == "__main__":
    main()
