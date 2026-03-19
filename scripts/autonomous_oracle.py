"""
Autonomous Oracle — scans for unexpressed art concepts and generates novel ideas.

Uses the Art Brain RAG pipeline to explore the latent space between existing artworks,
finding creative gaps and synthesizing 10 original art concepts. Each concept includes
a title, description, influences, suggested medium, and an image generation prompt.

Output: generated/concepts.json
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from art_brain import ArtBrain

GENERATED_DIR = Path(__file__).resolve().parent.parent / "generated"
OUTPUT_FILE = GENERATED_DIR / "concepts.json"
NUM_CONCEPTS = 10

# Seed prompts that explore gaps in the art historical record — each targets
# a cross-cultural or cross-temporal fusion that is unlikely to exist already.
EXPLORATION_SEEDS = [
    "What artwork would emerge from the intersection of Japanese wabi-sabi philosophy and brutalist architecture?",
    "Imagine a textile that fuses Andean weaving traditions with op-art optical illusions.",
    "Describe a sculpture combining ancient Mesopotamian cylinder seal aesthetics with kinetic art movement.",
    "What would a painting look like that merges Mughal miniature techniques with abstract expressionist gesture?",
    "Conceive a ceramic work blending Song dynasty celadon glazing with Memphis Group postmodern design.",
    "Envision a mixed-media installation that bridges Aboriginal Australian dot painting with data visualization art.",
    "Design a metalwork piece combining Viking interlace patterns with Art Nouveau organic forms.",
    "What fresco would result from merging Byzantine icon painting with Mexican muralism?",
    "Imagine a print that synthesizes Edo-period ukiyo-e with Afrofuturist visual language.",
    "Describe a glass artwork fusing Islamic geometric patterns with Venetian glassblowing techniques.",
    "Conceive a photograph that combines Pictorialist soft-focus with Constructivist dynamic composition.",
    "What would a mask look like that merges Northwest Coast formline design with Cubist fragmentation?",
]

CONCEPT_SYSTEM_PROMPT = """You are Art Oracle. Generate ONE original art concept as valid JSON only. No other text.

{"title":"poetic title","description":"2-sentence visual description","medium":"materials and techniques","dimensions":"size","influences":["influence1","influence2","influence3"],"cultural_synthesis":"what boundaries it bridges","image_prompt":"1-sentence image generation prompt"}

Be specific and grounded in real art history. Keep all values concise."""


def _parse_json_robust(text: str, index: int) -> dict:
    """Parse JSON from LLM output, handling truncation and quirks."""
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Extract from first { to last }
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    # Handle truncated JSON: find { and try to close it
    if start >= 0:
        fragment = text[start:]
        # Try adding closing braces/brackets
        for suffix in ['"}', '"]"}', '"]}', '"}]}']:
            try:
                return json.loads(fragment + suffix)
            except json.JSONDecodeError:
                continue

        # Last resort: extract fields with regex-like approach
        import re
        fields = {}
        for key in ["title", "description", "medium", "dimensions", "cultural_synthesis", "image_prompt"]:
            m = re.search(rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"', fragment)
            if m:
                fields[key] = m.group(1)
        # Extract influences array
        m = re.search(r'"influences"\s*:\s*\[(.*?)\]', fragment, re.DOTALL)
        if m:
            fields["influences"] = [s.strip().strip('"') for s in m.group(1).split(",") if s.strip().strip('"')]

        if fields.get("title"):
            print(f"  (extracted {len(fields)} fields from partial JSON)")
            return {
                "title": fields.get("title", f"Concept {index + 1}"),
                "description": fields.get("description", ""),
                "medium": fields.get("medium", "Mixed media"),
                "dimensions": fields.get("dimensions", "Variable"),
                "influences": fields.get("influences", []),
                "cultural_synthesis": fields.get("cultural_synthesis", ""),
                "image_prompt": fields.get("image_prompt", ""),
            }

    print(f"  Warning: Could not parse JSON, wrapping raw response")
    return {
        "title": f"Concept {index + 1}",
        "description": text[:500],
        "medium": "Mixed media",
        "dimensions": "Variable",
        "influences": [],
        "cultural_synthesis": "",
        "image_prompt": "",
    }


def generate_concept(brain: ArtBrain, seed_prompt: str, index: int) -> dict:
    """Generate a single art concept using the RAG pipeline."""
    print(f"\n[{index + 1}/{NUM_CONCEPTS}] Exploring: {seed_prompt[:80]}...")

    result = brain.generate(
        prompt=seed_prompt,
        system_prompt=CONCEPT_SYSTEM_PROMPT,
        temperature=0.85,
    )

    llm_text = result["llm_response"].strip()

    # Strip markdown code fences
    if "```json" in llm_text:
        llm_text = llm_text.split("```json")[1].split("```")[0].strip()
    elif "```" in llm_text:
        llm_text = llm_text.split("```")[1].split("```")[0].strip()

    concept = _parse_json_robust(llm_text, index)

    # Attach provenance metadata
    concept["_meta"] = {
        "seed_prompt": seed_prompt,
        "context_artworks": result["context_artworks"],
        "model": result["model"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    print(f"  -> \"{concept.get('title', 'Untitled')}\"")
    return concept


def main():
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  AUTONOMOUS ORACLE — Generating Novel Art Concepts")
    print("=" * 60)

    brain = ArtBrain()
    concepts = []
    seeds = EXPLORATION_SEEDS[:NUM_CONCEPTS]

    for i, seed in enumerate(seeds):
        try:
            concept = generate_concept(brain, seed, i)
            concepts.append(concept)
        except Exception as e:
            print(f"  Error generating concept {i + 1}: {e}")
            continue

        # Brief pause between generations
        time.sleep(1)

    # Save
    OUTPUT_FILE.write_text(
        json.dumps(concepts, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print(f"  Generated {len(concepts)} concepts -> {OUTPUT_FILE}")
    print("=" * 60)

    # Print summary
    for i, c in enumerate(concepts, 1):
        print(f"\n  {i}. \"{c.get('title', 'Untitled')}\"")
        desc = c.get("description", "")
        if desc:
            print(f"     {desc[:120]}...")

    return concepts


if __name__ == "__main__":
    main()
