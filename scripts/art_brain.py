"""
Art Brain — RAG pipeline using ChromaDB + Ollama (llama3.2).

Provides the core interface for querying art history context and generating
creative responses. Used by autonomous_oracle.py to produce novel art concepts.

Usage:
    from art_brain import ArtBrain
    brain = ArtBrain()
    result = brain.generate("What would a fusion of Japanese woodblock and Art Deco look like?")
"""

import json
import sys
from pathlib import Path

import chromadb
import ollama
from sentence_transformers import SentenceTransformer

EMBEDDINGS_DIR = Path(__file__).resolve().parent.parent / "embeddings"
DB_PATH = EMBEDDINGS_DIR / "artworks_db"
COLLECTION_NAME = "artworks"
OLLAMA_MODEL = "llama3.2"
TOP_K = 8


class ArtBrain:
    """RAG pipeline: semantic art history search + local LLM generation."""

    def __init__(self, db_path=None, model_name=OLLAMA_MODEL, top_k=TOP_K):
        self.top_k = top_k
        self.model_name = model_name

        db = db_path or DB_PATH
        if not db.exists():
            print(f"Error: ChromaDB not found at {db}. Run embed_artworks.py first.")
            sys.exit(1)

        print("Loading embedding model...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        print("Connecting to ChromaDB...")
        self.client = chromadb.PersistentClient(path=str(db))
        self.collection = self.client.get_collection(COLLECTION_NAME)
        print(f"Art Brain ready — {self.collection.count()} artworks in memory, LLM: {self.model_name}")

    def search(self, query: str, n_results: int = None) -> list[dict]:
        """Semantic search over artwork embeddings. Returns list of {document, metadata, distance}."""
        n = n_results or self.top_k
        embedding = self.embedder.encode([query]).tolist()
        results = self.collection.query(query_embeddings=embedding, n_results=n)

        hits = []
        for i in range(len(results["documents"][0])):
            hits.append({
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if results.get("distances") else None,
            })
        return hits

    def build_context(self, hits: list[dict]) -> str:
        """Format search hits into a context block for the LLM."""
        lines = []
        for i, hit in enumerate(hits, 1):
            meta = hit["metadata"]
            title = meta.get("title", "Untitled")
            artist = meta.get("artist", "Unknown artist")
            date = meta.get("date", "")
            medium = meta.get("medium", "")
            culture = meta.get("culture", "")

            entry = f"{i}. \"{title}\" by {artist}"
            if date:
                entry += f" ({date})"
            if medium:
                entry += f" — {medium}"
            if culture:
                entry += f" — {culture}"
            lines.append(entry)
        return "\n".join(lines)

    def generate(self, prompt: str, system_prompt: str = None, temperature: float = 0.9) -> dict:
        """
        Full RAG pipeline: search for relevant art, build context, generate via Ollama.

        Returns:
            {
                "query": str,
                "context_artworks": list[dict],
                "llm_response": str,
                "model": str,
            }
        """
        # Retrieve
        hits = self.search(prompt)
        context = self.build_context(hits)

        # Build system prompt
        sys_prompt = system_prompt or (
            "You are Art Oracle, an AI that has deeply studied art history across cultures and centuries. "
            "You draw on your knowledge of real artworks to generate original, creative art concepts. "
            "You are imaginative, specific, and grounded in art historical knowledge. "
            "When generating concepts, describe the visual qualities, techniques, materials, "
            "cultural influences, and emotional resonance in vivid detail."
        )

        # Build user message with context
        user_message = (
            f"Here are relevant artworks from art history for context:\n\n"
            f"{context}\n\n"
            f"Based on these influences and your knowledge of art history, respond to this:\n\n"
            f"{prompt}"
        )

        # Generate
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_message},
            ],
            options={"temperature": temperature, "num_predict": 2048},
        )

        llm_text = response["message"]["content"]

        return {
            "query": prompt,
            "context_artworks": [
                {
                    "title": h["metadata"].get("title", ""),
                    "artist": h["metadata"].get("artist", ""),
                    "date": h["metadata"].get("date", ""),
                    "medium": h["metadata"].get("medium", ""),
                    "url": h["metadata"].get("objectURL", ""),
                }
                for h in hits
            ],
            "llm_response": llm_text,
            "model": self.model_name,
        }


def main():
    """Interactive demo of the Art Brain."""
    brain = ArtBrain()

    print("\n--- Art Brain Interactive Mode ---")
    print("Type a query about art, or 'quit' to exit.\n")

    # Run a demo query
    demo_queries = [
        "What would happen if Hokusai collaborated with Klimt on a mural?",
        "Describe an artwork that fuses West African textile patterns with Bauhaus geometry",
    ]

    for q in demo_queries:
        print(f"\n>> {q}")
        result = brain.generate(q)
        print(f"\n{result['llm_response']}")
        print(f"\n  [Context from {len(result['context_artworks'])} artworks, model: {result['model']}]")
        print("-" * 60)


if __name__ == "__main__":
    main()
