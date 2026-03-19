# Art Oracle

An autonomous AI artist system that learns from art history to generate novel creative concepts.

## What This Project Does

Art Oracle scrapes artwork metadata from museum APIs (starting with the Met Museum), builds a semantic understanding of art history using vector embeddings, and uses local LLMs (via Ollama) to generate original art concepts informed by centuries of human creativity.

## Architecture

- **Data Collection** (`scripts/`): Scrapers that pull artwork metadata from museum APIs
- **Embeddings** (`embeddings/`): ChromaDB vector store of artwork descriptions and metadata
- **Generation** (`generated/`): AI-generated art concepts, prompts, and outputs
- **Logs** (`logs/`): Run logs and metrics

## Key Dependencies

- `chromadb` — vector database for artwork embeddings
- `sentence-transformers` — embedding model for semantic search
- `requests` — HTTP client for museum API scraping
- `ollama` — local LLM interface for art concept generation

## Data Pipeline

1. Scrape artwork metadata from the Met Museum Open Access API
2. Embed artwork descriptions into ChromaDB
3. Query semantically similar works to build context
4. Generate novel art concepts via Ollama

## Commands

```bash
# Activate environment
source venv/bin/activate

# Scrape Met Museum artworks
python scripts/scrape_met.py

# (future) Build embeddings
python scripts/build_embeddings.py

# (future) Generate art concepts
python scripts/generate.py
```
