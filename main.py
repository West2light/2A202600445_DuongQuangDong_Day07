from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.agent import KnowledgeBaseAgent
from src.chunking import FixedSizeChunker, RecursiveChunker, SentenceChunker
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore

# Force UTF-8 output on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Config ────────────────────────────────────────────────────────────────────

SAMPLE_FILES = [
    "data/Braised_Tofu.md",
    "data/Duck_Porridge.md",
    "data/Savory_Pancakes.md",
    "data/Grilled_Snails.md",
    "data/Orange_Fruit_Skin_Jam.md",
]

FILE_METADATA = {
    "Braised_Tofu":        {"category": "main_dish", "difficulty": "easy",   "source": "vietnamtourism"},
    "Duck_Porridge":       {"category": "main_dish", "difficulty": "medium", "source": "vietnamtourism"},
    "Savory_Pancakes":     {"category": "main_dish", "difficulty": "hard",   "source": "vietnamtourism"},
    "Grilled_Snails":      {"category": "seafood",   "difficulty": "easy",   "source": "vietnamtourism"},
    "Orange_Fruit_Skin_Jam": {"category": "dessert", "difficulty": "easy",   "source": "vietnamtourism"},
}

BENCHMARK_FILE = "benchmark_queries.json"

CHUNKERS = {
    "fixed":     FixedSizeChunker(chunk_size=300, overlap=50),
    "sentence":  SentenceChunker(max_sentences_per_chunk=3),
    "recursive": RecursiveChunker(chunk_size=300),
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_benchmark(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        fallback = Path("data") / path
        if fallback.exists():
            p = fallback
    if not p.exists():
        print(f"Benchmark file not found: {path}")
        sys.exit(1)
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_raw_files(file_paths: list[str]) -> list[tuple[str, str, dict]]:
    raw: list[tuple[str, str, dict]] = []
    for raw_path in file_paths:
        path = Path(raw_path)
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        if not path.exists():
            print(f"Missing: {path}")
            continue
        content = path.read_text(encoding="utf-8")
        meta = {"source": path.stem, "extension": path.suffix.lower()}
        meta.update(FILE_METADATA.get(path.stem, {}))
        raw.append((path.stem, content, meta))
    return raw


def make_chunked_documents(raw_files: list[tuple[str, str, dict]], chunker) -> list[Document]:
    docs: list[Document] = []
    for doc_id, content, meta in raw_files:
        for i, chunk in enumerate(chunker.chunk(content)):
            docs.append(Document(
                id=f"{doc_id}:{i}",
                content=chunk,
                metadata={**meta, "doc_id": doc_id, "chunk_index": i},
            ))
    return docs


def pick_embedder(provider: str):
    if provider == "local":
        try:
            return LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception as e:
            print(f"  LocalEmbedder failed ({e}), falling back to mock.")
    elif provider == "openai":
        try:
            return OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception as e:
            print(f"  OpenAIEmbedder failed ({e}), falling back to mock.")
    return _mock_embed


def demo_llm(prompt: str) -> str:
    """Extract context lines from prompt and return a readable mock answer."""
    context_section = prompt.split("Context:")[1].split("Question:")[0].strip() if "Context:" in prompt else ""
    lines = [ln.strip() for ln in context_section.splitlines() if ln.strip()]
    return " ".join(lines)[:400] + ("..." if len(" ".join(lines)) > 400 else "")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    load_dotenv(override=False)

    # Pick strategy from CLI: python main.py [strategy] ["optional query override"]
    args = sys.argv[1:]
    chosen_strategy: str | None = None
    if args and args[0] in CHUNKERS:
        chosen_strategy = args.pop(0)

    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    embedder = pick_embedder(provider)
    backend_name = getattr(embedder, "_backend_name", embedder.__class__.__name__)

    raw_files = load_raw_files(SAMPLE_FILES)
    if not raw_files:
        print("No valid documents loaded.")
        return 1

    benchmarks = load_benchmark(BENCHMARK_FILE)

    strategies_to_run = {chosen_strategy: CHUNKERS[chosen_strategy]} if chosen_strategy else CHUNKERS

    for strategy_name, chunker in strategies_to_run.items():
        docs = make_chunked_documents(raw_files, chunker)
        store = EmbeddingStore(collection_name=f"store_{strategy_name}", embedding_fn=embedder)
        store.add_documents(docs)
        agent = KnowledgeBaseAgent(store=store, llm_fn=demo_llm)

        print()
        print("=" * 70)
        print(f"  Strategy : {strategy_name.upper()}  ({len(docs)} chunks)")
        print(f"  Embedder : {backend_name}")
        print("=" * 70)

        for item in benchmarks:
            query       = item["query"]
            gold_answer = item["gold_answer"]
            result      = agent.answer_with_details(query, top_k=3)

            print(f"\nQ{item['id']}: {query}")
            print(f"  Gold   : {gold_answer}")
            for r in result["top_results"]:
                print(f"  #{r['rank']} [{r['doc_id']} chunk#{r['chunk_index']}] score={r['score']:.4f}  {r['content_preview']}...")
            print(f"  Chatbot: {result['answer']}")
            print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
