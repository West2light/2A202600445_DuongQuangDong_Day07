from __future__ import annotations

import sys
from pathlib import Path

from src.chunking import ChunkingStrategyComparator

DEFAULT_FILES = [
    "./data\\Braised_Tofu.md",
    "./data\\Duck_Porridge.md",
    "./data\\Orange_Fruit_Skin_Jam.md",
]


def load_text(path_str: str) -> tuple[Path, str] | None:
    path = Path(path_str)
    if not path.exists() or not path.is_file():
        print(f"Skipping missing file: {path}")
        return None

    if path.suffix.lower() not in {".txt", ".md"}:
        print(f"Skipping unsupported file type: {path}")
        return None

    return path, path.read_text(encoding="utf-8")


def print_comparison(path: Path, comparison: dict) -> None:
    print(f"\n=== {path.name} ===")
    for strategy_name, stats in comparison.items():
        print(f"- {strategy_name}")
        print(f"  count: {stats['count']}")
        print(f"  avg_length: {stats['avg_length']:.2f}")
        preview = stats["chunks"][:2]
        for index, chunk in enumerate(preview, start=1):
            compact = chunk.replace("\n", " ").strip()
            print(f"  chunk_{index}: {compact}")


def main() -> int:
    file_paths = sys.argv[1:] or DEFAULT_FILES
    comparator = ChunkingStrategyComparator()

    loaded = [item for item in (load_text(path) for path in file_paths) if item is not None]
    if not loaded:
        print("No valid .txt or .md files found.")
        return 1

    for path, text in loaded:
        comparison = comparator.compare(text, chunk_size=500, overlap=50)
        print_comparison(path, comparison)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
