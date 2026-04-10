from __future__ import annotations

from src import LocalEmbedder, compute_similarity


PAIRS = [
    ("The cat is sleeping on the sofa.", "A kitten is resting on the couch."),
    ("Python is a programming language.", "Python is used to build software applications."),
    ("Vietnamese mini savory pancakes use shrimp.", "Banh khot includes fresh and dried shrimp."),
    ("The stock market crashed yesterday.", "A duck porridge recipe uses grilled onion and ginger."),
    ("Orange fruit skin jam is a dessert.", "The dessert should be cooled and stored in a jar."),
]


def main() -> int:
    embedder = LocalEmbedder()

    print("Pair\tScore\tSentence A\tSentence B")
    for i, (sentence_a, sentence_b) in enumerate(PAIRS, 1):
        vec_a = embedder(sentence_a)
        vec_b = embedder(sentence_b)
        score = compute_similarity(vec_a, vec_b)
        print(f"{i}\t{score:.4f}\t{sentence_a}\t{sentence_b}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
