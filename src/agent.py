from __future__ import annotations

import os
from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str] | None = None) -> None:
        self.store = store
        self.llm_fn = llm_fn or self._openai_llm

    def _openai_llm(self, prompt: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("openai package is required. Run: pip install openai") from e

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in environment or .env file.")

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Answer the question using only "
                        "the provided context. Be concise and specific."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()

    def answer_with_details(self, question: str, top_k: int = 3) -> dict:
        """
        Retrieve top-k chunks and generate an answer.

        Returns a dict with:
            - answer: str
            - top_results: list of {rank, doc_id, chunk_index, score, content_preview}
        """
        results = self.store.search(question, top_k=top_k)

        context = "\n\n".join(
            f"[Source: {r['metadata'].get('doc_id', r['id'])}]\n{r['content']}"
            for r in results
        )
        prompt = (
            "Answer the question using the context below.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

        answer_text = self.llm_fn(prompt)

        top_results = [
            {
                "rank": i + 1,
                "doc_id": r["metadata"].get("doc_id", r["id"]),
                "chunk_index": r["metadata"].get("chunk_index", "?"),
                "score": round(r["score"], 4),
                "content_preview": r["content"][:120].replace("\n", " "),
            }
            for i, r in enumerate(results)
        ]

        return {
            "answer": answer_text,
            "top_results": top_results,
        }

    def answer(self, question: str, top_k: int = 3) -> str:
        """Return only the generated answer text for compatibility with the lab tests."""
        return self.answer_with_details(question, top_k=top_k)["answer"]
