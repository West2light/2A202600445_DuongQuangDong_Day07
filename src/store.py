from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb

            self._client = chromadb.Client()
            self._collection = self._client.get_or_create_collection(name=self._collection_name)
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        """Build a normalized stored record for one document."""
        embedding = self._embedding_fn(doc.content)
        metadata = dict(doc.metadata)
        metadata.setdefault("doc_id", doc.id)
        return {
            "id": doc.id,
            "content": doc.content,
            "metadata": metadata,
            "embedding": embedding,
        }

    def _format_search_result(self, record: dict[str, Any], score: float) -> dict[str, Any]:
        """Strip internal fields and expose a stable search result schema."""
        return {
            "id": record["id"],
            "content": record["content"],
            "metadata": record["metadata"],
            "score": score,
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        """Run in-memory similarity search over provided records."""
        if not records:
            return []

        query_embedding = self._embedding_fn(query)
        scored_records = []
        for record in records:
            similarity = _dot(query_embedding, record["embedding"])
            scored_records.append({"record": record, "similarity": similarity})

        scored_records.sort(key=lambda x: x["similarity"], reverse=True)
        return [
            self._format_search_result(item["record"], item["similarity"])
            for item in scored_records[:top_k]
        ]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.
        """
        if self._use_chroma and self._collection:
            ids = [doc.id for doc in docs]
            metadatas = []
            for doc in docs:
                metadata = dict(doc.metadata)
                metadata.setdefault("doc_id", doc.id)
                metadatas.append(metadata)
            documents = [doc.content for doc in docs]
            embeddings = [self._embedding_fn(doc.content) for doc in docs]
            self._collection.add(
                ids=ids,
                metadatas=metadatas,
                documents=documents,
                embeddings=embeddings
            )
        else:
            for doc in docs:
                self._store.append(self._make_record(doc))

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Find the top_k most similar documents to query."""
        if self._use_chroma and self._collection:
            results = self._collection.query(
                query_embeddings=[self._embedding_fn(query)],
                n_results=top_k
            )
            formatted = []
            if results["ids"]:
                distances = results.get("distances", [[]])
                for i in range(len(results["ids"][0])):
                    formatted.append({
                        "id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "score": -distances[0][i] if distances and distances[0] else 0.0,
                    })
            return formatted
        else:
            return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma and self._collection:
            return self._collection.count()
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """Search with optional metadata pre-filtering."""
        if self._use_chroma and self._collection:
            results = self._collection.query(
                query_embeddings=[self._embedding_fn(query)],
                n_results=top_k,
                where=metadata_filter
            )
            formatted = []
            if results["ids"]:
                distances = results.get("distances", [[]])
                for i in range(len(results["ids"][0])):
                    formatted.append({
                        "id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "score": -distances[0][i] if distances and distances[0] else 0.0,
                    })
            return formatted
        else:
            filtered_records = []
            for record in self._store:
                match = True
                if metadata_filter:
                    for key, value in metadata_filter.items():
                        if record["metadata"].get(key) != value:
                            match = False
                            break
                if match:
                    filtered_records.append(record)
            return self._search_records(query, filtered_records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """Remove all chunks belonging to a document."""
        if self._use_chroma and self._collection:
            count_before = self._collection.count()
            self._collection.delete(ids=[doc_id])
            count_after = self._collection.count()
            if count_after < count_before:
                return True

            self._collection.delete(where={"doc_id": doc_id})
            return self._collection.count() < count_after
        else:
            initial_len = len(self._store)
            self._store = [
                record for record in self._store
                if record["id"] != doc_id and record["metadata"].get("doc_id") != doc_id
            ]
            return len(self._store) < initial_len
