# EmbeddingStore Implementation Explanation

This document explains the implementation of the `EmbeddingStore` class in `src/store.py`. The store is designed to handle document storage and retrieval using both an in-memory fallback and [ChromaDB](https://www.trychroma.com/) when available.

## Key Features

### 1. Dual Mode Support
The store automatically detects if `chromadb` is installed. 
- If available, it initializes a ChromaDB client and collection.
- If not, it falls back to an in-memory list (`self._store`) for storing document records.

### 2. Document Record Creation
The `_make_record` method normalizes a `Document` object into a dictionary format suitable for in-memory storage. It generates an embedding for the document content using the provided `embedding_fn`.

### 3. Similarity Search
- **In-Memory**: Implemented manually in `_search_records` using the dot product (`_dot`) between the query embedding and stored embeddings. Results are sorted by similarity in descending order.
- **ChromaDB**: Utilizes ChromaDB's built-in `query` method, which is highly optimized for vector similarity search.

### 4. Metadata Filtering
Both storage modes support filtering by metadata:
- **In-Memory**: Iterates through records and checks if metadata keys match the provided filter.
- **ChromaDB**: Passes the `metadata_filter` directly to the `where` parameter of the `query` method.

### 5. Document Deletion
The `delete_document` method removes all chunks associated with a specific `doc_id` (stored in metadata).
- **In-Memory**: Filters out records where `metadata['doc_id'] == doc_id`.
- **ChromaDB**: Uses the `delete` method with a `where` filter.

## Summary of Implemented Methods

| Method | Description |
| :--- | :--- |
| `add_documents` | Embeds and stores a list of documents. |
| `search` | Finds the top-k most similar documents to a query. |
| `get_collection_size` | Returns the total number of stored chunks. |
| `search_with_filter` | Performs similarity search within a subset of documents matching a metadata filter. |
| `delete_document` | Removes chunks belonging to a specific document ID. |
