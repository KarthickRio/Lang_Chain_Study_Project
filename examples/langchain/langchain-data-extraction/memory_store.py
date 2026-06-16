"""
memory_store.py

Wraps ChromaDB to give the agent memory of past processed documents.

Why a vector store instead of a regular database
───────────────────────────────────────────────────
A regular database needs exact field matches to find related records.
A vector store searches by MEANING — "prescription with NPI mismatch"
will find similar documents even if the exact wording differs.
This matters because no two fax documents describe their issues
in identical words.

What gets stored
─────────────────
NOT the raw PDF text — that would make the store huge and noisy.
We store a short SUMMARY per document:
  - doc_type
  - which fields were missing
  - which fields were flagged suspicious and why
  - whether validation passed

This keeps retrieval focused on lessons learned, not document content.

Persistence
────────────
ChromaDB with persist_directory saves to disk — memory survives
between separate runs of main.py. Without this, every run would
start with no memory at all.
"""

import os
import json
import uuid
import chromadb
from chromadb.utils import embedding_functions

# ── Config ────────────────────────────────────────────────────────────────────

PERSIST_DIR     = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "fax_processing_memory"

# Using ChromaDB's default embedding function — runs locally,
# no API key needed, good enough for this POC's similarity search.
_embedding_fn = embedding_functions.DefaultEmbeddingFunction()


def _get_client():
    """Returns a persistent ChromaDB client — same data across runs."""
    return chromadb.PersistentClient(path=PERSIST_DIR)


def _get_collection():
    """
    Gets or creates the collection that stores document summaries.
    A collection in ChromaDB is like a table — one logical group of records.
    """
    client = _get_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_embedding_fn,
    )


def _build_summary_text(doc_type: str, missing_fields: list,
                         suspicious_fields: dict, validation_passed) -> str:
    """
    Build the text that gets embedded and searched.
    Written in plain language so semantic search works well —
    not just a dump of field names.
    """
    parts = [f"Document type: {doc_type}."]

    if missing_fields:
        parts.append(f"Missing fields: {', '.join(missing_fields)}.")

    if suspicious_fields:
        issues = "; ".join(f"{k} — {v}" for k, v in suspicious_fields.items())
        parts.append(f"Suspicious fields flagged: {issues}.")

    if validation_passed is False:
        parts.append("Validation did not pass — needed human review.")
    elif validation_passed is True:
        parts.append("Validation passed cleanly.")

    return " ".join(parts)


def save_document_memory(
    doc_type: str,
    missing_fields: list,
    suspicious_fields: dict,
    validation_passed,
    final_fields: dict,
) -> str:
    """
    Save a processed document's summary into memory.
    Call this after synthesis completes.

    Returns the unique ID assigned to this memory record.
    """
    collection = _get_collection()

    summary_text = _build_summary_text(
        doc_type, missing_fields, suspicious_fields, validation_passed
    )

    record_id = str(uuid.uuid4())

    # Metadata stored alongside the embedding — used to filter
    # and to reconstruct useful info when retrieved later
    metadata = {
        "doc_type":          str(doc_type),
        "missing_count":     len(missing_fields),
        "suspicious_count":  len(suspicious_fields),
        "validation_passed": str(validation_passed),
        # Store fields as JSON string — ChromaDB metadata must be
        # simple types (str/int/float/bool), not nested dicts
        "missing_fields_json":    json.dumps(missing_fields),
        "suspicious_fields_json": json.dumps(suspicious_fields),
    }

    collection.add(
        ids=[record_id],
        documents=[summary_text],
        metadatas=[metadata],
    )

    print(f"  💾 Saved to memory: {summary_text[:100]}...")
    return record_id


def search_similar_documents(doc_type: str, missing_fields: list, n_results: int = 2) -> list:
    """
    Search memory for past documents similar to the current one.
    Call this before AI extraction runs.

    Returns a list of dicts with summary text and metadata,
    most similar first. Empty list if no memory exists yet.
    """
    collection = _get_collection()

    # Build a query that describes the CURRENT situation
    # so ChromaDB finds past documents in a similar state
    query_text = _build_summary_text(doc_type, missing_fields, {}, None)

    try:
        # Skip search if collection is empty — avoids ChromaDB error
        if collection.count() == 0:
            print(f"  🔍 Memory search skipped — no documents stored yet")
            return []

        results = collection.query(
            query_texts=[query_text],
            n_results=min(n_results, collection.count()),
        )

        matches = []
        documents  = results.get("documents",  [[]])[0]
        metadatas  = results.get("metadatas",  [[]])[0]
        distances  = results.get("distances",  [[]])[0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            matches.append({
                "summary":           doc,
                "doc_type":          meta.get("doc_type", ""),
                "suspicious_fields": json.loads(meta.get("suspicious_fields_json", "{}")),
                "distance":          dist,   # lower = more similar
            })

        print(f"  🔍 Memory search found {len(matches)} similar past document(s)")
        for m in matches:
            print(f"     • [{m['doc_type']}] {m['summary'][:80]}... (distance: {m['distance']:.3f})")

        return matches

    except Exception as e:
        print(f"  ⚠️  Memory search failed: {e}")
        return []