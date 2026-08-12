"""
Embeddings (Gemini text-embedding-004, routed through GeminiKeyManager) and
Pinecone vector storage / similarity search.
"""
import logging
import uuid
from typing import Dict, List, Optional, Union

from google.genai import types
from pinecone import Pinecone, ServerlessSpec

from .. import config
from ..gemini_manager import key_manager

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Pinecone setup — deferred until first use (not at import time), so the
# merged app can start up without PINECONE_API_KEY set when only the
# ask-scholar routes are being used. The index is created on first call if
# it doesn't already exist.
# --------------------------------------------------------------------------
_pc = None
_index = None


def _get_index():
    global _pc, _index
    if _index is not None:
        return _index

    if not config.PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY is not set — RAG search/ingestion is not configured.")

    logger.info("Initializing Pinecone client (index=%s)", config.PINECONE_INDEX_NAME)
    _pc = Pinecone(api_key=config.PINECONE_API_KEY)
    if config.PINECONE_INDEX_NAME not in [i["name"] for i in _pc.list_indexes()]:
        logger.info(
            "Pinecone index %s not found, creating it (dim=%d, cloud=%s, region=%s)",
            config.PINECONE_INDEX_NAME, config.EMBEDDING_DIMENSION, config.PINECONE_CLOUD, config.PINECONE_REGION,
        )
        _pc.create_index(
            name=config.PINECONE_INDEX_NAME,
            dimension=config.EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
        )
    _index = _pc.Index(config.PINECONE_INDEX_NAME)
    return _index


def embed_texts(texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
    """
    Embeds each text using the Gemini embedding model, routed through
    GeminiKeyManager (task="embedding") so key rotation / model fallback
    apply here too.

    One API call per text, not one batched call for the whole list: the
    old model (text-embedding-004) batched fine, but its replacement
    (gemini-embedding-001) doesn't — passing a list of strings to
    embed_content() silently collapses them into a single merged
    embedding instead of one per string, which would corrupt every chunk
    stored after the first. Looping costs more requests but is correct.
    """
    if not texts:
        return []

    logger.info("Embedding %d text(s) (task_type=%s)", len(texts), task_type)
    embeddings = []
    for i, text in enumerate(texts):
        def fn(client, model, _text=text):
            result = client.models.embed_content(
                model=model,
                contents=_text,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=config.EMBEDDING_DIMENSION,
                ),
            )
            return result.embeddings[0].values

        try:
            embeddings.append(key_manager.call("embedding", fn))
        except Exception:
            logger.exception("Embedding failed for text %d/%d", i + 1, len(texts))
            raise
    logger.info("Embedded %d text(s) successfully", len(embeddings))
    return embeddings


def embed_query(query: str) -> List[float]:
    logger.info("Embedding query (len=%d chars)", len(query))
    return embed_texts([query], task_type="RETRIEVAL_QUERY")[0]


def store_chunks(
    chunks: List[Dict],
    source_id: str,
    source_type: str,
    source_url: str,
    title: Optional[str] = None,
    namespace: Optional[str] = None,
    batch_size: int = 50,
) -> int:
    """
    chunks: list of dicts each containing at least "text", plus optional
    "start_time" / "end_time" (YouTube) or "chunk_index" (documents).

    Embeds every chunk and upserts into Pinecone with metadata. Returns the
    number of chunks stored.
    """
    if not chunks:
        logger.info("store_chunks called with no chunks for source_id=%s, skipping", source_id)
        return 0

    logger.info(
        "Storing %d chunk(s) for source_id=%s source_type=%s namespace=%s",
        len(chunks), source_id, source_type, namespace or "(default)",
    )

    stored = 0
    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start : batch_start + batch_size]
        texts = [c["text"] for c in batch]
        logger.info(
            "Processing batch %d-%d of %d for source_id=%s",
            batch_start, batch_start + len(batch) - 1, len(chunks), source_id,
        )
        vectors = embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")

        upserts = []
        for i, (chunk, vector) in enumerate(zip(batch, vectors)):
            logger.debug(
                "Chunk %d text preview: %r",
                batch_start + i, chunk["text"][:120],
            )
            metadata = {
                "text": chunk["text"],
                "source_id": source_id,
                "source_type": source_type,
                "source_url": source_url,
                "title": title or "",
                "chunk_index": batch_start + i,
            }
            if chunk.get("start_time") is not None:
                metadata["start_time"] = chunk["start_time"]
            if chunk.get("end_time") is not None:
                metadata["end_time"] = chunk["end_time"]
            if chunk.get("page_number") is not None:
                metadata["page_number"] = chunk["page_number"]

            upserts.append(
                {
                    "id": f"{source_id}-{batch_start + i}-{uuid.uuid4().hex[:8]}",
                    "values": vector,
                    "metadata": metadata,
                }
            )

        logger.info("Upserting %d vector(s) to Pinecone namespace=%s", len(upserts), namespace or "(default)")
        try:
            _get_index().upsert(vectors=upserts, namespace=namespace or "")
        except Exception:
            logger.exception(
                "Pinecone upsert failed for source_id=%s batch %d-%d",
                source_id, batch_start, batch_start + len(batch) - 1,
            )
            raise
        stored += len(upserts)

    logger.info("Finished storing %d/%d chunk(s) for source_id=%s", stored, len(chunks), source_id)
    return stored


def query_similar(
    query: str,
    top_k: int = 5,
    namespace: Optional[str] = None,
    source_type: Optional[Union[str, List[str]]] = None,
) -> List[Dict]:
    """
    source_type accepts a single value ("youtube" / "document"), a list of
    values (e.g. ["youtube", "document"] to search both in one call, mixed
    together and ranked by score), or None for no filter at all.
    """
    logger.info(
        "Querying similar chunks: top_k=%d namespace=%s source_type=%s",
        top_k, namespace or "(default)", source_type,
    )
    vector = embed_query(query)

    flt = None
    if source_type:
        types_ = [source_type] if isinstance(source_type, str) else list(source_type)
        flt = {"source_type": types_[0]} if len(types_) == 1 else {"source_type": {"$in": types_}}

    try:
        result = _get_index().query(
            vector=vector,
            top_k=top_k,
            namespace=namespace or "",
            include_metadata=True,
            filter=flt,
        )
    except Exception:
        logger.exception("Pinecone query failed (namespace=%s)", namespace or "(default)")
        raise

    matches = []
    for m in result.get("matches", []):
        md = m.get("metadata", {})
        matches.append(
            {
                "score": m.get("score", 0.0),
                "text": md.get("text", ""),
                "source_id": md.get("source_id"),
                "source_type": md.get("source_type", ""),
                "source_url": md.get("source_url", ""),
                "title": md.get("title"),
                "start_time": md.get("start_time"),
                "end_time": md.get("end_time"),
                "page_number": md.get("page_number"),
                "chunk_index": md.get("chunk_index"),
            }
        )
    logger.info("Query returned %d match(es)", len(matches))
    return matches