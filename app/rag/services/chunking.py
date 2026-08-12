"""
Chunking utilities.

Three flavors:
- chunk_plain_text: for documents with no page concept (no timestamps).
- chunk_pages: for documents where we know the per-page text (currently
  PDFs) — chunks each page independently so every chunk can be tagged with
  the exact page number it came from, for "<document>, p. <n>" citations.
- chunk_transcript_segments: for YouTube transcripts, where the Gemini
  transcription step returns [{"start": s, "end": s, "text": t}, ...] and we
  need to group those segments into ~CHUNK_SIZE_CHARS-sized chunks while
  keeping the start/end timestamp range of each chunk.
"""
import logging
from typing import Dict, List

from .. import config

logger = logging.getLogger(__name__)


def chunk_plain_text(
    text: str,
    chunk_size: int = config.CHUNK_SIZE_CHARS,
    overlap: int = config.CHUNK_OVERLAP_CHARS,
) -> List[str]:
    text = " ".join(text.split())  # normalize whitespace
    if not text:
        logger.debug("chunk_plain_text called with empty text, returning no chunks")
        return []

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        # try to break on a sentence/space boundary near the end
        if end < n:
            boundary = text.rfind(" ", start + int(chunk_size * 0.6), end)
            if boundary != -1:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, start + 1)

    logger.debug(
        "chunk_plain_text: %d char(s) -> %d chunk(s) (chunk_size=%d, overlap=%d)",
        n, len(chunks), chunk_size, overlap,
    )
    return chunks


def chunk_pages(
    pages: List[str],
    chunk_size: int = config.CHUNK_SIZE_CHARS,
    overlap: int = config.CHUNK_OVERLAP_CHARS,
) -> List[Dict]:
    """
    pages: text content of each page in order, e.g. pdfplumber's
    per-page extraction (index 0 = page 1).

    Deliberately chunks each page on its own instead of running
    chunk_plain_text over the whole document — a chunk never spans two
    pages, so every chunk keeps one unambiguous page number, at the cost of
    occasional small/short chunks right at a page boundary. That trade-off
    is worth it: page number is the whole point of this function (it's how
    askQuestion cites "<document>, p. <n>" back to the user).

    Returns: [{"text": str, "page_number": int}, ...]
    """
    logger.info("Chunking %d page(s) (chunk_size=%d, overlap=%d)", len(pages), chunk_size, overlap)
    chunks = []
    for page_number, page_text in enumerate(pages, start=1):
        page_chunks = chunk_plain_text(page_text, chunk_size=chunk_size, overlap=overlap)
        logger.debug("Page %d -> %d chunk(s)", page_number, len(page_chunks))
        for text in page_chunks:
            chunks.append({"text": text, "page_number": page_number})

    logger.info("Chunked %d page(s) into %d total chunk(s)", len(pages), len(chunks))
    return chunks


def chunk_transcript_segments(
    segments: List[Dict],
    chunk_size: int = config.CHUNK_SIZE_CHARS,
) -> List[Dict]:
    """
    segments: [{"start": float, "end": float, "text": str}, ...] in order.
    Returns: [{"text": str, "start_time": float, "end_time": float}, ...]
    """
    logger.info("Chunking %d transcript segment(s) (chunk_size=%d)", len(segments), chunk_size)

    chunks = []
    buf_text = []
    buf_start = None
    buf_end = None
    buf_len = 0

    def flush():
        nonlocal buf_text, buf_start, buf_end, buf_len
        if buf_text:
            chunks.append(
                {
                    "text": " ".join(buf_text).strip(),
                    "start_time": buf_start,
                    "end_time": buf_end,
                }
            )
            logger.debug(
                "Flushed chunk %d: %.1fs-%.1fs, %d char(s)",
                len(chunks), buf_start or 0.0, buf_end or 0.0, buf_len,
            )
        buf_text, buf_start, buf_end, buf_len = [], None, None, 0

    skipped_empty = 0
    for seg in segments:
        seg_text = (seg.get("text") or "").strip()
        if not seg_text:
            skipped_empty += 1
            continue
        if buf_start is None:
            buf_start = seg.get("start", 0.0)
        buf_end = seg.get("end", buf_start)

        if buf_len + len(seg_text) > chunk_size and buf_text:
            flush()
            buf_start = seg.get("start", 0.0)
            buf_end = seg.get("end", buf_start)

        buf_text.append(seg_text)
        buf_len += len(seg_text) + 1

    flush()

    if skipped_empty:
        logger.debug("Skipped %d empty segment(s) while chunking", skipped_empty)
    logger.info("Chunked %d segment(s) into %d chunk(s)", len(segments), len(chunks))
    return chunks