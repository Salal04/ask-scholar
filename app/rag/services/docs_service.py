"""
Document ingestion pipeline:

1. Fetch a document (by URL, or an uploaded file) to a temp path.
   Google Docs share links are rewritten to their plain-text export URL.
   The local doc_id is derived deterministically from the URL (or the
   uploaded content's hash) instead of a random uuid, so re-ingesting the
   same document — including resuming one that was interrupted — reuses
   the same ingest_state.json entry instead of starting fresh every time.
2. Extract text depending on file type (pdf / docx / txt) — per PAGE
   where the format has a real page concept (pdf), so chunks can carry a
   page number for citations. PDFs are extracted in batches of
   DOC_BATCH_PAGES (20) pages at a time rather than the whole document in
   one shot, and each finished batch is checkpointed to ingest_state.json
   immediately — a long document that gets interrupted resumes from the
   next unfinished batch instead of re-extracting pages already done.
3. Chunk the text (page-aware for pdf, whole-document for docx/txt).
4. Delete the local file (always, once no longer needed).

Citation note: page numbers are assigned per-page during extraction and
never recomputed relative to a batch — page 47 is page 47 whether it was
extracted in batch 0 or batch 2, exactly like YouTube segment timestamps
must stay absolute to the full video (see youtube_service.py).
"""
import hashlib
import logging
import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import httpx
import pdfplumber
from docx import Document as DocxDocument

from .. import config
from . import ingest_state
from .chunking import chunk_pages, chunk_plain_text

logger = logging.getLogger(__name__)


def _rewrite_google_doc_url(url: str) -> str:
    match = re.search(r"/document/d/([a-zA-Z0-9_-]+)", url)
    if match:
        doc_id = match.group(1)
        rewritten = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
        logger.info("Rewrote Google Docs URL %s -> %s", url, rewritten)
        return rewritten
    return url


def _guess_extension(url: str, content_type: str) -> str:
    if url.lower().endswith(".pdf") or "pdf" in content_type:
        return ".pdf"
    if url.lower().endswith(".docx") or "wordprocessingml" in content_type:
        return ".docx"
    return ".txt"


def _deterministic_doc_id(*, url: Optional[str] = None, content: Optional[bytes] = None) -> str:
    """
    A stable id for the same document across runs, so resumable state
    actually resumes instead of every retry getting a fresh random id
    (which would silently defeat ingest_state.json entirely).
    - URL source: hash of the URL itself.
    - Upload source: hash of the file's own bytes, so re-uploading the
      identical file resumes too, while a genuinely different file (even
      with the same filename) gets a different id.
    """
    if url:
        basis = f"url:{url}".encode("utf-8")
    else:
        basis = b"content:" + (content or b"")
    return hashlib.sha256(basis).hexdigest()[:16]


def download_document(url: str) -> Path:
    fetch_url = _rewrite_google_doc_url(url)
    doc_id = _deterministic_doc_id(url=url)

    logger.info("Downloading document from %s (doc_id=%s)", fetch_url, doc_id)
    try:
        with httpx.Client(follow_redirects=True, timeout=60) as client:
            resp = client.get(fetch_url)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            ext = _guess_extension(url, content_type)
            out_path = config.TMP_DIR / f"{doc_id}{ext}"
            out_path.write_bytes(resp.content)
    except httpx.HTTPError:
        logger.exception("Failed to download document from %s", fetch_url)
        raise

    logger.info("Downloaded document %s -> %s (%d bytes)", url, out_path, len(resp.content))
    return out_path, doc_id


def save_uploaded_file(filename: str, content: bytes) -> Path:
    ext = Path(filename).suffix.lower() or ".txt"
    doc_id = _deterministic_doc_id(content=content)
    out_path = config.TMP_DIR / f"{doc_id}{ext}"
    out_path.write_bytes(content)
    logger.info("Saved uploaded file %s -> %s (%d bytes, doc_id=%s)", filename, out_path, len(content), doc_id)
    return out_path, doc_id


def _pdf_page_count(path: Path) -> int:
    with pdfplumber.open(path) as pdf:
        return len(pdf.pages)


def _extract_pdf_page_range(path: Path, start_index: int, end_index: int) -> List[str]:
    """Extracts text for pages [start_index, end_index) (0-based, half-open)."""
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages[start_index:end_index]:
            pages.append(page.extract_text() or "")
    return pages


def extract_pdf_pages_batched(path: Path, doc_id: str, batch_size: int) -> List[str]:
    """
    Extracts a PDF's text batch_size pages at a time instead of loading
    the whole document at once, checkpointing each finished batch to
    ingest_state.json so an interrupted extraction resumes from the next
    unfinished batch rather than starting over from page 1.
    Returns the full list of per-page text (index 0 = page 1), in order.
    """
    total_pages = _pdf_page_count(path)
    ingest_state.init_document_state(
        doc_id=doc_id, source=str(path), local_path=str(path),
        batch_size_pages=batch_size, total_pages=total_pages,
    )

    state = ingest_state.get_document_state(doc_id) or {}
    completed_batches = state.get("batches", {})

    all_pages: List[str] = []
    num_batches = max(1, (total_pages + batch_size - 1) // batch_size)
    for batch_idx in range(num_batches):
        key = str(batch_idx)
        if key in completed_batches:
            logger.info(
                "doc_id=%s batch=%d already extracted (resuming interrupted job) — reusing it",
                doc_id, batch_idx,
            )
            all_pages.extend(completed_batches[key])
            continue

        start = batch_idx * batch_size
        end = min(start + batch_size, total_pages)
        logger.info("Extracting doc_id=%s pages %d-%d of %d", doc_id, start + 1, end, total_pages)
        batch_pages = _extract_pdf_page_range(path, start, end)
        ingest_state.save_document_batch(doc_id, batch_idx, batch_pages)
        all_pages.extend(batch_pages)

    ingest_state.mark_document_done(doc_id)
    ingest_state.clear_document_state(doc_id)  # fully done, nothing left to resume
    return all_pages


def extract_pages(path: Path, doc_id: Optional[str] = None) -> List[str]:
    """
    Returns the document's text as a list of per-page strings (index 0 =
    page 1). Only PDFs have a real page concept:
    - .docx / .txt / exported google doc: treated as a single "page 1" —
      python-docx doesn't expose pagination (that's a rendering-time
      concept in Word, not stored in the file), so page-level citations
      aren't possible for those formats; the document's name is enough
      there, same as before.

    PDFs are extracted in DOC_BATCH_PAGES-page batches (see
    extract_pdf_pages_batched) rather than all at once.
    """
    ext = path.suffix.lower()
    logger.info("Extracting text from %s (ext=%s)", path, ext)

    if ext == ".pdf":
        try:
            pages = extract_pdf_pages_batched(path, doc_id or path.stem, config.DOC_BATCH_PAGES)
        except Exception:
            logger.exception("Failed to extract text from PDF %s", path)
            raise
        logger.info("Extracted %d page(s) from PDF %s", len(pages), path)
        return pages

    if ext == ".docx":
        try:
            doc = DocxDocument(str(path))
            text = ["\n".join(p.text for p in doc.paragraphs)]
        except Exception:
            logger.exception("Failed to extract text from DOCX %s", path)
            raise
        logger.info("Extracted %d char(s) from DOCX %s", len(text[0]), path)
        return text

    # plain text / txt / exported google doc
    text = path.read_text(encoding="utf-8", errors="ignore")
    logger.info("Read %d char(s) from text file %s", len(text), path)
    return [text]


def ingest_document(
    url: Optional[str] = None,
    filename: Optional[str] = None,
    content: Optional[bytes] = None,
    title: Optional[str] = None,
) -> Dict:
    """
    Provide either `url`, or (`filename` + `content`) for a direct upload.
    The downloaded/saved file is deleted once ingestion succeeds. On
    failure it is deliberately LEFT ON DISK (unlike the old always-clean-up
    behavior): if extraction was interrupted partway through a long
    document, the next attempt needs that file to resume from the last
    completed batch instead of re-downloading and starting at page 1
    again. get_document_state() prunes any state entry whose file has
    genuinely disappeared, so this never leaves permanently-stale state.
    """
    logger.info("Starting document ingestion (url=%s, filename=%s)", url, filename)
    path = None
    try:
        if url:
            path, doc_id = download_document(url)
            source_url = url
        elif filename and content is not None:
            path, doc_id = save_uploaded_file(filename, content)
            source_url = filename
        else:
            raise ValueError("Must provide either `url` or `filename`+`content`.")

        pages = extract_pages(path, doc_id=doc_id)
        is_paginated = path.suffix.lower() == ".pdf"

        if is_paginated:
            # Chunk per-page so every chunk carries the exact page number
            # it came from (askQuestion cites "<document>, p. <n>").
            page_chunks = chunk_pages(pages)
            chunks = [
                {"text": c["text"], "chunk_index": i, "page_number": c["page_number"]}
                for i, c in enumerate(page_chunks)
            ]
        else:
            # docx/txt have no real page concept — chunk the whole
            # document's text as before, no page_number attached.
            chunks_text = chunk_plain_text(pages[0] if pages else "")
            chunks = [{"text": t, "chunk_index": i} for i, t in enumerate(chunks_text)]

        logger.info(
            "Ingestion complete for doc_id=%s source_url=%s: %d chunk(s), paginated=%s",
            doc_id, source_url, len(chunks), is_paginated,
        )

        # Success — the file has served its purpose and any resumable
        # state for it was already cleared inside extract_pdf_pages_batched.
        if path is not None and path.exists():
            logger.info("Cleaning up local document file: %s", path)
            path.unlink(missing_ok=True)

        return {
            "source_id": doc_id,
            "title": title or (Path(source_url).name if source_url else doc_id),
            "source_url": source_url,
            "chunks": chunks,
        }
    except Exception:
        logger.exception(
            "Document ingestion failed (url=%s, filename=%s) — leaving %s on disk "
            "so a retry can resume from the last completed page batch",
            url, filename, path,
        )
        raise
