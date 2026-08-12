"""
Resumable ingestion state.

Long videos are split into ~20-minute audio segments and long documents
into ~20-page batches (see youtube_service / docs_service) so we never
hand one giant file to Gemini in a single call. This module is what lets
that splitting actually help: every segment/batch we finish gets written
here immediately, so if the process is killed/restarted mid-video or
mid-document (background task crash, deploy, server restart, ...) the
next run picks up from the next unfinished piece instead of starting the
whole video/document over from scratch.

File layout (ingest_state.json, separate from model_state.json which is
for Gemini model fallback position, not ingestion progress):

{
  "youtube": {
    "<video_id>": {
      "url": "...",
      "title": "...",
      "audio_path": "/abs/path/original/audio.mp3",   # for staleness check
      "segment_seconds": 1200,
      "total_segments": 4,
      "segments": {                                     # completed only
        "0": {"offset": 0.0, "duration": 1198.3, "transcript": [...]},
        "1": {...}
      },
      "status": "in_progress" | "done"
    }
  },
  "document": {
    "<doc_id>": {
      "source": "url-or-filename",
      "local_path": "/abs/path/downloaded/file.pdf",
      "batch_size_pages": 20,
      "total_pages": 55,
      "batches": {                                       # completed only
        "0": ["page 1 text", "page 2 text", ...]
      },
      "status": "in_progress" | "done"
    }
  }
}

Staleness handling: if a source's state says it depends on a local file
(audio_path / local_path) and that file no longer exists on disk when we
go to load the state, the entry is dropped — we have no way to resume a
mid-flight download/extraction without the file, so it's safer to treat
it as "never started" than to trust stale/partial data. This is checked
on every load, not just at startup, since ingestion runs as ad-hoc
background tasks rather than one long-lived process.
"""
import json
import logging
import os
import threading
from typing import Dict, Optional

from .. import config

logger = logging.getLogger(__name__)

STATE_PATH = config.BASE_DIR / "ingest_state.json"

# Ingestion can run several background tasks concurrently (multiple
# scholars' videos ingesting at once); guard read-modify-write of the
# shared JSON file so two segments finishing at the same instant don't
# clobber each other.
_lock = threading.Lock()


def _read() -> Dict:
    if not STATE_PATH.exists():
        return {"youtube": {}, "document": {}}
    try:
        with open(STATE_PATH, "r") as f:
            data = json.load(f)
    except Exception:
        logger.exception("Failed to read %s, starting from empty state", STATE_PATH)
        return {"youtube": {}, "document": {}}
    data.setdefault("youtube", {})
    data.setdefault("document", {})
    return data


def _write(data: Dict) -> None:
    tmp_path = str(STATE_PATH) + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, STATE_PATH)  # atomic swap, avoids a torn file on crash


def _prune_stale(data: Dict, section: str, path_field: str) -> bool:
    """Drops any entry in data[section] whose referenced local file is
    gone. Returns True if anything changed."""
    changed = False
    for key in list(data.get(section, {}).keys()):
        entry = data[section][key]
        path = entry.get(path_field)
        if entry.get("status") != "done" and path and not os.path.exists(path):
            logger.warning(
                "Dropping stale %s ingestion state for %s: %s no longer exists on "
                "disk (probably deleted) — will restart from scratch.",
                section, key, path,
            )
            del data[section][key]
            changed = True
    return changed


def get_youtube_state(video_id: str) -> Optional[Dict]:
    with _lock:
        data = _read()
        if _prune_stale(data, "youtube", "audio_path"):
            _write(data)
        return data["youtube"].get(video_id)


def init_youtube_state(video_id: str, url: str, title: str, audio_path: str,
                        segment_seconds: int, total_segments: int) -> None:
    with _lock:
        data = _read()
        existing = data["youtube"].get(video_id, {})
        data["youtube"][video_id] = {
            **existing,
            "url": url,
            "title": title,
            "audio_path": audio_path,
            "segment_seconds": segment_seconds,
            "total_segments": total_segments,
            "segments": existing.get("segments", {}),
            "status": "in_progress",
        }
        _write(data)


def save_youtube_segment(video_id: str, segment_index: int, offset: float,
                          duration: float, transcript: list) -> None:
    with _lock:
        data = _read()
        entry = data["youtube"].setdefault(video_id, {"segments": {}, "status": "in_progress"})
        entry.setdefault("segments", {})[str(segment_index)] = {
            "offset": offset,
            "duration": duration,
            "transcript": transcript,
        }
        _write(data)
        logger.info(
            "Saved resumable state for youtube video_id=%s segment=%d "
            "(offset=%.1fs, %d segment-lines)",
            video_id, segment_index, offset, len(transcript),
        )


def mark_youtube_done(video_id: str) -> None:
    with _lock:
        data = _read()
        if video_id in data["youtube"]:
            data["youtube"][video_id]["status"] = "done"
            _write(data)


def clear_youtube_state(video_id: str) -> None:
    with _lock:
        data = _read()
        if data["youtube"].pop(video_id, None) is not None:
            _write(data)


def get_document_state(doc_id: str) -> Optional[Dict]:
    with _lock:
        data = _read()
        if _prune_stale(data, "document", "local_path"):
            _write(data)
        return data["document"].get(doc_id)


def init_document_state(doc_id: str, source: str, local_path: str,
                         batch_size_pages: int, total_pages: int) -> None:
    with _lock:
        data = _read()
        existing = data["document"].get(doc_id, {})
        data["document"][doc_id] = {
            **existing,
            "source": source,
            "local_path": local_path,
            "batch_size_pages": batch_size_pages,
            "total_pages": total_pages,
            "batches": existing.get("batches", {}),
            "status": "in_progress",
        }
        _write(data)


def save_document_batch(doc_id: str, batch_index: int, pages_text: list) -> None:
    with _lock:
        data = _read()
        entry = data["document"].setdefault(doc_id, {"batches": {}, "status": "in_progress"})
        entry.setdefault("batches", {})[str(batch_index)] = pages_text
        _write(data)
        logger.info(
            "Saved resumable state for document doc_id=%s batch=%d (%d page(s))",
            doc_id, batch_index, len(pages_text),
        )


def mark_document_done(doc_id: str) -> None:
    with _lock:
        data = _read()
        if doc_id in data["document"]:
            data["document"][doc_id]["status"] = "done"
            _write(data)


def clear_document_state(doc_id: str) -> None:
    with _lock:
        data = _read()
        if data["document"].pop(doc_id, None) is not None:
            _write(data)
