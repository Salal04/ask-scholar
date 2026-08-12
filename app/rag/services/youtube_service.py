"""
YouTube ingestion pipeline:

1. Download only the audio track of the video (yt_dlp, with fallback
   strategies if the default extraction fails).
2. Split the audio into ~VIDEO_SEGMENT_SECONDS (20 min) pieces — long
   videos were previously handed to Gemini as a single audio file, which
   is exactly what made long-video transcription slow/flaky. Each piece
   is transcribed on its own, then converted back onto the full video's
   absolute timeline (see the citation note below).
3. Every finished segment is checkpointed to ingest_state.json
   (services/ingest_state.py) immediately, so a crash/restart mid-video
   resumes from the next unfinished segment instead of re-downloading or
   re-transcribing everything already done. A fully-finished video is
   reused entirely from cache — no re-download, no Gemini calls at all.
4. Chunk the (now absolute-timestamped) segments into ~CHUNK_SIZE_CHARS
   pieces, each tagged with its start/end time and the source video URL.
5. Delete local audio files once no longer needed.

Non-negotiable citation rule: every chunk's start_time/end_time must be
seconds into the ORIGINAL, full-length video — never seconds into a
20-minute slice. A citation pointing at "3:10" inside segment #3 instead
of the correct "43:10" in the actual video is worse than no citation.
Splitting only changes what we hand Gemini at once; timestamps are
converted back to the full video's timeline (offset by the cumulative,
*actually probed* duration of every prior segment, not an assumed fixed
length) before anything is chunked, stored, or handed back to a user.
This file is the only place that conversion happens.
"""
import logging
import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Dict, List

import yt_dlp
from pydantic import BaseModel

from .. import config
from ..gemini_manager import key_manager
from . import ingest_state, json_utils
from .chunking import chunk_transcript_segments

logger = logging.getLogger(__name__)

TRANSCRIBE_PROMPT = """\
Transcribe this audio in full. Return a JSON array (no markdown, no
commentary) of segments in chronological order, each shaped like:
{"start": <seconds as float, relative to the start of THIS audio file>, \
"end": <seconds as float>, "text": "<segment text>"}

Break segments at natural sentence/phrase boundaries roughly every 5-20
seconds. Cover the entire audio from start to finish. Every "start"/"end"
must be a plain number of seconds into this audio file (0 = the very
beginning of this file), never a timestamp string.

Translate each segment's "text" into English while preserving names,
numbers, and important terminology. Do not add commentary or extra info.
"""


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str


def _extract_video_id(url: str) -> str:
    match = re.search(r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{6,})", url)
    return match.group(1) if match else uuid.uuid4().hex[:11]


def _base_ydl_opts(out_template: str) -> dict:
    return {
        "format": "bestaudio/best",
        "outtmpl": out_template,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "128",
            }
        ],
        "quiet": True,
        "noplaylist": True,
    }


def _try_yt_dlp_default(url: str, out_template: str) -> None:
    """Strategy 1: plain yt-dlp with cookies.txt if present."""
    opts = _base_ydl_opts(out_template)
    if os.path.exists("cookies.txt"):
        opts["cookiejar"] = None
        opts["cookiefile"] = "cookies.txt"
    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.extract_info(url, download=True)


def _try_yt_dlp_android(url: str, out_template: str) -> None:
    """Strategy 2: force the android player client — no cookies, since
    android client rejects cookie auth entirely and yt-dlp silently
    falls back to the (broken) default client if cookies are present."""
    opts = _base_ydl_opts(out_template)
    opts["extractor_args"] = {"youtube": {"player_client": ["android"]}}
    # Deliberately no cookiefile here — android client doesn't support it.
    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.extract_info(url, download=True)


def _try_pytubefix(url: str, video_id: str) -> None:
    """Strategy 3: pytubefix as a last-resort fallback (different
    extraction logic than yt-dlp, sometimes survives when yt-dlp is
    mid-breakage from a YouTube change)."""
    from pytubefix import YouTube  # local import: optional dependency

    yt = YouTube(url)
    stream = yt.streams.get_audio_only()
    if stream is None:
        raise RuntimeError("pytubefix: no audio-only stream available")

    downloaded = stream.download(
        output_path=str(config.TMP_DIR), filename=f"{video_id}_pytube.mp4"
    )

    # Normalize to mp3 via ffmpeg so downstream code always finds a .mp3
    mp3_path = config.TMP_DIR / f"{video_id}.mp3"
    os.system(f'ffmpeg -y -i "{downloaded}" -q:a 2 "{mp3_path}"')
    if os.path.exists(downloaded):
        os.remove(downloaded)

    return {"title": yt.title}


def download_audio(url: str) -> Path:
    video_id = _extract_video_id(url)
    out_template = str(config.TMP_DIR / f"{video_id}.%(ext)s")
    audio_path = config.TMP_DIR / f"{video_id}.mp3"

    logger.info("Cookies exist? = %s", os.path.exists("cookies.txt"))

    strategies = [
        ("yt-dlp default", lambda: _try_yt_dlp_default(url, out_template)),
        ("yt-dlp android client", lambda: _try_yt_dlp_android(url, out_template)),
        ("pytubefix", lambda: _try_pytubefix(url, video_id)),
    ]

    last_error = None
    title = video_id
    for name, strategy in strategies:
        try:
            logger.info("Trying download strategy: %s (url=%s)", name, url)
            info = strategy()
            if info and isinstance(info, dict) and info.get("title"):
                title = info.get("title")
            if audio_path.exists():
                logger.info("Success with strategy: %s (url=%s)", name, url)
                return audio_path, title, video_id
            # Strategy ran without raising but produced no file — treat as failure.
            raise FileNotFoundError(f"{name} completed but {audio_path} not found.")
        except Exception as e:
            logger.warning("Strategy '%s' failed for url=%s: %s", name, url, e)
            last_error = e
            continue

    logger.error("All download strategies failed for %s. Last error: %s", url, last_error)
    raise RuntimeError(
        f"All download strategies failed for {url}. Last error: {last_error}"
    )


def _probe_duration_seconds(path: Path) -> float:
    """ffprobe the media duration in seconds. Returns 0.0 if ffprobe is
    unavailable or fails — callers treat that as "couldn't determine
    duration, don't split"."""
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True, text=True, timeout=30,
        )
        return float(out.stdout.strip())
    except Exception:
        logger.warning("ffprobe failed for %s, will not split into segments", path, exc_info=True)
        return 0.0


def _segment_dir(video_id: str) -> Path:
    return config.TMP_DIR / f"{video_id}_segments"


def _segment_path(video_id: str, index: int) -> Path:
    return _segment_dir(video_id) / f"{video_id}_{index:03d}.mp3"


def _split_audio_into_segments(audio_path: Path, video_id: str, segment_seconds: int) -> int:
    """
    Splits audio_path into consecutive ~segment_seconds mp3 files under
    _segment_dir(video_id), named so _segment_path() can find each one by
    index. Returns the number of segments produced (always >= 1).
    """
    seg_dir = _segment_dir(video_id)
    seg_dir.mkdir(exist_ok=True)
    out_template = str(seg_dir / f"{video_id}_%03d.mp3")

    cmd = [
        "ffmpeg", "-y", "-i", str(audio_path),
        "-f", "segment", "-segment_time", str(segment_seconds),
        "-c", "copy", "-reset_timestamps", "1",
        out_template,
    ]
    logger.info("Splitting %s into ~%ds segments -> %s", audio_path, segment_seconds, seg_dir)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        logger.error("ffmpeg segment split failed for %s: %s", audio_path, result.stderr[-2000:])
        raise RuntimeError(f"ffmpeg segment split failed for {audio_path}: {result.stderr[-500:]}")

    produced = sorted(seg_dir.glob(f"{video_id}_*.mp3"))
    if not produced:
        raise RuntimeError(f"ffmpeg segment split produced no files for {audio_path}")
    logger.info("Split %s into %d segment(s)", audio_path, len(produced))
    return len(produced)


def _transcribe_segment_file(audio_path: Path) -> List[Dict]:
    """Uploads one (<= ~20min) audio segment to Gemini and returns its
    transcript as [{"start": float, "end": float, "text": str}, ...],
    with timestamps relative to THIS segment's own start (0-based)."""

    def fn(client, model):
        logger.info("Uploading audio segment to Gemini: %s (model=%s)", audio_path, model)
        uploaded = client.files.upload(file=str(audio_path))
        while getattr(uploaded, "state", None) and uploaded.state.name == "PROCESSING":
            time.sleep(2)
            uploaded = client.files.get(name=uploaded.name)
        logger.info("Gemini file processing complete for %s, requesting transcript", audio_path)

        response = client.models.generate_content(
            model=model,
            contents=[uploaded, TRANSCRIBE_PROMPT],
            # Strict JSON: response_schema constrains generation itself
            # (not just "please format as JSON"), so malformed/half-JSON
            # responses become rare. loads_relaxed() below is the safety
            # net for whatever still slips through.
            config=json_utils.build_json_config(schema=list[TranscriptSegment]),
        )

        parsed = json_utils.extract_parsed(response)
        if parsed is not None:
            segments = [{"start": float(p.start), "end": float(p.end), "text": p.text} for p in parsed]
        else:
            raw = (response.text or "").strip()
            data = json_utils.loads_relaxed(raw)
            if not isinstance(data, list):
                raise json_utils.JSONRepairError("Expected a JSON array of transcript segments.")
            segments = [
                {
                    "start": float(item.get("start", 0.0)),
                    "end": float(item.get("end", 0.0)),
                    "text": (item.get("text") or "").strip(),
                }
                for item in data
            ]

        logger.info("Transcript parsed for %s: %d segment-line(s)", audio_path, len(segments))
        return segments

    return key_manager.call("transcribe", fn)


def transcribe_audio(audio_path: Path) -> List[Dict]:
    """Back-compat single-file entrypoint. Prefer ingest_youtube_video for
    real ingestion — it splits + resumes + offsets timestamps."""
    return _transcribe_segment_file(audio_path)


def _prepare_segments_on_disk(video_id: str, audio_path: Path) -> int:
    """Splits audio_path (if long enough) into segment files, or moves it
    into place as the sole segment 0 otherwise. Always consumes
    audio_path (moved or deleted). Returns the total segment count."""
    duration = _probe_duration_seconds(audio_path)

    if duration > config.VIDEO_SEGMENT_SECONDS:
        total_segments = _split_audio_into_segments(audio_path, video_id, config.VIDEO_SEGMENT_SECONDS)
        # The whole, un-split file is no longer needed once the
        # individual pieces are on disk — each piece is independently
        # resumable, so there's no reason to keep the merged copy.
        audio_path.unlink(missing_ok=True)
        return total_segments

    seg_dir = _segment_dir(video_id)
    seg_dir.mkdir(exist_ok=True)
    shutil.move(str(audio_path), str(_segment_path(video_id, 0)))
    return 1


def _transcribe_all_segments(video_id: str, url: str, title: str, total_segments: int) -> List[Dict]:
    """
    Transcribes every not-yet-completed segment for video_id (reusing
    anything already in ingest_state), then returns one flat list of
    transcript lines with timestamps converted to ABSOLUTE seconds into
    the original full video.
    """
    ingest_state.init_youtube_state(
        video_id=video_id, url=url, title=title,
        audio_path=str(_segment_dir(video_id)),
        segment_seconds=config.VIDEO_SEGMENT_SECONDS,
        total_segments=total_segments,
    )

    state = ingest_state.get_youtube_state(video_id) or {}
    completed = state.get("segments", {})

    cumulative_offset = 0.0
    all_lines: List[Dict] = []

    for idx in range(total_segments):
        key = str(idx)
        if key in completed:
            seg_info = completed[key]
            offset, seg_duration, transcript = seg_info["offset"], seg_info["duration"], seg_info["transcript"]
            logger.info(
                "video_id=%s segment=%d already transcribed (resuming interrupted job) — reusing it",
                video_id, idx,
            )
        else:
            seg_path = _segment_path(video_id, idx)
            if not seg_path.exists():
                raise FileNotFoundError(
                    f"Expected segment file {seg_path} not found on disk — "
                    f"cannot resume video_id={video_id}, restart ingestion for this video."
                )
            seg_duration = _probe_duration_seconds(seg_path) or float(config.VIDEO_SEGMENT_SECONDS)
            offset = cumulative_offset
            transcript = _transcribe_segment_file(seg_path)
            ingest_state.save_youtube_segment(video_id, idx, offset, seg_duration, transcript)
            seg_path.unlink(missing_ok=True)

        cumulative_offset = offset + seg_duration

        # THE non-negotiable step: local -> absolute video time.
        for line in transcript:
            all_lines.append({"start": offset + line["start"], "end": offset + line["end"], "text": line["text"]})

    ingest_state.mark_youtube_done(video_id)
    shutil.rmtree(_segment_dir(video_id), ignore_errors=True)
    ingest_state.clear_youtube_state(video_id)  # fully done, nothing left to resume
    return all_lines


def _cached_transcript_if_done(video_id: str) -> "tuple[str, List[Dict]] | None":
    """If this exact video already finished ingesting in a previous run,
    rebuild its absolute-timestamped transcript straight from
    ingest_state.json — no download, no Gemini calls. Returns
    (title, transcript) or None if there's nothing usable cached."""
    state = ingest_state.get_youtube_state(video_id)
    if not state or state.get("status") != "done" or not state.get("segments"):
        return None

    total_segments = state.get("total_segments", len(state["segments"]))
    transcript: List[Dict] = []
    for idx in range(total_segments):
        seg_info = state["segments"].get(str(idx))
        if seg_info is None:
            return None  # incomplete cache, don't trust it — fall back to a fresh run
        offset = seg_info["offset"]
        for line in seg_info["transcript"]:
            transcript.append({"start": offset + line["start"], "end": offset + line["end"], "text": line["text"]})

    return state.get("title") or video_id, transcript


def ingest_youtube_video(url: str) -> Dict:
    """
    Full pipeline for one video. Returns a dict with the chunks (with
    ABSOLUTE timestamps + source url) ready to be embedded/stored, plus
    metadata. Resumable: an interrupted run picks up from the next
    unfinished ~20-minute segment; a fully-finished video is served
    straight from ingest_state.json with no re-download or re-transcription.
    """
    logger.info("Starting ingestion pipeline for %s", url)
    video_id = _extract_video_id(url)

    try:
        cached = _cached_transcript_if_done(video_id)
        if cached is not None:
            title, transcript = cached
            logger.info(
                "video_id=%s already fully ingested per ingest_state.json — reusing cached transcript",
                video_id,
            )
        else:
            audio_path, title, _ = download_audio(url)
            logger.info("Downloaded audio for %s -> %s (title=%r)", url, audio_path, title)
            total_segments = _prepare_segments_on_disk(video_id, audio_path)
            transcript = _transcribe_all_segments(video_id, url, title, total_segments)

        logger.info("Transcribed %s: %d raw segment-line(s)", url, len(transcript))

        chunks = chunk_transcript_segments(transcript)
        for c in chunks:
            c["source_url"] = url
        logger.info("Chunked %s into %d chunk(s)", url, len(chunks))

        return {
            "source_id": video_id,
            "title": title,
            "source_url": url,
            "chunks": chunks,
        }
    except Exception:
        logger.exception("Ingestion pipeline failed for %s", url)
        raise
